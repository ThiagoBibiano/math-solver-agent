"""Runtime controls for admission, job lifecycle and lightweight telemetry."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_provider(value: Optional[str]) -> str:
    provider = str(value or "unknown").strip().lower()
    return provider or "unknown"


class AdmissionRejectedError(RuntimeError):
    """Raised when the admission controller cannot accept a request."""

    def __init__(self, message: str, reason: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.reason = reason
        self.retry_after_seconds = max(1, int(retry_after_seconds))


@dataclass
class AdmissionTicket:
    provider: str
    queued_at_monotonic: float
    acquired_at_monotonic: float
    queue_position: int
    released: bool = False


class AdmissionController:
    """Controls in-flight concurrency and bounded waiting queue."""

    def __init__(
        self,
        max_inflight_global: int = 4,
        max_inflight_by_provider: Optional[Dict[str, int]] = None,
        max_queue_size: int = 32,
        queue_wait_timeout_seconds: float = 2.0,
    ) -> None:
        self.max_inflight_global = max(1, int(max_inflight_global))
        self.max_inflight_by_provider = {
            _normalize_provider(key): max(1, int(value))
            for key, value in dict(max_inflight_by_provider or {}).items()
        }
        self.max_queue_size = max(1, int(max_queue_size))
        self.queue_wait_timeout_seconds = max(0.1, float(queue_wait_timeout_seconds))

        self._condition = asyncio.Condition()
        self._inflight_total = 0
        self._inflight_by_provider: Dict[str, int] = {}
        self._waiting = 0

    def _provider_limit(self, provider: str) -> int:
        return int(self.max_inflight_by_provider.get(provider, self.max_inflight_global))

    def _can_admit_locked(self, provider: str) -> bool:
        if self._inflight_total >= self.max_inflight_global:
            return False
        return self._inflight_by_provider.get(provider, 0) < self._provider_limit(provider)

    async def acquire(self, provider: Optional[str]) -> AdmissionTicket:
        normalized_provider = _normalize_provider(provider)
        queued_at = time.monotonic()

        async with self._condition:
            if (not self._can_admit_locked(normalized_provider)) and self._waiting >= self.max_queue_size:
                raise AdmissionRejectedError(
                    message="Runtime queue is full.",
                    reason="queue_full",
                    retry_after_seconds=max(1, int(self.queue_wait_timeout_seconds)),
                )

            self._waiting += 1
            queue_position = self._waiting
            admitted = False
            try:
                while not self._can_admit_locked(normalized_provider):
                    elapsed = time.monotonic() - queued_at
                    remaining = self.queue_wait_timeout_seconds - elapsed
                    if remaining <= 0:
                        raise AdmissionRejectedError(
                            message="Runtime queue wait timeout exceeded.",
                            reason="queue_wait_timeout",
                            retry_after_seconds=max(1, int(self.queue_wait_timeout_seconds)),
                        )
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError as exc:
                        raise AdmissionRejectedError(
                            message="Runtime queue wait timeout exceeded.",
                            reason="queue_wait_timeout",
                            retry_after_seconds=max(1, int(self.queue_wait_timeout_seconds)),
                        ) from exc

                admitted = True
                self._waiting = max(0, self._waiting - 1)
                self._inflight_total += 1
                self._inflight_by_provider[normalized_provider] = self._inflight_by_provider.get(normalized_provider, 0) + 1
                acquired_at = time.monotonic()
                return AdmissionTicket(
                    provider=normalized_provider,
                    queued_at_monotonic=queued_at,
                    acquired_at_monotonic=acquired_at,
                    queue_position=queue_position,
                )
            finally:
                if not admitted:
                    self._waiting = max(0, self._waiting - 1)
                    self._condition.notify_all()

    async def release(self, ticket: AdmissionTicket) -> None:
        if ticket.released:
            return
        async with self._condition:
            if ticket.released:
                return
            ticket.released = True
            provider = _normalize_provider(ticket.provider)
            if self._inflight_total > 0:
                self._inflight_total -= 1
            if provider in self._inflight_by_provider:
                current = self._inflight_by_provider.get(provider, 0) - 1
                if current <= 0:
                    self._inflight_by_provider.pop(provider, None)
                else:
                    self._inflight_by_provider[provider] = current
            self._condition.notify_all()

    async def snapshot(self) -> Dict[str, Any]:
        async with self._condition:
            in_flight_by_provider = dict(self._inflight_by_provider)
            busy = self._inflight_total >= self.max_inflight_global or self._waiting > 0
            return {
                "busy": busy,
                "in_flight_total": self._inflight_total,
                "queue_depth": self._waiting,
                "queue_capacity": self.max_queue_size,
                "in_flight_by_provider": in_flight_by_provider,
                "suggested_retry_after_seconds": max(1, int(self.queue_wait_timeout_seconds)),
            }


class RuntimeStats:
    """Small in-memory metrics window for runtime visibility."""

    def __init__(self, window_seconds: int = 300) -> None:
        self.window_seconds = max(30, int(window_seconds))
        self._events: Deque[Dict[str, Any]] = deque()
        self._lock = threading.Lock()

    def _purge_locked(self, now_monotonic: float) -> None:
        threshold = now_monotonic - float(self.window_seconds)
        while self._events and self._events[0]["ts"] < threshold:
            self._events.popleft()

    def record(self, provider: Optional[str], latency_ms: float, timed_out: bool = False) -> None:
        now_mono = time.monotonic()
        payload = {
            "ts": now_mono,
            "provider": _normalize_provider(provider),
            "latency_ms": max(0.0, float(latency_ms)),
            "timed_out": bool(timed_out),
        }
        with self._lock:
            self._events.append(payload)
            self._purge_locked(now_mono)

    def snapshot(self) -> Dict[str, Any]:
        now_mono = time.monotonic()
        with self._lock:
            self._purge_locked(now_mono)
            events = list(self._events)

        if not events:
            return {"timeouts_last_5m_by_provider": {}, "avg_latency_ms_last_5m": 0.0}

        timeout_counts: Dict[str, int] = {}
        latencies = []
        for item in events:
            provider = str(item.get("provider", "unknown"))
            if bool(item.get("timed_out")):
                timeout_counts[provider] = timeout_counts.get(provider, 0) + 1
            else:
                latencies.append(float(item.get("latency_ms", 0.0)))
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        return {
            "timeouts_last_5m_by_provider": timeout_counts,
            "avg_latency_ms_last_5m": round(avg_latency, 2),
        }


class JobQueueFullError(RuntimeError):
    """Raised when in-memory job queue has reached capacity."""


class JobNotFoundError(KeyError):
    """Raised when requested job id does not exist."""


class InMemoryJobStore:
    """Single-process in-memory store with bounded queue and TTL cleanup."""

    TERMINAL_STATUSES = {"succeeded", "failed", "timeout", "canceled"}

    def __init__(self, max_queue_size: int = 32, retention_seconds: int = 1800) -> None:
        self.max_queue_size = max(1, int(max_queue_size))
        self.retention_seconds = max(60, int(retention_seconds))

        self._condition = asyncio.Condition()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._queue: Deque[str] = deque()

    def _cleanup_locked(self) -> None:
        now_mono = time.monotonic()
        stale = []
        for job_id, payload in self._jobs.items():
            if payload.get("status") not in self.TERMINAL_STATUSES:
                continue
            finished_mono = payload.get("finished_monotonic")
            if finished_mono is None:
                continue
            if now_mono - float(finished_mono) > float(self.retention_seconds):
                stale.append(job_id)
        for job_id in stale:
            self._jobs.pop(job_id, None)

    def _queue_position_locked(self, job_id: str) -> Optional[int]:
        try:
            return list(self._queue).index(job_id) + 1
        except ValueError:
            return None

    def _public_locked(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = str(payload["job_id"])
        queue_position = self._queue_position_locked(job_id) if payload.get("status") == "queued" else None
        return {
            "job_id": job_id,
            "status": str(payload.get("status", "queued")),
            "queue_position": queue_position,
            "submitted_at": payload.get("submitted_at"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "error": payload.get("error"),
            "result": payload.get("result"),
            "provider": payload.get("provider"),
            "cancel_requested": bool(payload.get("cancel_requested", False)),
        }

    async def submit(self, payload: Dict[str, Any], provider: Optional[str]) -> Dict[str, Any]:
        async with self._condition:
            self._cleanup_locked()
            if len(self._queue) >= self.max_queue_size:
                raise JobQueueFullError("Job queue capacity reached.")
            job_id = uuid.uuid4().hex
            record = {
                "job_id": job_id,
                "status": "queued",
                "payload": dict(payload),
                "provider": _normalize_provider(provider),
                "submitted_at": _utc_now_iso(),
                "started_at": None,
                "finished_at": None,
                "finished_monotonic": None,
                "error": None,
                "result": None,
                "cancel_requested": False,
            }
            self._jobs[job_id] = record
            self._queue.append(job_id)
            self._condition.notify_all()
            return self._public_locked(record)

    async def pop_next(self) -> Dict[str, Any]:
        async with self._condition:
            while True:
                self._cleanup_locked()
                while not self._queue:
                    await self._condition.wait()
                job_id = self._queue.popleft()
                record = self._jobs.get(job_id)
                if not record:
                    continue
                if record.get("status") == "canceled":
                    continue
                record["status"] = "running"
                record["started_at"] = _utc_now_iso()
                return dict(record)

    async def get(self, job_id: str) -> Dict[str, Any]:
        async with self._condition:
            self._cleanup_locked()
            record = self._jobs.get(str(job_id))
            if not record:
                raise JobNotFoundError(job_id)
            return self._public_locked(record)

    async def set_terminal(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_status = str(status or "failed").strip().lower()
        if normalized_status not in self.TERMINAL_STATUSES:
            normalized_status = "failed"
        async with self._condition:
            self._cleanup_locked()
            record = self._jobs.get(str(job_id))
            if not record:
                raise JobNotFoundError(job_id)
            if record.get("cancel_requested"):
                normalized_status = "canceled"
                result = None
                error = error or "Job canceled by user."
            record["status"] = normalized_status
            record["result"] = result
            record["error"] = error
            record["finished_at"] = _utc_now_iso()
            record["finished_monotonic"] = time.monotonic()
            return self._public_locked(record)

    async def cancel(self, job_id: str) -> Dict[str, Any]:
        async with self._condition:
            self._cleanup_locked()
            record = self._jobs.get(str(job_id))
            if not record:
                raise JobNotFoundError(job_id)
            status = str(record.get("status", "queued"))
            if status in self.TERMINAL_STATUSES:
                return self._public_locked(record)
            if status == "queued":
                record["status"] = "canceled"
                record["error"] = "Job canceled by user."
                record["finished_at"] = _utc_now_iso()
                record["finished_monotonic"] = time.monotonic()
                try:
                    self._queue.remove(str(job_id))
                except ValueError:
                    pass
                return self._public_locked(record)
            record["cancel_requested"] = True
            return self._public_locked(record)

    async def queue_depth(self) -> int:
        async with self._condition:
            self._cleanup_locked()
            return len(self._queue)

