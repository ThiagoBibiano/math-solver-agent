import unittest

from src.api.runtime import AdmissionController, InMemoryJobStore, RuntimeStats


class BusyVisibilityIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_snapshot_reflects_busy_and_queue(self) -> None:
        admission = AdmissionController(
            max_inflight_global=1,
            max_inflight_by_provider={"nvidia": 1},
            max_queue_size=4,
            queue_wait_timeout_seconds=1.0,
        )
        store = InMemoryJobStore(max_queue_size=2, retention_seconds=120)
        ticket = await admission.acquire("nvidia")
        await store.submit(payload={"problem": "2+2"}, provider="nvidia")

        snapshot = await admission.snapshot()
        self.assertTrue(snapshot["busy"])
        self.assertEqual(snapshot["in_flight_total"], 1)
        self.assertEqual(await store.queue_depth(), 1)

        await admission.release(ticket)

    async def test_runtime_stats_exposes_timeout_counts(self) -> None:
        stats = RuntimeStats(window_seconds=300)
        stats.record(provider="nvidia", latency_ms=10.0, timed_out=False)
        stats.record(provider="nvidia", latency_ms=120000.0, timed_out=True)
        stats.record(provider="maritaca", latency_ms=20.0, timed_out=False)
        payload = stats.snapshot()
        self.assertGreaterEqual(payload["avg_latency_ms_last_5m"], 10.0)
        self.assertEqual(payload["timeouts_last_5m_by_provider"].get("nvidia"), 1)


if __name__ == "__main__":
    unittest.main()
