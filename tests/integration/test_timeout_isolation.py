import asyncio
import time
import unittest

from src.api.runtime import AdmissionController, AdmissionRejectedError


class TimeoutIsolationIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_slow_provider_does_not_block_other_provider_slot(self) -> None:
        controller = AdmissionController(
            max_inflight_global=2,
            max_inflight_by_provider={"nvidia": 1, "maritaca": 1},
            max_queue_size=4,
            queue_wait_timeout_seconds=1.0,
        )

        nvidia_ticket = await controller.acquire("nvidia")

        started = time.perf_counter()
        maritaca_ticket = await controller.acquire("maritaca")
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self.assertLess(elapsed_ms, 150.0)

        await controller.release(maritaca_ticket)
        await controller.release(nvidia_ticket)

    async def test_queue_wait_timeout_is_enforced(self) -> None:
        controller = AdmissionController(
            max_inflight_global=1,
            max_inflight_by_provider={"nvidia": 1},
            max_queue_size=2,
            queue_wait_timeout_seconds=0.05,
        )
        ticket = await controller.acquire("nvidia")
        with self.assertRaises(AdmissionRejectedError):
            await controller.acquire("nvidia")
        await controller.release(ticket)


if __name__ == "__main__":
    unittest.main()
