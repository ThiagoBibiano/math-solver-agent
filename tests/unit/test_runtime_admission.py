import asyncio
import unittest

from src.api.runtime import AdmissionController, AdmissionRejectedError


class RuntimeAdmissionTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_acquire_release_and_snapshot(self) -> None:
        controller = AdmissionController(
            max_inflight_global=2,
            max_inflight_by_provider={"nvidia": 1},
            max_queue_size=4,
            queue_wait_timeout_seconds=0.5,
        )
        first = await controller.acquire("nvidia")
        snapshot = await controller.snapshot()
        self.assertEqual(snapshot["in_flight_total"], 1)
        self.assertEqual(snapshot["in_flight_by_provider"].get("nvidia"), 1)
        await controller.release(first)
        snapshot_after = await controller.snapshot()
        self.assertEqual(snapshot_after["in_flight_total"], 0)

    async def test_queue_timeout(self) -> None:
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

    async def test_queue_full_rejects_fast(self) -> None:
        controller = AdmissionController(
            max_inflight_global=1,
            max_inflight_by_provider={"nvidia": 1},
            max_queue_size=1,
            queue_wait_timeout_seconds=0.5,
        )
        first = await controller.acquire("nvidia")
        second_task = asyncio.create_task(controller.acquire("nvidia"))
        await asyncio.sleep(0.01)
        with self.assertRaises(AdmissionRejectedError):
            await controller.acquire("nvidia")
        await controller.release(first)
        second = await second_task
        await controller.release(second)


if __name__ == "__main__":
    unittest.main()
