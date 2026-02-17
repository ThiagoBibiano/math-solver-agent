import unittest

from src.api.runtime import InMemoryJobStore


class JobStoreTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_submit_pop_and_complete(self) -> None:
        store = InMemoryJobStore(max_queue_size=4, retention_seconds=120)
        submitted = await store.submit(payload={"problem": "2+2"}, provider="nvidia")
        self.assertEqual(submitted["status"], "queued")
        self.assertEqual(submitted["queue_position"], 1)

        running = await store.pop_next()
        self.assertEqual(running["status"], "running")

        done = await store.set_terminal(job_id=running["job_id"], status="succeeded", result={"result": "4"})
        self.assertEqual(done["status"], "succeeded")
        fetched = await store.get(running["job_id"])
        self.assertEqual(fetched["status"], "succeeded")
        self.assertEqual(fetched["result"]["result"], "4")

    async def test_cancel_queued_job(self) -> None:
        store = InMemoryJobStore(max_queue_size=4, retention_seconds=120)
        submitted = await store.submit(payload={"problem": "x"}, provider="nvidia")
        canceled = await store.cancel(submitted["job_id"])
        self.assertEqual(canceled["status"], "canceled")

    async def test_cancel_running_marks_request(self) -> None:
        store = InMemoryJobStore(max_queue_size=4, retention_seconds=120)
        submitted = await store.submit(payload={"problem": "x"}, provider="nvidia")
        running = await store.pop_next()
        self.assertEqual(running["job_id"], submitted["job_id"])

        canceled = await store.cancel(submitted["job_id"])
        self.assertEqual(canceled["status"], "running")
        self.assertTrue(canceled["cancel_requested"])

        final = await store.set_terminal(job_id=submitted["job_id"], status="succeeded", result={"x": 1})
        self.assertEqual(final["status"], "canceled")


if __name__ == "__main__":
    unittest.main()
