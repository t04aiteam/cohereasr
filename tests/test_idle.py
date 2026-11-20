import asyncio
import threading
import unittest

from models.idle import IdleModel


async def wait_until(check):
    async with asyncio.timeout(1):
        while not check():
            await asyncio.sleep(0.005)


class IdleModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_unload_and_reload_once_for_two_requests(self):
        calls = []
        model = IdleModel(lambda: calls.append("load"), lambda: calls.append("unload"), 0.02)
        async with model.lifespan():
            await wait_until(lambda: calls == ["load", "unload"])
            values = await asyncio.gather(model.run(lambda: 1), model.run(lambda: 2))
            self.assertEqual(values, [1, 2])
            self.assertEqual(calls, ["load", "unload", "load"])

    async def test_cancelled_wait_and_queued_work_prevent_unload(self):
        calls = []
        started = threading.Event()
        finish = threading.Event()

        def infer():
            started.set()
            if not finish.wait(2):
                raise TimeoutError("Test inference was not released")
            return 1

        model = IdleModel(lambda: calls.append("load"), lambda: calls.append("unload"), 0.02,
                          serialize=True)
        async with model.lifespan():
            first = asyncio.create_task(model.run(infer))
            try:
                await wait_until(started.is_set)
                second = asyncio.create_task(model.run(lambda: 2))
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                await asyncio.sleep(0.08)
                self.assertFalse(second.done())
                self.assertEqual(calls, ["load"])
            finally:
                finish.set()
            self.assertEqual(await second, 2)
            await wait_until(lambda: calls == ["load", "unload"])

    async def test_failed_reload_can_be_retried(self):
        calls = []

        def load():
            calls.append("load")
            if len(calls) == 3:
                raise RuntimeError("failed load")

        model = IdleModel(load, lambda: calls.append("unload"), 0.02)
        async with model.lifespan():
            await wait_until(lambda: calls == ["load", "unload"])
            with self.assertRaisesRegex(RuntimeError, "failed load"):
                await model.run(lambda: None)
            self.assertEqual(await model.run(lambda: 3), 3)
            self.assertEqual(calls, ["load", "unload", "load", "load"])
