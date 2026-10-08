"""Gate tests: main.py takes DEVICE and PORT from the environment.

c09 runs the ASR on its GPU and on port 8000; 222 keeps CPU and 6221. Both
must come from the environment, and a missing value must keep the 222
defaults. The model loader and routes are stubbed, so no torch import.

Run: cohere_env/bin/python -m unittest discover -s tests -v
"""

import importlib
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def import_main(env):
    """Import a fresh main.py under the given environment, with stub models.

    Args:
        env: Environment variables to set (others named DEVICE/PORT are cleared).

    Returns:
        The imported main module and the uvicorn.run mock call list.
    """
    loader = types.ModuleType("models.loader")
    loader.load_model = lambda device: None
    routes = types.ModuleType("routes.transcribe")
    from fastapi import APIRouter

    routes.router = APIRouter()
    stubs = {
        "models.loader": loader,
        "routes.transcribe": routes,
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *a, **k: None),
    }
    clean = {k: v for k, v in os.environ.items() if k not in ("DEVICE", "PORT")}
    clean.update(env)
    with (
        mock.patch.dict(sys.modules, stubs),
        mock.patch.dict(os.environ, clean, clear=True),
    ):
        sys.modules.pop("main", None)
        return importlib.import_module("main")


class DeviceFromEnv(unittest.TestCase):
    """DEVICE comes from the environment; cpu when unset."""

    def test_unset_keeps_cpu(self):
        """No DEVICE in the environment keeps the 222 CPU default."""
        self.assertEqual(import_main({}).DEVICE, "cpu")

    def test_cuda_from_env(self):
        """DEVICE=cuda (c09) reaches load_model."""
        self.assertEqual(import_main({"DEVICE": "cuda"}).DEVICE, "cuda")


class PortFromEnv(unittest.TestCase):
    """PORT comes from the environment; 6221 when unset."""

    def run_main(self, env):
        """Run main.py as __main__ with uvicorn.run mocked.

        Args:
            env: Environment variables to set.

        Returns:
            The port passed to uvicorn.run.
        """

        import uvicorn

        main = import_main(env)
        with (
            mock.patch.object(uvicorn, "run") as run,
            mock.patch.dict(os.environ, env),
            mock.patch.dict(sys.modules, {"main": main}),
        ):
            src = open(os.path.join(ROOT, "main.py")).read()
            tail = src[src.index('if __name__ == "__main__":') :]
            exec(
                compile(tail, "main.py", "exec"), {**vars(main), "__name__": "__main__"}
            )
        return run.call_args.kwargs["port"]

    def test_unset_keeps_6221(self):
        """No PORT keeps 6221, the 222 port."""
        self.assertEqual(self.run_main({}), 6221)

    def test_port_from_env(self):
        """PORT=8000 (c09) reaches uvicorn."""
        self.assertEqual(self.run_main({"PORT": "8000"}), 8000)


if __name__ == "__main__":
    unittest.main()
