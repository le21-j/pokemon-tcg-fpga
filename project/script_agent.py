"""Run a teammate's Python policy in a separate process, without model assumptions."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid


CALL_STYLES = ("observation", "observation_rng", "observation_configuration")


def resolve_script(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        for name in ("agent.py", "main.py", "__init__.py"):
            if (path / name).is_file():
                return path / name
        scripts = list(path.glob("*.py"))
        if len(scripts) != 1:
            raise ValueError(f"Choose a .py file inside {path}; no unambiguous agent entry file found")
        path = scripts[0]
    if not path.is_file() or path.suffix != ".py":
        raise ValueError(f"Agent path must be a Python file or a folder containing one: {path}")
    return path


class ScriptAgent:
    """Callable arena adapter. A fresh worker loads the script and weights each game.

    The script exports `agent(observation)` by default. Its working directory and
    import path are its containing folder. Optional hooks are `start_game(deck)`
    on the callable or module, and `close()` on this adapter releases the worker.
    """

    def __init__(self, path, entrypoint="agent", call_style="observation",
                 python_executable=None, timeout=60, configuration=None):
        self.path = resolve_script(path)
        if call_style not in CALL_STYLES:
            raise ValueError(f"call_style must be one of {CALL_STYLES}")
        if not entrypoint.isidentifier() or timeout <= 0:
            raise ValueError("Provide an exported callable name and a positive timeout")
        executable = shutil.which(os.path.expanduser(str(python_executable or sys.executable)))
        if executable is None:
            raise ValueError(f"Python interpreter not found: {python_executable}")
        if configuration is not None and not isinstance(configuration, dict):
            raise ValueError("configuration must be a JSON object")
        self.entrypoint, self.call_style = entrypoint, call_style
        # Keep virtualenv symlinks intact, and remain valid after changing the worker cwd.
        self.python_executable, self.timeout = os.path.abspath(executable), timeout
        self.configuration = configuration if configuration is not None else {}
        self.stats = {}
        self._process = None
        self._log = None

    def identity(self):
        return {"script": str(self.path), "script_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "entrypoint": self.entrypoint, "call_style": self.call_style,
                "python_executable": self.python_executable, "timeout_seconds": self.timeout,
                "configuration": self.configuration}

    def start_game(self, own_deck):
        self.close()
        self.stats = {"script_decisions": 0, "script_action_seconds": 0.0}
        self._responses = queue.Queue()
        self._log = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            [self.python_executable, "-u", str(Path(__file__).resolve()), "--worker",
             str(self.path), self.entrypoint, self.call_style],
            cwd=self.path.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, encoding="utf-8", bufsize=1,
        )
        # A queue keeps response timeouts portable across Windows, macOS, and Linux.
        def read_responses(stream, responses):
            for line in stream:
                responses.put(line)
            responses.put(None)
        self._reader = threading.Thread(target=read_responses,
                                        args=(self._process.stdout, self._responses), daemon=True)
        self._reader.start()
        self._receive()  # Wait for imports/weight loading before starting a native game.
        self._request({"event": "start_game", "own_deck": own_deck})

    def _fail(self, message):
        details = ""
        if self._log is not None:
            self._log.seek(0, os.SEEK_END)
            self._log.seek(max(0, self._log.tell() - 8000))
            details = self._log.read().decode("utf-8", errors="replace")
        self.close()
        raise RuntimeError(f"Script agent {self.path.name}: {message}\n{details}".rstrip())

    def _receive(self):
        try:
            line = self._responses.get(timeout=self.timeout)
        except queue.Empty:
            self._fail(f"timed out after {self.timeout} seconds")
        if line is None:
            self._fail("worker exited before responding")
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            self._fail(f"invalid worker response: {line[:200]}")
        if not result["ok"]:
            self._fail(result["error"])
        return result

    def _request(self, message):
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            self._fail("worker stopped accepting observations")
        return self._receive()

    def __call__(self, observation, rng):
        if self._process is None:
            raise RuntimeError("Call start_game(own_deck) before requesting a move")
        started = time.perf_counter()
        response = self._request({"event": "act", "observation": observation,
                                  "seed": rng.getrandbits(64), "configuration": self.configuration})
        self.stats["script_decisions"] += 1
        self.stats["script_action_seconds"] += time.perf_counter() - started
        return response["action"]

    def close(self):
        if self._process is not None:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._reader.join(timeout=1)
            self._process.stdout.close()
            self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None


def _load_policy(path, entrypoint):
    # A package namespace permits `from .helper import ...` as well as sibling imports.
    name = "_arena_policy_" + uuid.uuid4().hex
    sys.path.insert(0, str(path.parent))
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    if path.name == "__init__.py":
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    else:
        package = types.ModuleType(name)
        package.__path__ = [str(path.parent)]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + ".entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    policy = getattr(module, entrypoint)
    if not callable(policy) or isinstance(policy, type):
        raise TypeError(f"{entrypoint} must be a function or callable instance, not a class")
    return module, policy


def _worker(path, entrypoint, call_style):
    # Keep Python prints AND native-library stdout away from the JSON response pipe.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def reply(**values):
        protocol.write(json.dumps(values) + "\n")
        protocol.flush()

    try:
        module, policy = _load_policy(Path(path), entrypoint)
        reply(ok=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request["event"] == "start_game":
                hook = getattr(policy, "start_game", None) or getattr(module, "start_game", None)
                if hook is not None:
                    hook(request["own_deck"])
                reply(ok=True)
            elif request["event"] == "act":
                observation = request["observation"]
                if call_style == "observation_rng":
                    action = policy(observation, random.Random(request["seed"]))
                elif call_style == "observation_configuration":
                    action = policy(observation, request["configuration"])
                else:
                    action = policy(observation)
                reply(ok=True, action=action)
            else:
                raise ValueError(f"Unknown arena request: {request['event']}")
    except BaseException:
        reply(ok=False, error=traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("Used by project.arena.ipynb; provide a Python policy through its agent forms")
    _worker(*sys.argv[2:])
