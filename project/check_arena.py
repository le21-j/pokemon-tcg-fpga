"""Runnable checks for series scoring, cleanup, and external Python policies."""

import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from .arena import Agent, run_series, validate_action
from .collect_matches import DEFAULT_DECK
from .script_agent import ScriptAgent


def check_script_agents():
    observation = {"select": {"minCount": 1, "maxCount": 1, "option": [{}, {}]}}
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        adapters = []
        try:
            for name, choice in (("alice", 0), ("bob", 1)):
                folder = root / name
                folder.mkdir()
                (folder / "weights.json").write_text(json.dumps({"choice": choice}))
                (folder / "helper.py").write_text("import json\nfrom pathlib import Path\nCHOICE = json.loads(Path('weights.json').read_text())['choice']\n")
                (folder / "agent.py").write_text('''from .helper import CHOICE
import os
print("model loaded", flush=True)
os.write(1, b"native output also stays out of the protocol\\n")
calls = 0
def start_game(deck):
    assert len(deck) == 60
def agent(observation):
    global calls
    calls += 1
    assert calls == 1, "the game must start with a fresh module"
    return [CHOICE]
''')
                # Exercise a file path and a directory path with identical helper names.
                python = os.path.relpath(sys.executable) if choice == 0 else sys.executable
                adapters.append(ScriptAgent(folder / "agent.py" if choice == 0 else folder,
                                            python_executable=python))
            for _ in range(2):
                for choice, adapter in enumerate(adapters):
                    adapter.start_game(DEFAULT_DECK)
                    action = adapter(observation, random.Random(7))
                    validate_action(observation, action)
                    assert action == [choice]
                    assert adapter.stats["script_decisions"] == 1
                    assert adapter.identity()["script_sha256"]
            for adapter in adapters:
                adapter.close()
                assert adapter._process is None

            script = root / "configured.py"
            script.write_text('def policy(observation, configuration):\n    return [configuration["choice"]]\n')
            configured = ScriptAgent(script, entrypoint="policy", call_style="observation_configuration",
                                     configuration={"choice": 1})
            adapters.append(configured)
            configured.start_game(DEFAULT_DECK)
            assert configured(observation, random.Random(7)) == [1]
            configured.close()

            script.write_text('def agent(observation, rng):\n    return [rng.randrange(2)]\n')
            seeded = ScriptAgent(script, call_style="observation_rng")
            adapters.append(seeded)
            choices = []
            for _ in range(2):
                seeded.start_game(DEFAULT_DECK)
                choices.append(seeded(observation, random.Random(7)))
            assert choices[0] == choices[1]
            seeded.close()

            for source, expected in (
                ('raise ImportError("missing teammate dependency")\n', "missing teammate dependency"),
                ('def agent(observation):\n    raise ValueError("bad model output")\n', "bad model output"),
                ('import os\ndef agent(observation):\n    os._exit(7)\n', "worker exited"),
                ('import time\ndef agent(observation):\n    time.sleep(5)\n', "timed out"),
            ):
                script.write_text(source)
                adapter = ScriptAgent(script)
                adapters.append(adapter)
                try:
                    adapter.start_game(DEFAULT_DECK)
                    if expected == "timed out":
                        adapter.timeout = 0.1
                    adapter(observation, random.Random(7))
                except RuntimeError as error:
                    assert expected in str(error), str(error)
                    assert adapter._process is None
                else:
                    raise AssertionError(f"expected script failure: {expected}")
        finally:
            for adapter in adapters:
                adapter.close()
    print("PASS: file/folder policies, local weights/imports, process isolation, per-game reset, signatures, errors and timeouts")


class FinishedGame:
    def __init__(self, results):
        self.results = iter(results)
        self.sim = SimpleNamespace(Battle=SimpleNamespace(battle_ptr=None))
        self.finished = 0

    def battle_start(self, deck0, deck1):
        self.sim.Battle.battle_ptr = 1
        return {"current": {"result": next(self.results), "turn": 1}}, SimpleNamespace(errorPlayer=-1)

    def battle_finish(self):
        self.finished += 1


def main():
    def action(observation, rng):
        raise AssertionError("a completed game must not request another action")

    agents = (Agent("A", action), Agent("B", action))
    pool = {"sample-water": DEFAULT_DECK}
    game = FinishedGame([0] * 5)
    result = run_series(game, game.sim, *agents, pool, progress=None)
    assert result["wins"] == [3, 2] and result["winner"] == "A"
    assert len(result["games"]) == game.finished == 5
    assert [g["seat0"] for g in result["games"]] == ["A", "B", "A", "B", "A"]
    assert game.sim.Battle.battle_ptr is None

    game = FinishedGame([2, 0, 1, 0])
    result = run_series(game, game.sim, *agents, pool, progress=None)
    assert result["wins"] == [0, 3] and result["draws"] == 1
    assert result["winner"] == "B" and len(result["games"]) == 4
    assert game.finished == 4 and game.sim.Battle.battle_ptr is None

    game = FinishedGame([2] * 5)
    result = run_series(game, game.sim, *agents, pool, max_total_games=5, progress=None)
    assert not result["completed"] and result["winner"] is None
    assert result["wins"] == [0, 0] and result["draws"] == 5

    observation = {"select": {"minCount": 1, "maxCount": 2, "option": [{}, {}]}}
    validate_action(observation, [1, 0])
    for invalid in ([0, 0], [2], [], [True]):
        try:
            validate_action(observation, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid action accepted: {invalid}")
    print("PASS: first-to-three stopping, alternating seats, draw replay, incomplete draws, action validation, cleanup")
    check_script_agents()


if __name__ == "__main__":
    main()
