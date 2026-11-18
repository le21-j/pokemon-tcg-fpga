"""Exercise the native Kaggle training and arena interfaces with a small real run."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import torch

from .agent.native_search import NativeSearch
from .arena import Agent, ValueSearchAgent, kaggle_agent, play_game, priority_action, validate_action
from .collect_matches import DEFAULT_DECK, ENGINE_ROOT, load_runtime
from .energy_attachment import check_game_traces
from .sample_sdk import prepare_sample_sdk


def check_records(path):
    """Verify every saved choice and label against its own decision observation."""
    steps = Counter()
    results = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            observation = row["observation"]
            match = row["match_id"]
            player = observation["current"]["yourIndex"]
            result = row["result"]
            assert "search_begin_input" not in observation
            assert observation["current"]["result"] < 0 and row["observer"] == player
            assert player in (0, 1) and result in (0, 1, 2)
            assert row["target"] == (0 if result == 2 else 1 if player == result else -1)
            assert steps[match] == row["step"] and results.setdefault(match, result) == result
            validate_action(observation, row["action"])
            steps[match] += 1
    assert steps, f"empty game dataset {path}"
    print(f"{path.name}: {len(steps)} complete-game groups, {sum(steps.values())} legal selections")
    return steps


def check_search_perspective(model, opening):
    """A +0.5 leaf favors a player who acts again and disfavors handing over the turn."""
    model = deepcopy(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.value_head[-1].bias.fill_(0.5)
    root = deepcopy(opening)
    root["search_begin_input"] = "stub"  # Opaque field is intentionally absent in saved training rows.
    assert root["select"]["minCount"] == root["select"]["maxCount"] == 1
    assert len(root["select"]["option"]) == 2  # The opening question asks who goes first.
    other = deepcopy(root)
    other["current"]["players"].reverse()
    other["current"]["yourIndex"] = 1 - root["current"]["yourIndex"]
    children = [other, root]
    search = Mock()
    search.begin.return_value = {"searchId": 0, "observation": root}
    search.step.side_effect = lambda _, choice: {"searchId": 1 + choice[0], "observation": children[choice[0]]}
    actor = ValueSearchAgent(model, search, [DEFAULT_DECK])
    actor.start_game(DEFAULT_DECK)
    assert actor(root, random.Random(7)) == [1]
    children[0] = {"current": {"result": root["current"]["yourIndex"]}, "select": None}
    assert actor(root, random.Random(7)) == [0]  # A terminal win beats a +0.5 continuation.
    forced = deepcopy(root)
    forced["select"].update(minCount=0, maxCount=0, option=[])
    assert actor(forced, random.Random(7)) == []
    multiple = deepcopy(root)
    multiple["select"].update(minCount=0, maxCount=2)
    search.begin.return_value = {"searchId": 0, "observation": multiple}

    def multiple_step(_, choice):
        child = children[0] if len(choice) == 2 else root
        return {"searchId": 1, "observation": child}

    search.step.side_effect = multiple_step
    selected = actor(multiple, random.Random(7))
    validate_action(multiple, selected)
    assert set(selected) == {0, 1}  # Search can choose both optional cards when this branch wins.


def check_run(run_dir, cg_lib):
    report = json.loads((run_dir / "report.json").read_text())
    groups = []
    for stage in ("warmup", "self_play", "validation", "evaluation"):
        path = run_dir / (stage.replace("_", "-") + ".jsonl")
        steps = check_records(path)
        summary = report[stage]
        assert {g["game_id"]: g["positions"] for g in summary["games"]} == dict(steps)
        assert summary["positions"] == sum(steps.values())
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == summary["dataset_sha256"]
        groups.append(set(steps))
    warmup, self_play, validation, evaluation = groups
    assert warmup.isdisjoint(self_play) and (warmup | self_play).isdisjoint(validation)
    assert evaluation.isdisjoint(warmup | self_play | validation)
    assert report["self_play"]["agent_stats"]["searched_decisions"] > 0

    previous = torch.load(run_dir / "warmup.pt", map_location="cpu", weights_only=True)
    checkpoint = run_dir / "model.pt"
    with checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == report["checkpoint"]["sha256"]
    trained = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert trained["epoch"] >= 1 and trained["optimizer"]["state"]
    assert set(trained["metadata"]["validation_groups"]) == validation
    train_groups = set(trained["metadata"]["train_groups"])
    assert train_groups == warmup | self_play
    assert trained["metadata"]["train_positions"] == report["warmup"]["positions"] + report["self_play"]["positions"]
    assert trained["metadata"]["validation_positions"] == report["validation"]["positions"]
    for module in ("encoder", "value_head"):
        weights = [name for name in trained["model"] if name.startswith(module + ".") and name.endswith(".weight")]
        assert any(not torch.equal(trained["model"][name], previous["model"][name]) for name in weights), module
    assert all(torch.isfinite(value).all() for value in trained["model"].values())

    if cg_lib.resolve() == ENGINE_ROOT.resolve():
        game, sim, sample_api, search = prepare_sample_sdk()
    else:
        game, sim = load_runtime(cg_lib)
        sample_api, search = None, NativeSearch(sim.Battle.lib)
    agent = ValueSearchAgent.from_checkpoint(checkpoint, search, [DEFAULT_DECK])
    # Audit the trained accumulator across actual decisions from both seats.
    with torch.inference_mode():
        check_game_traces(agent.model, run_dir / "self-play.jsonl")
    # Reuse an actual native opening for deterministic branch outcomes and actor switches.
    with (run_dir / "warmup.jsonl").open() as stream:
        opening = json.loads(next(stream))["observation"]
    check_search_perspective(agent.model, opening)

    decks = [("sample-water", DEFAULT_DECK)] * 2
    rngs = [random.Random(701), random.Random(702)]
    callback = kaggle_agent(agent, DEFAULT_DECK, seed=701)

    def choose(observation, rng):
        selected = callback(observation)
        if sample_api is not None:
            # The original Kaggle sample converts these same raw inputs to cg.api dataclasses.
            view = sample_api.to_observation_class(observation)
            assert view.current.yourIndex == observation["current"]["yourIndex"]
            assert view.select.minCount <= len(selected) <= view.select.maxCount
            assert all(0 <= index < len(view.select.option) for index in selected)
        return selected

    players = [Agent("trained", choose), Agent("priority", priority_action)]
    # A very short game can finish before search has a genuine choice.
    for number in range(3):
        selected_deck = callback({"current": None, "select": None})
        assert selected_deck == DEFAULT_DECK and selected_deck is not DEFAULT_DECK
        decks[0] = ("sample-water", selected_deck)
        seats = [0, 1] if number % 2 == 0 else [1, 0]
        match = play_game(game, sim, players, decks, seats, rngs, max_steps=10000)
        assert not sim.Battle.battle_ptr and match["winner_index"] in (None, 0, 1)
        stats = agent.stats
        if stats.get("searched_decisions", 0):
            assert stats["evaluated_actions"] >= stats["searched_decisions"]
            print(f"Fresh arena game: {match['decisions']} legal selections, "
                  f"{stats['searched_decisions']} searched decisions, {stats['evaluated_actions']} evaluated actions")
            break
    else:
        raise AssertionError("reloaded checkpoint made no searched decisions in three fresh native games")
    print("PASS: native observations and legal moves, held-out games, learned encoder and head, "
          "incremental values, checkpoint reload and arena play")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="verify an existing Deep Sets training run")
    parser.add_argument("--cg-lib", type=Path, default=ENGINE_ROOT)
    parser.add_argument("--download", action="store_true", help="fetch the pinned runtime for a fresh test")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.run_dir:
        check_run(args.run_dir, args.cg_lib)
        return
    with TemporaryDirectory() as directory:
        run_dir = Path(directory) / "training"
        command = [sys.executable, "-m", "project.train_deep_sets", "--output", str(run_dir),
                   "--cg-lib", str(args.cg_lib), "--warmup-games", "2", "--self-play-games", "2",
                   "--validation-games", "2", "--epochs", "1", "--device", "cpu"]
        if args.download:
            command.append("--download")
        subprocess.run(command, check=True)
        check_run(run_dir, args.cg_lib)


if __name__ == "__main__":
    main()
