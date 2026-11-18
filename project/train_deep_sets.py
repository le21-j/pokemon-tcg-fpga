"""Train the Deep Sets evaluator from completed native Kaggle CABT games.

A heuristic warm start produces a first evaluator. Its frozen native-search
policy plays both seats to create new examples. A second AdamW/Huber fit uses
the warmup and self-play positions, followed by fresh games against Random.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import random
import uuid

import torch

from .agent.native_search import NativeSearch
from .arena import ValueSearchAgent, priority_action
from .collect_matches import (DEFAULT_DECK, ENGINE_ROOT, choose_random, collect_game,
                              download_runtime, load_runtime, runtime_identity)
from .train_baseline import dataset_hash, read_examples, train_value


def collect_stage(game, sim, destination, run_id, games, actions, names, seed, max_steps):
    """Alternate seats and keep each game's two observer perspectives together."""
    stage = destination.stem
    rng = random.Random(seed)
    games_played, totals = [], Counter()
    temporary = destination.with_suffix(".part")
    with temporary.open("x") as stream:
        for index in range(games):
            seats = (0, 1) if index % 2 == 0 else (1, 0)
            game_id = f"{run_id}-{stage}-{index + 1}"
            game_seed = rng.getrandbits(63)
            records, _, result = collect_game(
                game, sim, random.Random(game_seed), game_id, max_steps,
                actions=tuple(actions[i] for i in seats),
            )
            for row in records:
                row.update(stage=stage, action_seed=game_seed)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts = Counter()
            for actor in actions:
                counts.update(getattr(actor, "stats", {}))
            totals.update(counts)
            winner = "draw" if result == 2 else names[seats[result]]
            games_played.append({"game_id": game_id, "seats": [names[i] for i in seats],
                                 "action_seed": game_seed, "result": result, "winner": winner,
                                 "positions": len(records), "agent_stats": dict(counts)})
            print(f"{stage} game {index + 1}/{games}: {winner}, {len(records)} decisions", flush=True)
    temporary.replace(destination)
    return {"games": games_played, "positions": sum(g["positions"] for g in games_played),
            "agent_stats": dict(totals), "dataset_sha256": dataset_hash(destination)}


def run(args):
    counts = (args.warmup_games, args.validation_games, args.self_play_games,
              args.evaluation_games, args.max_steps, args.epochs, args.batch_size, args.threads)
    if min(counts) < 1:
        raise ValueError("game counts, epochs, batch size, threads and max steps must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    if args.output.exists():
        raise FileExistsError(f"choose a new run directory: {args.output}")
    if args.download:
        try:
            pinned = runtime_identity(args.cg_lib)["runtime_source"] == "pinned_public_manifest"
        except FileNotFoundError:
            pinned = False
        if not pinned:
            download_runtime(args.cg_lib)
    engine = runtime_identity(args.cg_lib)
    game, sim = load_runtime(args.cg_lib)
    search_api = NativeSearch(sim.Battle.lib)

    directory = args.output
    directory.mkdir(parents=True)
    run_id = uuid.uuid4().hex
    stage_seeds = random.Random(args.seed)

    def collect(name, game_count, actors, labels):
        return collect_stage(game, sim, directory / f"{name}.jsonl", run_id, game_count,
                             actors, labels, stage_seeds.getrandbits(63), args.max_steps)

    heuristic = (priority_action, choose_random)
    names = ("priority", "random")
    warmup = collect("warmup", args.warmup_games, heuristic, names)
    validation = collect("validation", args.validation_games, heuristic, names)
    warm_train = read_examples(directory / "warmup.jsonl")
    held_out = read_examples(directory / "validation.jsonl")
    fit = dict(epochs=args.epochs, batch_size=args.batch_size, device=device,
               lr=3e-4, loss_name="huber")
    data_hashes = {"warmup": warmup["dataset_sha256"], "validation": validation["dataset_sha256"]}
    warm_checkpoint = directory / "warmup.pt"
    warm_fit = train_value(warm_train, held_out, warm_checkpoint, **fit, seed=args.seed,
                           provenance={"dataset_hashes": dict(data_hashes)})

    def actor(checkpoint):
        return ValueSearchAgent.from_checkpoint(checkpoint, search_api, [DEFAULT_DECK])

    self_play = collect("self-play", args.self_play_games,
                        (actor(warm_checkpoint), actor(warm_checkpoint)), ("frozen_a", "frozen_b"))
    self_examples = read_examples(directory / "self-play.jsonl")
    data_hashes["self_play"] = self_play["dataset_sha256"]
    final_checkpoint = directory / "model.pt"
    final_fit = train_value(warm_train + self_examples, held_out, final_checkpoint,
                            **fit, seed=args.seed + 1, initial_checkpoint=warm_checkpoint,
                            provenance={"dataset_hashes": data_hashes,
                                        "initial_checkpoint_sha256": dataset_hash(warm_checkpoint)})

    evaluation = collect("evaluation", args.evaluation_games, (actor(final_checkpoint), choose_random),
                         ("deep_sets", "random"))
    results = Counter(g["winner"] for g in evaluation["games"])
    evaluation.update(wins=results["deep_sets"], losses=results["random"], draws=results["draw"])
    report = {
        "run_id": run_id, "seed": args.seed, "native_shuffle_seed": None, "engine": engine,
        "environment": {"platform": platform.system(), "architecture": platform.machine(),
                        "python": platform.python_version(), "torch": str(torch.__version__),
                        "device": str(device), "cpu_threads": torch.get_num_threads(),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "deck": {"name": "sample-water", "cards": DEFAULT_DECK},
        "target": "terminal single-game return for current.yourIndex (+1, 0, -1)",
        "warmup": warmup, "validation": validation, "self_play": self_play, "evaluation": evaluation,
        "training": {"warmup": warm_fit, "self_play": final_fit},
        "checkpoint": {"name": final_checkpoint.name, "sha256": dataset_hash(final_checkpoint)},
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Fresh games vs Random: {evaluation['wins']}W/{evaluation['losses']}L/"
          f"{evaluation['draws']}D. Report: {directory / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new run directory")
    parser.add_argument("--cg-lib", type=Path, default=ENGINE_ROOT)
    parser.add_argument("--download", action="store_true", help="fetch pinned runtime on cache miss")
    parser.add_argument("--warmup-games", type=int, default=4)
    parser.add_argument("--self-play-games", type=int, default=2)
    parser.add_argument("--validation-games", type=int, default=2)
    parser.add_argument("--evaluation-games", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1, help="optimizer epochs per stage")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--device", default="cpu", help="PyTorch training device, or auto")
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads")
    parser.add_argument("--seed", type=int, default=7, help="Python/model seed; native shuffle is independent")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
