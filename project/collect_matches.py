"""Run native CABT games and collect acting-player observations with outcomes.

Uses the same battle_start/select/finish entry points as the Kaggle notebook.
Games here are individual games. A caller can supply a policy for each seat.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import uuid


ENGINE_REVISION = "998b15c14b1b77aafb1843bed63858455035affd"
ENGINE_ROOT = Path(__file__).parent / ".runtime" / f"cabt-{ENGINE_REVISION}"
DEFAULT_DECK = (
    [721, 721, 722, 722, 722, 722, 723, 723, 723, 723, 1092, 1121, 1121,
     1145, 1145, 1163, 1163, 1219, 1219, 1219, 1219, 1227, 1227, 1227,
     1227, 1262, 1262] + [3] * 33
)


def native_library_name():
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin":
        return "libcg.dylib"
    elif system == "Linux":
        return "libcg-arm64.so" if machine in ("arm64", "aarch64") else "libcg.so"
    elif system == "Windows":
        return "cg.dll"
    raise RuntimeError(f"unsupported native runtime platform: {system}/{machine}")


def download_runtime(destination: Path):
    library = native_library_name()
    base = f"https://raw.githubusercontent.com/Kaggle/kaggle-environments/{ENGINE_REVISION}"
    manifest = {"revision": ENGINE_REVISION, "files": {}}
    paths = {f"cg/{name}": f"kaggle_environments/envs/cabt/cg/{name}"
             for name in ("__init__.py", "game.py", "sim.py", library)}
    paths["LICENSE"] = "LICENSE"
    for relative, source in paths.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "2", f"{base}/{source}", "--output", str(temporary)], check=True)
        temporary.replace(target)
        manifest["files"][relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_runtime(root: Path):
    if not (root / "cg" / "game.py").is_file():
        raise FileNotFoundError(f"cg/game.py missing under {root}; use --download or --cg-lib")
    sys.path.insert(0, str(root.resolve()))
    game, sim = importlib.import_module("cg.game"), importlib.import_module("cg.sim")
    if Path(game.__file__).resolve() != (root / "cg" / "game.py").resolve():
        raise RuntimeError("a different cg runtime is already loaded; use a fresh worker process")
    return game, sim


def runtime_identity(root: Path):
    names = ("cg/__init__.py", "cg/game.py", "cg/sim.py", f"cg/{native_library_name()}")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    manifest_path = root / "runtime-manifest.json"
    pinned = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        pinned = (manifest.get("revision") == ENGINE_REVISION and
                  all(manifest.get("files", {}).get(name) == digest for name, digest in hashes.items()))
    return {"engine_revision": ENGINE_REVISION if pinned else None,
            "engine_version": 2 if pinned else None,
            "runtime_source": "pinned_public_manifest" if pinned else "external_unverified",
            "runtime_file_sha256": hashes}


def choose_random(observation: dict, rng: random.Random) -> list[int]:
    selection = observation["select"]
    count = selection["maxCount"]
    size = len(selection["option"])
    if not 0 <= selection["minCount"] <= count <= size:
        raise ValueError("selection needs a dedicated cardinality handler")
    return rng.sample(range(size), count)


def public_observation(observation: dict) -> dict:
    # Opaque search serialization is not a model-visible observation feature.
    return {key: value for key, value in observation.items() if key != "search_begin_input"}


def validate_action(observation, action):
    selection = observation["select"]
    if (not isinstance(action, list) or any(type(i) is not int for i in action) or
            len(set(action)) != len(action) or
            not selection["minCount"] <= len(action) <= selection["maxCount"] or
            any(i < 0 or i >= len(selection["option"]) for i in action)):
        raise ValueError(f"agent returned an invalid selection: {action}")


def game_return(result, observer):
    if result not in (0, 1, 2) or observer not in (0, 1):
        raise ValueError(f"invalid native outcome/perspective: {result}/{observer}")
    return 0.0 if result == 2 else (1.0 if result == observer else -1.0)


def collect_game(game, sim, rng, game_id, max_steps, *, actions=(choose_random, choose_random),
                 decks=(DEFAULT_DECK, DEFAULT_DECK)):
    records, attachment = [], None
    for action, deck in zip(actions, decks, strict=True):
        if hasattr(action, "start_game"):
            action.start_game(list(deck))
    observation, start = game.battle_start(*decks)
    try:
        if start.errorPlayer >= 0 or observation is None:
            raise RuntimeError(f"deck/start error: player={start.errorPlayer}, type={start.errorType}")
        while observation["current"]["result"] < 0:
            if len(records) >= max_steps:
                raise RuntimeError(f"game exceeded {max_steps} selections; no outcome labels written")
            before = public_observation(observation)
            observer = observation["current"]["yourIndex"]
            choice = actions[observer](observation, rng)
            validate_action(observation, choice)
            records.append({"game_id": game_id, "match_id": game_id, "observer": observer,
                            "step": len(records), "observation": before, "action": choice})
            is_attachment = (len(choice) == 1 and
                             observation["select"]["option"][choice[0]]["type"] == 8)
            observation = game.battle_select(choice)
            if (attachment is None and is_attachment and
                    observation["current"]["yourIndex"] == observer and
                    observation["current"]["result"] < 0):
                attachment = (before, public_observation(observation))
        result = observation["current"]["result"]
        if result not in (0, 1, 2):
            raise ValueError(f"unexpected native result {result}")
        for record in records:
            record["result"] = result
            record["target"] = game_return(result, record["observer"])
        return records, attachment, result
    finally:
        if sim.Battle.battle_ptr:
            game.battle_finish()
            sim.Battle.battle_ptr = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download the pinned public native runtime; requires curl")
    parser.add_argument("--cg-lib", type=Path, default=ENGINE_ROOT)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7, help="Python action RNG only; native shuffle seed is not exposed")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1 or args.max_steps < 1:
        parser.error("games and max-steps must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new output path")
    if args.download:
        download_runtime(args.cg_lib)
    runtime = runtime_identity(args.cg_lib)
    game, sim = load_runtime(args.cg_lib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng, run_id = random.Random(args.seed), uuid.uuid4().hex
    captured = False
    with args.output.open("x") as output:
        for index in range(args.games):
            records, attachment, result = collect_game(game, sim, rng, f"{run_id}-{index}", args.max_steps)
            for record in records:
                record.update(runtime, format="single_native_game", action_seed=args.seed)
                output.write(json.dumps(record) + "\n")
            output.flush()
            print(f"Game {index + 1}: result={result}, decisions={len(records)}", flush=True)
            if attachment and not captured:
                for name, observation in zip(("energy-before", "energy-after"), attachment):
                    (args.output.parent / f"{name}.json").write_text(json.dumps(observation, indent=2) + "\n")
                captured = True
    print(f"Dataset: {args.output}\nEnergy attachment captured: {captured}")


if __name__ == "__main__":
    main()
