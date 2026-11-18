"""Check Energy attachment using full, incremental, and restored evaluations."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import torch

from .agent.evaluator import IncrementalEvaluator
from .agent.features import Position, SCHEMA_ID, SUPPORTED_SCHEMAS, position_from_observation
from .agent.model import DeepSetsValue, batch_positions


def example_observations():
    """A synthetic API-shaped example; optional CLI files use real native observations."""
    def pokemon(serial, energy_count):
        return {"id": 721, "serial": serial, "hp": 90, "energies": [3] * energy_count,
                "energyCards": [{"id": 3, "serial": serial + 1000 + i} for i in range(energy_count)],
                "tools": [], "appearThisTurn": False}

    own = {"active": [pokemon(100, 1)], "bench": [pokemon(101, 0)],
           "hand": [{"id": 3, "serial": 200}, {"id": 1121, "serial": 201}],
           "handCount": 2, "deckCount": 40, "prize": [None] * 6, "discard": []}
    opponent = {"active": [pokemon(300, 1)], "bench": [], "hand": None,
                "handCount": 5, "deckCount": 40, "prize": [None] * 6, "discard": []}
    before = {"current": {"yourIndex": 0, "turn": 3, "turnActionCount": 0,
                           "firstPlayer": 0, "energyAttached": False, "result": -1,
                           "players": [own, opponent]},
              "select": {"type": 0, "minCount": 1, "maxCount": 1,
                         "option": [{"type": 8}, {"type": 14}]}, "logs": []}

    def attach(target):
        observation = deepcopy(before)
        player = observation["current"]["players"][0]
        energy = player["hand"].pop(0)
        player["handCount"] -= 1
        recipient = player[target][0]
        recipient["energyCards"].append(energy)
        recipient["energies"].append(3)
        observation["current"]["energyAttached"] = True
        observation["current"]["turnActionCount"] += 1
        return observation

    return before, attach("active"), attach("bench")


def check_attachment(model, before, after):
    evaluator = IncrementalEvaluator(model)
    evaluator.sync(before)
    original = evaluator.value()
    original_pools = evaluator.pooled_state
    evaluator.push()
    counts = evaluator.update(after)
    full = model.evaluate_position(after)
    incremental = evaluator.value()
    torch.testing.assert_close(incremental, full, rtol=1e-5, atol=1e-6)
    tensors = batch_positions([after])
    full_pools = model.pools(tensors[0], tensors[1], tensors[4])[0]
    torch.testing.assert_close(evaluator.pooled_state, full_pools, rtol=1e-5, atol=1e-6)
    evaluator.pop()
    torch.testing.assert_close(evaluator.pooled_state, original_pools, rtol=0, atol=0)
    torch.testing.assert_close(evaluator.value(), original, rtol=0, atol=0)
    torch.testing.assert_close(original, model.evaluate_position(before), rtol=1e-5, atol=1e-6)
    print(f"Original value:        {original.item():.9f}")
    print(f"After, full:           {full.item():.9f}")
    print(f"After, incremental:    {incremental.item():.9f}")
    print(f"Absolute difference:   {abs(full.item() - incremental.item()):.3g}")
    print(f"Restored value:        {evaluator.value().item():.9f}")
    print(f"Entity update counts:  {counts}")
    return evaluator


def check_game_traces(model, path, schema_id=SCHEMA_ID):
    caches = {}
    maximum_error, count = 0.0, 0
    game_ids = set()
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            position = position_from_observation(record["observation"], schema_id=schema_id)
            key = (record["game_id"], position.observer)
            if key not in caches:
                caches[key] = IncrementalEvaluator(model)
                caches[key].sync(position)
            else:
                caches[key].update(position)
            tensors = batch_positions([position])
            full_pools = model.pools(tensors[0], tensors[1], tensors[4])[0]
            torch.testing.assert_close(caches[key].pooled_state, full_pools, rtol=1e-5, atol=1e-6)
            full, incremental = model.evaluate_position(position), caches[key].value()
            torch.testing.assert_close(full, incremental, rtol=1e-5, atol=1e-6)
            maximum_error = max(maximum_error, abs(full.item() - incremental.item()))
            game_ids.add(record["game_id"])
            count += 1
    if not count:
        raise ValueError("empty trace dataset")
    print(f"Trace check:           {count} positions across {len(game_ids)} games; max error={maximum_error:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="evaluate trained weights instead of the fixed random initialization")
    parser.add_argument("--dataset", type=Path, help="also compare full/incremental updates across recorded game traces")
    args = parser.parse_args()
    if bool(args.before) != bool(args.after):
        parser.error("provide both --before and --after")
    torch.manual_seed(7)
    model = DeepSetsValue().eval()
    schema_id = SCHEMA_ID
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        schema_id = checkpoint["metadata"]["schema_id"]
        if schema_id not in SUPPORTED_SCHEMAS:
            raise ValueError(f"unknown checkpoint feature schema {schema_id!r}")
        model.load_state_dict(checkpoint["model"])
        print(f"Checkpoint epoch:      {checkpoint['epoch']}")
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 8865
    print(f"Model parameters:      {parameter_count}")
    if args.before:
        observations = (json.loads(args.before.read_text()), json.loads(args.after.read_text()))
        alternative = None
        print("Input source:          native observation files")
    else:
        first, second, third = example_observations()
        observations, alternative = (first, second), position_from_observation(third, schema_id=schema_id)
        print("Input source:          synthetic attachment fixture")
    before, after = (position_from_observation(obs, schema_id=schema_id) for obs in observations)
    with torch.inference_mode():
        evaluator = check_attachment(model, before, after)
        if alternative:
            evaluator.push()
            evaluator.update(alternative)
            torch.testing.assert_close(evaluator.value(), model.evaluate_position(alternative))
            evaluator.pop()
        reversed_after = replace(after, entities=tuple(reversed(after.entities)))
        torch.testing.assert_close(model.evaluate_position(after), model.evaluate_position(reversed_after))
        empty = Position((), torch.zeros(32), torch.zeros(32), before.observer, schema_id)
        batched = model(*batch_positions([before, after, empty]))
        for index, position in enumerate([before, after, empty]):
            torch.testing.assert_close(batched[index], model.evaluate_position(position))
        if args.dataset:
            check_game_traces(model, args.dataset, schema_id)

    # Both learned components must receive gradients through the grouped sums.
    loss = (model(*batch_positions([before, after])) - 1).square().mean()
    loss.backward()
    assert model.encoder[0].weight.grad.abs().sum().item() > 0
    assert model.value_head[0].weight.grad.abs().sum().item() > 0
    with torch.no_grad():
        model.encoder[0].weight.add_(0.001)
    try:
        evaluator.value()
    except RuntimeError as error:
        assert "weights changed" in str(error)
    else:
        raise AssertionError("stale learned-vector cache was accepted")
    print("PASS: attachment, restoration, permutation, variable-size batching, padding, gradients, stale-cache checks")


if __name__ == "__main__":
    main()
