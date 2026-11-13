"""Train the small value model on completed native-game observations.

AdamW(lr=3e-4) and optional Huber(delta=.2) follow the supplied Kaggle sample.
Targets are completed-game returns; search-score targets and PPO are separate work.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn

from .agent.features import ENTITY_FEATURES, GLOBAL_FEATURES, SCHEMA_ID, position_from_observation
from .agent.model import DeepSetsValue, batch_positions


def read_examples(path):
    examples = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            observer, result = row["observer"], row["result"]
            expected = 0.0 if result == 2 else (1.0 if result == observer else -1.0)
            if (result not in (0, 1, 2) or row["target"] != expected or
                    row["observation"]["current"]["yourIndex"] != observer):
                raise ValueError(f"inconsistent perspective/outcome on line {line_number}")
            examples.append((row["match_id"], position_from_observation(row["observation"]), expected))
    if not examples:
        raise ValueError("empty training dataset")
    return examples


@torch.inference_mode()
def measure(model, examples, batch_size, loss_fn, device):
    model.eval()
    total, mse = 0.0, 0.0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        predictions = model(*batch_positions([item[1] for item in batch], device))
        targets = torch.tensor([item[2] for item in batch], device=device)
        total += loss_fn(predictions, targets).item() * len(batch)
        mse += (predictions - targets).square().sum().item()
    return {"loss": total / len(examples), "mse": mse / len(examples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    if args.output.exists():
        parser.error("output exists; choose a new checkpoint path")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    examples = read_examples(args.data)
    groups = sorted({item[0] for item in examples})
    if len(groups) < 2:
        parser.error("at least two complete match groups are needed for a held-out split")
    rng.shuffle(groups)
    validation_groups = set(groups[:max(1, len(groups) // 5)])
    train = [item for item in examples if item[0] not in validation_groups]
    validation = [item for item in examples if item[0] in validation_groups]
    model = DeepSetsValue().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss() if args.loss == "mse" else nn.HuberLoss(delta=0.2)
    metadata = {
        "schema_id": SCHEMA_ID, "entity_features": ENTITY_FEATURES,
        "global_features": GLOBAL_FEATURES, "history": "32 zero slots in v0",
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_groups": sorted(set(groups) - validation_groups),
        "validation_groups": sorted(validation_groups),
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(f"Training positions: {len(train)}, validation positions: {len(validation)}")
    print(f"Optimizer: AdamW, lr={args.lr}; loss={args.loss}")
    history = []
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            rng.shuffle(train)
            for start in range(0, len(train), args.batch_size):
                batch = train[start:start + args.batch_size]
                targets = torch.tensor([item[2] for item in batch], device=args.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(*batch_positions([item[1] for item in batch], args.device))
                loss = loss_fn(predictions, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                optimizer.step()
        metrics = {"epoch": epoch,
                   "train": measure(model, train, args.batch_size, loss_fn, args.device),
                   "validation": measure(model, validation, args.batch_size, loss_fn, args.device)}
        history.append(metrics)
        print(f"Epoch {epoch:02d}: train={metrics['train']['loss']:.6f}, "
              f"validation={metrics['validation']['loss']:.6f}", flush=True)
        if metrics["validation"]["loss"] < best_loss:
            best_loss = metrics["validation"]["loss"]
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": epoch, "metrics": metrics, "metadata": metadata,
                          "torch_rng": torch.get_rng_state(), "python_rng": rng.getstate()}
            temporary = args.output.with_name(args.output.name + ".part")
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)
    report = {"metadata": metadata, "history": history, "best_validation_loss": best_loss}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Best validation checkpoint: {args.output}")


if __name__ == "__main__":
    main()
