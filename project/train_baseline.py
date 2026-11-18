"""Train the small value model on completed native-game observations.

AdamW(lr=3e-4) and optional Huber(delta=.2) follow the supplied Kaggle sample.
Targets are completed-game returns; search-score targets and PPO are separate work.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch import nn

from .agent.features import ENTITY_FEATURES, GLOBAL_FEATURES, SCHEMA_ID, position_from_observation
from .agent.model import DeepSetsValue, batch_positions
from .collect_matches import game_return, validate_action


def read_examples(path):
    examples = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            observer, result = row["observer"], row["result"]
            expected = game_return(result, observer)
            observation = row["observation"]
            if row["target"] != expected or observation["current"]["yourIndex"] != observer:
                raise ValueError(f"inconsistent perspective/outcome on line {line_number}")
            if "action" in row:
                validate_action(observation, row["action"])
            examples.append((row["match_id"], position_from_observation(observation), expected))
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
    metrics = {"loss": total / len(examples), "mse": mse / len(examples)}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("nonfinite evaluation loss or MSE")
    return metrics


def train_value(train, validation, output, *, epochs, batch_size, lr, loss_name,
                seed, device, provenance, initial_checkpoint=None, include_initial=False):
    """Fit both networks jointly; normally select among epochs that received updates."""
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"checkpoint or report already exists for {output}")
    if epochs < 1 or batch_size < 1 or lr <= 0 or loss_name not in ("huber", "mse"):
        raise ValueError("invalid value-training configuration")
    train_groups, validation_groups = ({item[0] for item in data} for data in (train, validation))
    if not train_groups or not validation_groups or train_groups & validation_groups:
        raise ValueError("training and validation must contain disjoint, nonempty game groups")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = DeepSetsValue().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if initial_checkpoint is not None:
        source = torch.load(initial_checkpoint, map_location=device, weights_only=True)
        if source["metadata"]["schema_id"] != SCHEMA_ID:
            raise ValueError("checkpoint feature schema mismatch")
        model.load_state_dict(source["model"])
        optimizer.load_state_dict(source["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
    loss_fn = nn.MSELoss() if loss_name == "mse" else nn.HuberLoss(delta=0.2)
    metadata = {
        "schema_id": SCHEMA_ID, "entity_features": ENTITY_FEATURES,
        "global_features": GLOBAL_FEATURES, "history": "32 zero slots",
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "train_positions": len(train), "validation_positions": len(validation),
        "config": {"optimizer": "AdamW", "lr": lr, "loss": loss_name,
                   "huber_delta": 0.2 if loss_name == "huber" else None,
                   "batch_size": batch_size, "epochs": epochs, "seed": seed, "device": str(device),
                   "include_initial_in_selection": include_initial},
        "parameters": sum(p.numel() for p in model.parameters()),
        **provenance,
    }
    print(f"Training positions: {len(train)}, validation positions: {len(validation)}")
    print(f"Optimizer: AdamW, lr={lr}; loss={loss_name}")
    history = []
    best_loss = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    updates = 0
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            rng.shuffle(train)
            for start in range(0, len(train), batch_size):
                batch = train[start:start + batch_size]
                targets = torch.tensor([item[2] for item in batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(*batch_positions([item[1] for item in batch], device))
                loss = loss_fn(predictions, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                optimizer.step()
                updates += 1
        metrics = {"epoch": epoch, "updates": updates,
                   "train": measure(model, train, batch_size, loss_fn, device),
                   "validation": measure(model, validation, batch_size, loss_fn, device)}
        history.append(metrics)
        print(f"Epoch {epoch:02d}: train={metrics['train']['loss']:.6f}, "
              f"validation={metrics['validation']['loss']:.6f}", flush=True)
        if (epoch or include_initial) and metrics["validation"]["loss"] < best_loss:
            best_loss = metrics["validation"]["loss"]
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": epoch, "metrics": metrics, "metadata": metadata,
                          "torch_rng": torch.get_rng_state(), "python_rng": rng.getstate()}
            temporary = output.with_name(output.name + ".part")
            torch.save(checkpoint, temporary)
            temporary.replace(output)
    report = {"metadata": metadata, "history": history, "best_validation_loss": best_loss}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Best validation checkpoint: {output}")
    return report


def dataset_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    examples = read_examples(args.data)
    groups = sorted({item[0] for item in examples})
    if len(groups) < 2:
        parser.error("at least two complete match groups are needed for a held-out split")
    random.Random(args.seed).shuffle(groups)
    validation_groups = set(groups[:max(1, len(groups) // 5)])
    train = [item for item in examples if item[0] not in validation_groups]
    validation = [item for item in examples if item[0] in validation_groups]
    train_value(train, validation, args.output, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, loss_name=args.loss, seed=args.seed, device=args.device,
                provenance={"dataset_sha256": dataset_hash(args.data)}, include_initial=True)


if __name__ == "__main__":
    main()
