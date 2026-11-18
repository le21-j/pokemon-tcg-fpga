"""Trainable full-computation reference for the 8,865-parameter evaluator."""

import torch
from torch import Tensor, nn

from .features import Position


def batch_positions(positions: list[Position], device="cpu") -> tuple[Tensor, ...]:
    if not positions:
        raise ValueError("empty position batch")
    count = max(len(p.entities) for p in positions)
    # Positions are encoded on the host; transfer complete tensors to the trainer.
    entities = torch.zeros(len(positions), count, 16)
    groups = torch.zeros(len(positions), count, dtype=torch.long)
    mask = torch.zeros(len(positions), count)
    for row, position in enumerate(positions):
        size = len(position.entities)
        if size:
            entities[row, :size] = torch.stack([e.features for e in position.entities])
            groups[row, :size] = torch.tensor([int(e.group) for e in position.entities])
            mask[row, :size] = 1
    return (entities.to(device), groups.to(device), torch.stack([p.globals for p in positions]).to(device),
            torch.stack([p.history for p in positions]).to(device), mask.to(device))


class DeepSetsValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
                                     nn.Linear(16, 16), nn.ReLU())
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def pools(self, entities: Tensor, groups: Tensor, mask: Tensor) -> Tensor:
        encoded = self.encoder(entities)
        encoded = encoded * mask.unsqueeze(-1)  # Mask bias outputs as well as inputs.
        pools = encoded.new_zeros(entities.shape[0], 4, 16)
        indices = groups.unsqueeze(-1).expand(-1, -1, 16)
        return pools.scatter_add(1, indices, encoded)

    def value_from_pools(self, pools: Tensor, globals: Tensor, history: Tensor) -> Tensor:
        state = torch.cat((pools.flatten(start_dim=-2), globals, history), dim=-1)
        return self.value_head(state).squeeze(-1)

    def forward(self, entities: Tensor, groups: Tensor, globals: Tensor,
                history: Tensor, mask: Tensor) -> Tensor:
        return self.value_from_pools(self.pools(entities, groups, mask), globals, history)

    def evaluate_position(self, position: Position) -> Tensor:
        return self(*batch_positions([position], next(self.parameters()).device))[0]
