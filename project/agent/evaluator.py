"""Inference-only floating-point cache with transactional changes and restoration."""

from dataclasses import dataclass

import torch

from .features import Position
from .model import DeepSetsValue


@dataclass
class CachedEntity:
    group: int
    features: torch.Tensor
    encoded: torch.Tensor


class IncrementalEvaluator:
    def __init__(self, model: DeepSetsValue):
        self.model = model
        self._parameter_versions = tuple((id(p), p._version) for p in model.parameters())
        self._stack = []
        self._state = None

    def _check_weights(self):
        current = tuple((id(p), p._version) for p in self.model.parameters())
        if current != self._parameter_versions:
            raise RuntimeError("model weights changed; create and synchronize a new evaluator")

    @torch.inference_mode()
    def sync(self, position: Position):
        self._check_weights()
        self._stack.clear()
        device = next(self.model.parameters()).device
        pools = torch.zeros(4, 16, device=device)
        entries = {}
        for entity in position.entities:
            features = entity.features.to(device).clone()
            encoded = self.model.encoder(features)
            pools[int(entity.group)] += encoded
            entries[entity.instance_id] = CachedEntity(int(entity.group), features, encoded.clone())
        self._state = (entries, pools, position.globals.to(device).clone(),
                       position.history.to(device).clone(), position.observer, position.schema_id)

    @torch.inference_mode()
    def update(self, position: Position) -> dict[str, int]:
        """Derive add/remove/replace deltas from a complete canonical position."""
        self._check_weights()
        if self._state is None:
            raise RuntimeError("synchronize a root before updating")
        old_entries, old_pools, _, _, observer, schema = self._state
        if position.observer != observer or position.schema_id != schema:
            raise ValueError("observer/schema changed; synchronize a new root")
        entries = dict(old_entries)
        pools = old_pools.clone()  # Commit only after every replacement succeeds.
        incoming = {e.instance_id: e for e in position.entities}
        counts = {"removed": 0, "encoded": 0, "reused": 0}
        for instance_id, cached in old_entries.items():
            if instance_id not in incoming:
                pools[cached.group] -= cached.encoded
                del entries[instance_id]
                counts["removed"] += 1
        for entity in position.entities:
            features = entity.features.to(pools.device)
            cached = old_entries.get(entity.instance_id)
            if cached is not None and cached.group == int(entity.group) and torch.equal(cached.features, features):
                counts["reused"] += 1
                continue
            if cached is not None:
                pools[cached.group] -= cached.encoded
            encoded = self.model.encoder(features)
            pools[int(entity.group)] += encoded
            entries[entity.instance_id] = CachedEntity(int(entity.group), features.clone(), encoded.clone())
            counts["encoded"] += 1
        self._state = (entries, pools, position.globals.to(pools.device).clone(),
                       position.history.to(pools.device).clone(), observer, schema)
        return counts

    @torch.inference_mode()
    def value(self) -> torch.Tensor:
        self._check_weights()
        if self._state is None:
            raise RuntimeError("synchronize a root before evaluation")
        _, pools, globals, history, _, _ = self._state
        return self.model.value_from_pools(pools, globals, history).clone()

    def push(self):
        self._check_weights()
        if self._state is None:
            raise RuntimeError("no position to save")
        entries, pools, globals, history, observer, schema = self._state
        self._stack.append((dict(entries), pools.clone(), globals.clone(), history.clone(), observer, schema))

    def pop(self):
        self._check_weights()
        if not self._stack:
            raise RuntimeError("empty saved-position stack")
        self._state = self._stack.pop()

    @property
    def pooled_state(self):
        if self._state is None:
            raise RuntimeError("no synchronized position")
        return self._state[1].clone()
