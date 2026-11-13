"""Observed-state prototype schema, without card-ID or hidden-state inputs.

v0 represents resources/status but needs card metadata before strategic training.
History slots are zero until an audited persistent-history encoder is added.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor


class Group(IntEnum):
    OWN_BOARD = 0
    OPPONENT_BOARD = 1
    OWN_HAND = 2
    PUBLIC_DISCARD = 3


SCHEMA_ID = "observed-resources-v0"
ENTITY_FEATURES = (
    "hp_div_400", "energy_units_div_10", "energy_cards_div_10",
    "tools_div_4", "active", "bench", "own", "hand", "discard",
    "appeared_this_turn", "poisoned", "burned", "asleep", "paralyzed",
    "confused", "pokemon",
)
GLOBAL_FEATURES = (
    "turn_div_100", "turn_actions_div_100", "observer_started",
    "supporter_played", "stadium_played", "energy_attached", "retreated",
    "own_hand_div_60", "opponent_hand_div_60", "own_deck_div_60",
    "opponent_deck_div_60", "own_prizes_div_6", "opponent_prizes_div_6",
    "own_board_div_6", "opponent_board_div_6", "min_count_div_60",
    "max_count_div_60", "option_count_div_100",
) + tuple(f"selection_type_{i}" for i in range(14))


@dataclass(frozen=True)
class Entity:
    instance_id: str
    group: Group
    features: Tensor


@dataclass(frozen=True)
class Position:
    entities: tuple[Entity, ...]
    globals: Tensor
    history: Tensor
    observer: int
    schema_id: str = SCHEMA_ID

    def __post_init__(self):
        if self.observer not in (0, 1):
            raise ValueError("observer must be 0 or 1")
        ids = [e.instance_id for e in self.entities]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate entity instance IDs")
        for entity in self.entities:
            if int(entity.group) not in range(4):
                raise ValueError("invalid entity group")
            if entity.features.shape != (16,) or not torch.isfinite(entity.features).all():
                raise ValueError("entity features must be 16 finite values")
        for vector in (self.globals, self.history):
            if vector.shape != (32,) or not torch.isfinite(vector).all():
                raise ValueError("global/history features must be 32 finite values")


def position_from_observation(observation: dict) -> Position:
    """Encode one acting-player observation from cg.game, ignoring opaque data."""
    state = observation.get("current")
    selection = observation.get("select")
    if state is None or selection is None:
        raise ValueError("a decision observation with current/select is required")
    observer = state["yourIndex"]
    players = state["players"]
    if players[observer].get("hand") is None:
        raise ValueError("acting player's hand is unavailable")
    entities = []
    for owner, player in enumerate(players):
        own = owner == observer
        for zone in ("active", "bench", "hand", "discard"):
            if zone == "hand" and not own:
                continue
            for card in player.get(zone) or []:
                if card is None:
                    continue
                board = zone in ("active", "bench")
                group = (Group.OWN_BOARD if own else Group.OPPONENT_BOARD) if board else (
                    Group.OWN_HAND if zone == "hand" else Group.PUBLIC_DISCARD
                )
                active = zone == "active"
                values = [
                    card.get("hp", 0) / 400,
                    len(card.get("energies") or []) / 10,
                    len(card.get("energyCards") or []) / 10,
                    len(card.get("tools") or []) / 4,
                    active, zone == "bench", own, zone == "hand", zone == "discard",
                    card.get("appearThisTurn", False),
                    *(bool(player.get(status, False)) and active for status in
                      ("poisoned", "burned", "asleep", "paralyzed", "confused")),
                    board,
                ]
                entities.append(Entity(
                    f"player{owner}:card{card['serial']}", group,
                    torch.tensor(values, dtype=torch.float32),
                ))
    own, opponent = players[observer], players[1 - observer]
    values = [
        state["turn"] / 100, state.get("turnActionCount", 0) / 100,
        state.get("firstPlayer") == observer, state.get("supporterPlayed", False),
        state.get("stadiumPlayed", False), state.get("energyAttached", False),
        state.get("retreated", False), own["handCount"] / 60,
        opponent["handCount"] / 60, own["deckCount"] / 60, opponent["deckCount"] / 60,
        len(own["prize"]) / 6, len(opponent["prize"]) / 6,
        (len(own["active"]) + len(own["bench"])) / 6,
        (len(opponent["active"]) + len(opponent["bench"])) / 6,
        selection["minCount"] / 60, selection["maxCount"] / 60,
        len(selection["option"]) / 100,
        *(selection["type"] == i for i in range(14)),
    ]
    return Position(tuple(entities), torch.tensor(values, dtype=torch.float32),
                    torch.zeros(32), observer)
