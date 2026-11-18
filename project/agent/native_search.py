"""Raw-dictionary bindings derived from the original Kiyota cg-lib v9 ABI.

Source: https://www.kaggle.com/datasets/kiyotah/cg-lib/versions/9
The original sim.py/api.py were inspected before binding these functions.
Use the pinned platform library already loaded by project.collect_matches.
"""

from collections import Counter
import ctypes
import json


class NativeSearch:
    def __init__(self, library):
        self.lib = library
        integer_array = ctypes.POINTER(ctypes.c_int)
        library.AgentStart.restype = ctypes.c_void_p
        library.AgentStart.argtypes = []
        library.SearchBegin.restype = ctypes.c_char_p
        library.SearchBegin.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                       *([integer_array] * 6), ctypes.c_int]
        library.SearchStep.restype = ctypes.c_char_p
        library.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, integer_array, ctypes.c_int]
        library.SearchEnd.restype = None
        library.SearchEnd.argtypes = [ctypes.c_void_p]
        library.SearchRelease.restype = None
        library.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        for name in ("AllCard", "AllAttack"):
            function = getattr(library, name)
            function.restype = ctypes.c_char_p
            function.argtypes = []
        self.cards = {card["cardId"]: card for card in json.loads(library.AllCard())}
        self.attacks = {attack["attackId"]: attack for attack in json.loads(library.AllAttack())}
        self.agent_pointer = library.AgentStart()
        if not self.agent_pointer:
            raise RuntimeError("native search-agent allocation failed")

    @staticmethod
    def _array(values):
        return (ctypes.c_int * len(values))(*values)

    @staticmethod
    def _state(raw):
        if not raw:
            raise RuntimeError("empty native search result")
        result = json.loads(raw)
        if result["error"] != 0:
            raise ValueError(f"native search API error {result['error']}")
        return result["state"]

    def begin(self, observation, predictions):
        serialized = observation["search_begin_input"].encode("ascii")
        return self._state(self.lib.SearchBegin(
            self.agent_pointer, serialized, len(serialized),
            *(self._array(predictions[name]) for name in
              ("your_deck", "your_prize", "opponent_deck", "opponent_prize",
               "opponent_hand", "opponent_active")), 0,
        ))

    def step(self, search_id, selection):
        return self._state(self.lib.SearchStep(
            self.agent_pointer, search_id, self._array(selection), len(selection)))

    def release(self, search_id):
        self.lib.SearchRelease(self.agent_pointer, search_id)

    def end(self):
        self.lib.SearchEnd(self.agent_pointer)


def known_cards(player, public_cards=()):
    """Count visible physical instances once, including nested attached/evolved cards."""
    instances = {}

    def visit(value):
        if isinstance(value, dict):
            if "id" in value and "serial" in value:
                instances[value["serial"]] = value["id"]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for zone in ("active", "bench", "hand", "discard", "prize"):
        visit(player.get(zone))
    for card in public_cards:
        visit(card)
    return Counter(instances.values())


def predict_hidden_cards(observation, own_deck, possible_opponent_decks, rng):
    """Sample compatible deck priors; no access to the opponent's selected deck.

    Strict accounting skips contexts whose temporary zones need more history.
    The caller records a fallback for those unsupported belief contexts.
    """
    state = observation["current"]
    own = state["players"][state["yourIndex"]]
    # CABT redacts the opponent's hand. Keep that boundary even for enriched replays.
    opponent = {**state["players"][1 - state["yourIndex"]], "hand": None}
    if any(card is None for player in (own, opponent) for card in player["active"]):
        raise ValueError("face-down setup requires a dedicated belief handler")
    stadium = state.get("stadium") or []

    def visible(player, seat):
        return known_cards(player, (card for card in stadium if card["playerIndex"] == seat))

    own_known = visible(own, state["yourIndex"])
    opponent_known = visible(opponent, 1 - state["yourIndex"])
    compatible = [deck for deck in possible_opponent_decks if not (opponent_known - Counter(deck))]
    if not compatible:
        raise ValueError("public opponent cards do not match the configured deck priors")
    opponent_prior = rng.choice(compatible)

    def partition(player, deck, known, opponent_side):
        if known - Counter(deck):
            raise ValueError("visible cards do not fit the configured deck")
        remaining = list((Counter(deck) - known).elements())
        unknown_prizes = sum(card is None for card in player["prize"])
        visible_hand = player.get("hand") or []
        unknown_hand = player["handCount"] - len(visible_hand) if opponent_side else 0
        needed = player["deckCount"] + unknown_prizes + unknown_hand
        if len(remaining) != needed:
            raise ValueError("visible-card accounting requires a context/history extension")
        rng.shuffle(remaining)
        prizes = [remaining.pop() if card is None else card["id"] for card in player["prize"]]
        hand = [card["id"] for card in visible_hand]
        hand.extend(remaining.pop() for _ in range(unknown_hand))
        return remaining, prizes, hand

    own_remaining, own_prizes, _ = partition(own, own_deck, own_known, False)
    opponent_remaining, opponent_prizes, opponent_hand = partition(
        opponent, opponent_prior, opponent_known, True)
    # cg.api uses the actual visible deck for deck-search selections.
    if observation["select"].get("deck") is not None:
        own_remaining = []
    return {"your_deck": own_remaining, "your_prize": own_prizes,
            "opponent_deck": opponent_remaining, "opponent_prize": opponent_prizes,
            "opponent_hand": opponent_hand, "opponent_active": []}
