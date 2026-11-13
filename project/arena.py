"""Configurable agents, legal deck pools, and first-to-three native CABT series."""

from collections import Counter
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import time
import uuid
import zipfile

import torch

from .agent.features import SCHEMA_ID, position_from_observation
from .agent.model import DeepSetsValue
from .agent.native_search import predict_hidden_cards
from .collect_matches import DEFAULT_DECK, ENGINE_ROOT, choose_random


EXAMPLE_DECKS = {
    "mega-lucario": ("mega-lucario-ex-deck", 1),
    "mega-abomasnow": ("mega-abomasnow-ex-deck", 1),
    "dragapult": ("dragapult-ex-deck", 2),
}


def load_deck_pool(directory=None):
    directory = Path(directory or ENGINE_ROOT.parent / "example-decks")
    directory.mkdir(parents=True, exist_ok=True)
    decks = {"sample-water": list(DEFAULT_DECK)}
    for name, (slug, version) in EXAMPLE_DECKS.items():
        path = directory / f"{name}-v{version}.json"
        if not path.exists():
            url = f"https://www.kaggle.com/api/v1/datasets/download/kiyotah/{slug}?datasetVersionNumber={version}"
            archive = subprocess.run(["curl", "-fsSL", "--retry", "2", url],
                                     check=True, capture_output=True).stdout
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                files = [p for p in bundle.namelist() if p.endswith("deck.csv")]
                if len(files) != 1:
                    raise ValueError(f"expected one deck.csv in {slug}")
                raw = bundle.read(files[0])
            cards = [int(line.strip()) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
            if len(cards) != 60:
                raise ValueError(f"{name} does not contain 60 card IDs")
            document = {"name": name, "cards": cards, "source": url,
                        "csv_sha256": hashlib.sha256(raw).hexdigest()}
            temporary = path.with_suffix(".part")
            temporary.write_text(json.dumps(document, indent=2) + "\n")
            temporary.replace(path)
        decks[name] = json.loads(path.read_text())["cards"]
    return decks


def check_deck_shape(deck):
    if len(deck) != 60 or any(type(card) is not int or card <= 0 for card in deck):
        raise ValueError("a deck must contain exactly 60 positive integer card IDs")


def validate_deck_pool(game, sim, pool):
    for name, deck in pool.items():
        check_deck_shape(deck)
        _, start = game.battle_start(deck, DEFAULT_DECK)
        try:
            if start.errorPlayer >= 0:
                raise ValueError(f"deck {name}: native error {start.errorType}, player {start.errorPlayer}")
        finally:
            if sim.Battle.battle_ptr:
                game.battle_finish()
                sim.Battle.battle_ptr = None


def priority_action(observation, rng):
    """A simple legal baseline that prefers development and attacks in MAIN."""
    selection = observation["select"]
    if selection["context"] == 0 and selection["minCount"] == selection["maxCount"] == 1:
        for kind in (9, 8, 7, 13, 10, 14):  # evolve, attach, play, attack, ability, end
            indices = [i for i, option in enumerate(selection["option"]) if option["type"] == kind]
            if indices:
                return [rng.choice(indices)]
    return choose_random(observation, rng)


class ValueSearchAgent:
    """One sampled hidden world, one native step per legal single-choice option.

    Leaf values use the acting player's view and are aligned to the root player.
    Unsupported belief/selection contexts use a counted legal baseline fallback.
    """
    def __init__(self, model, search_api, opponent_deck_priors, max_candidates=128):
        self.model = model.eval()
        self.api = search_api
        self.priors = [list(deck) for deck in opponent_deck_priors]
        self.max_candidates = max_candidates
        self.own_deck = None
        self.stats = Counter()

    @classmethod
    def from_checkpoint(cls, path, search_api, opponent_deck_priors, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["metadata"]["schema_id"] != SCHEMA_ID:
            raise ValueError("checkpoint feature schema mismatch")
        model = DeepSetsValue()
        model.load_state_dict(checkpoint["model"])
        return cls(model, search_api, opponent_deck_priors, **kwargs)

    def start_game(self, own_deck):
        self.own_deck = list(own_deck)
        self.stats.clear()

    @torch.inference_mode()
    def __call__(self, observation, rng):
        selection = observation["select"]
        count = len(selection["option"])
        if (selection["minCount"] != 1 or selection["maxCount"] != 1 or
                count < 2 or count > self.max_candidates):
            self.stats["selection_fallbacks"] += 1
            return priority_action(observation, rng)
        if self.own_deck is None:
            raise RuntimeError("start_game must supply this agent's own deck")
        try:
            predictions = predict_hidden_cards(observation, self.own_deck, self.priors, rng)
        except ValueError:
            self.stats["belief_fallbacks"] += 1
            return priority_action(observation, rng)
        root_player = observation["current"]["yourIndex"]
        best_index, best_score = 0, -float("inf")
        try:
            root = self.api.begin(observation, predictions)
            for index in range(count):
                child = self.api.step(root["searchId"], [index])
                try:
                    child_observation = child["observation"]
                    state = child_observation["current"]
                    if state["result"] >= 0:
                        score = 0.0 if state["result"] == 2 else (1.0 if state["result"] == root_player else -1.0)
                    else:
                        score = self.model.evaluate_position(position_from_observation(child_observation)).item()
                        score = max(-1.0, min(1.0, score))
                        if state["yourIndex"] != root_player:
                            score = -score
                    if score > best_score:
                        best_index, best_score = index, score
                    self.stats["evaluated_actions"] += 1
                finally:
                    self.api.release(child["searchId"])
            self.stats["searched_decisions"] += 1
            return [best_index]
        finally:
            self.api.end()


@dataclass
class Agent:
    name: str
    action: object  # Callable(observation, random.Random) -> list[int]
    deck: object = "sample-water"  # pool name, "random", 60 IDs, or callable(pool, rng, game_number)


def select_deck(agent, pool, rng, game_number):
    choice = agent.deck
    if callable(choice):
        choice = choice({name: list(cards) for name, cards in pool.items()}, rng, game_number)
    if isinstance(choice, str):
        name = rng.choice(list(pool)) if choice == "random" else choice
        deck = list(pool[name])
    else:
        name, deck = "custom", list(choice)
    check_deck_shape(deck)
    return name, deck


def validate_action(observation, action):
    selection = observation["select"]
    if (not isinstance(action, list) or any(type(i) is not int for i in action) or
            len(set(action)) != len(action) or
            not selection["minCount"] <= len(action) <= selection["maxCount"] or
            any(i < 0 or i >= len(selection["option"]) for i in action)):
        raise ValueError(f"agent returned an invalid selection: {action}")


def play_game(game, sim, agents, selected_decks, seats, action_rngs, max_steps):
    for agent, (_, deck) in zip(agents, selected_decks):
        if hasattr(agent.action, "start_game"):
            agent.action.start_game(list(deck))
    decks_by_seat = [selected_decks[index][1] for index in seats]
    opening_hands = {}
    started = time.perf_counter()
    observation, start = game.battle_start(*decks_by_seat)
    try:
        if start.errorPlayer >= 0 or observation is None:
            raise ValueError(f"native deck/start error {start.errorType} for seat {start.errorPlayer}")
        steps = 0
        while observation["current"]["result"] < 0:
            if steps >= max_steps:
                raise RuntimeError("game step limit reached; series remains incomplete")
            seat = observation["current"]["yourIndex"]
            index = seats[seat]
            hand = observation["current"]["players"][seat].get("hand")
            if hand and index not in opening_hands:
                opening_hands[index] = [card["id"] for card in hand]
            action = agents[index].action(deepcopy(observation), action_rngs[index])
            validate_action(observation, action)
            observation = game.battle_select(action)
            steps += 1
        native_result = observation["current"]["result"]
        if native_result not in (0, 1, 2):
            raise ValueError(f"unexpected terminal result {native_result}")
        winner = None if native_result == 2 else seats[native_result]
        return {"winner_index": winner, "decisions": steps,
                "turn": observation["current"].get("turn"),
                "seconds": time.perf_counter() - started, "opening_hands": opening_hands,
                "agent_stats": {agent.name: dict(getattr(agent.action, "stats", {})) for agent in agents}}
    finally:
        if sim.Battle.battle_ptr:
            game.battle_finish()
            sim.Battle.battle_ptr = None


def run_series(game, sim, agent_a, agent_b, deck_pool, seed=7,
               reselect_decks_each_game=False, max_total_games=20, max_steps=10000, progress=print):
    """First to three wins, alternating seats. Draws are replayed and counted separately."""
    if agent_a.name == agent_b.name:
        raise ValueError("agents need distinct display names")
    if agent_a.action is agent_b.action and hasattr(agent_a.action, "start_game"):
        raise ValueError("stateful agents need separate action instances")
    if max_total_games < 3 or max_steps < 1:
        raise ValueError("allow at least three total games and a positive step limit")
    agents = (agent_a, agent_b)
    deck_rng = random.Random(seed)
    action_rngs = [random.Random(seed + 1009), random.Random(seed + 2027)]
    fixed_decks = None
    wins, games = [0, 0], []
    while max(wins) < 3 and len(games) < max_total_games:
        number = len(games) + 1
        if fixed_decks is None or reselect_decks_each_game:
            selected = [select_deck(agent, deck_pool, deck_rng, number) for agent in agents]
            fixed_decks = selected
        else:
            selected = fixed_decks
        seats = [0, 1] if number % 2 else [1, 0]
        result = play_game(game, sim, agents, selected, seats, action_rngs, max_steps)
        winner = result["winner_index"]
        if winner is not None:
            wins[winner] += 1
        result.update(game=number, seat0=agents[seats[0]].name, seat1=agents[seats[1]].name,
                      decks={agent.name: {"name": choice[0], "cards": choice[1]}
                             for agent, choice in zip(agents, selected)},
                      winner="draw" if winner is None else agents[winner].name,
                      score=list(wins))
        games.append(result)
        if progress:
            progress(f"Game {number}: {result['winner']} | {agent_a.name} {wins[0]} - {wins[1]} {agent_b.name}")
    finished = max(wins) == 3
    return {"series_id": uuid.uuid4().hex, "format": "first_to_3_wins",
            "agents": [agent.name for agent in agents], "wins": wins,
            "draws": sum(g["winner_index"] is None for g in games), "completed": finished,
            "winner": agents[wins.index(3)].name if finished else None,
            "deck_selection": "per_game" if reselect_decks_each_game else "per_series",
            "action_and_deck_seed": seed, "native_shuffle_seed": None, "games": games}
