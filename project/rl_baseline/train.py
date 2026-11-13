"""Search-guided self-play adaptation of the supplied Kaggle RL/MCTS notebook.

Adapted from Kiyota's Apache-2.0 sample; see the repository LICENSE and NOTICE.
Modified search integration, belief handling, targets, batching, and export.

Uses the project's verified raw native API. No Kaggle-only cg.api installation is
required. The Transformer/features are in model.py and export with the policy.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import shutil

import torch

from .model import (LearnInput, as_view, candidate_actions, eval_nn,
                    get_decoder_input, get_encoder_input)
from ..agent.native_search import known_cards, predict_hidden_cards
from ..arena import validate_action


@dataclass
class Sample:
    value: float
    policy: list
    encoder: object
    decoder: object
    policy_valid: bool = True


@dataclass
class Node:
    state: dict
    parent: object = None
    value: float = 0.0
    total: float = 0.0
    visits: int = 0
    children: list = field(default_factory=list)

    def backup(self, value):
        self.total += value
        self.visits += 1
        if self.parent is not None:
            self.parent.backup(value)


class MCTSAgent:
    def __init__(self, model, search_api, deck_priors, search_steps=2, max_candidates=32):
        if search_steps < 1 or max_candidates < 1:
            raise ValueError("search_steps and max_candidates must be positive")
        self.model, self.api, self.priors = model, search_api, list(deck_priors)
        self.search_steps, self.max_candidates = search_steps, max_candidates
        self.stats = Counter()
        self.own_deck = None

    def start_game(self, own_deck):
        self.own_deck = list(own_deck)
        self.stats.clear()

    def _sample(self, observation, own_deck):
        view = as_view(observation)
        actions = candidate_actions(view.select, self.max_candidates)
        encoder = get_encoder_input(view, own_deck)
        decoder = get_decoder_input(view, actions)
        value, scores = eval_nn(encoder, decoder, self.model)
        return actions, Sample(value, scores, encoder, decoder)

    def _node(self, state, parent, root_player, decks):
        node = Node(state, parent)
        observation = state["observation"]
        current = observation["current"]
        if current["result"] >= 0:
            node.value = 0.0 if current["result"] == 2 else (1.0 if current["result"] == root_player else -1.0)
            sample = None
        else:
            actions, sample = self._sample(observation, decks[current["yourIndex"]])
            node.value = sample.value if current["yourIndex"] == root_player else -sample.value
            maximum = max(sample.policy)
            weights = [math.exp((p - maximum) * 10.0) for p in sample.policy]
            denominator = sum(weights)
            node.children = [{"action": a, "prior": w / denominator, "node": None}
                             for a, w in zip(actions, weights)]
        node.backup(node.value)
        return node, sample

    @torch.inference_mode()
    def select(self, observation, rng):
        if self.own_deck is None:
            raise RuntimeError("start_game must provide this actor's deck")
        actions, fallback = self._sample(observation, self.own_deck)
        fallback.policy_valid = False
        if len(actions) == 1:
            self.stats["forced_decisions"] += 1
            return actions[0], fallback
        root_player = observation["current"]["yourIndex"]
        opponent = observation["current"]["players"][1 - root_player]
        compatible = [d for d in self.priors if not (known_cards(opponent) - Counter(d))]
        try:
            if not compatible:
                raise ValueError("no compatible opponent deck prior")
            opponent_deck = rng.choice(compatible)
            predictions = predict_hidden_cards(observation, self.own_deck, [opponent_deck], rng)
        except ValueError:
            self.stats["belief_fallbacks"] += 1
            return actions[max(range(len(actions)), key=lambda i: fallback.policy[i])], fallback

        try:
            decks = {root_player: self.own_deck, 1 - root_player: opponent_deck}
            root, sample = self._node(self.api.begin(observation, predictions), None, root_player, decks)
            for _ in range(self.search_steps):
                current = root
                while current.children:
                    direction = 1 if current.state["observation"]["current"]["yourIndex"] == root_player else -1
                    def ucb(child):
                        node = child["node"]
                        visits = node.visits if node is not None else 0
                        mean = node.total / visits if visits else current.total / current.visits
                        return direction * mean + 0.4 * math.sqrt(current.visits) * child["prior"] / (1 + visits)
                    child = max(current.children, key=ucb)
                    if child["node"] is None:
                        state = self.api.step(current.state["searchId"], child["action"])
                        child["node"], _ = self._node(state, current, root_player, decks)
                        self.stats["expanded_nodes"] += 1
                        break
                    current = child["node"]
                else:
                    current.backup(current.value)

            visited = [c for c in root.children if c["node"] is not None]
            best = max(visited, key=lambda c: c["node"].visits)
            minimum = min(c["node"].total / c["node"].visits for c in visited)
            sample.value = root.total / root.visits
            sample.policy = [max(-1.0, min(1.0,
                (c["node"].total / c["node"].visits if c["node"] is not None else minimum - 0.03) - sample.value))
                for c in root.children]
            self.stats["searched_decisions"] += 1
            return best["action"], sample
        except ValueError:
            self.stats["search_fallbacks"] += 1
            return actions[max(range(len(actions)), key=lambda i: fallback.policy[i])], fallback
        finally:
            self.api.end()

    def __call__(self, observation, rng):
        return self.select(observation, rng)[0]


def label_game(samples, result, mixing=0.9):
    """Observer-relative outcome/search targets, including zero for drawn games."""
    labeled = []
    for player, history in enumerate(samples):
        value = 0.0 if result == 2 else (1.0 if result == player else -1.0)
        for sample in reversed(history):
            label = (value + sample.value) * 0.5
            value = value * mixing + sample.value * (1.0 - mixing)
            sample.value = label
            labeled.append(sample)
    return labeled


def self_play(game, sim, model, search_api, deck_pool, games=2, seed=7,
              search_steps=2, max_candidates=32, max_steps=10000, progress=print):
    rng = random.Random(seed)
    samples, reports = [], []
    priors = list(deck_pool.values())
    for number in range(games):
        decks = [list(rng.choice(priors)) for _ in range(2)]
        actors = [MCTSAgent(model, search_api, priors, search_steps, max_candidates) for _ in range(2)]
        for actor, deck in zip(actors, decks):
            actor.start_game(deck)
        observation, start = game.battle_start(*decks)
        histories = [[], []]
        try:
            if start.errorPlayer >= 0 or observation is None:
                raise ValueError(f"native deck error {start.errorType}")
            decisions = 0
            while observation["current"]["result"] < 0:
                if decisions >= max_steps:
                    raise RuntimeError("self-play exceeded the step limit; no terminal labels written")
                player = observation["current"]["yourIndex"]
                action, sample = actors[player].select(observation, rng)
                validate_action(observation, action)
                histories[player].append(sample)
                observation = game.battle_select(action)
                decisions += 1
            result = observation["current"]["result"]
            if result not in (0, 1, 2):
                raise ValueError(f"unexpected terminal result {result}")
            samples.extend(label_game(histories, result))
            counts = Counter()
            for actor in actors:
                counts.update(actor.stats)
            reports.append({"game": number + 1, "result": result, "decisions": decisions, **counts})
            if progress:
                progress(f"Self-play {number + 1}/{games}: result={result}, decisions={decisions}, searched={counts['searched_decisions']}")
        finally:
            if sim.Battle.battle_ptr:
                game.battle_finish()
                sim.Battle.battle_ptr = None
    return samples, reports


def optimize(model, samples, optimizer, batch_size=16, epochs=1, seed=7):
    if not samples or batch_size < 1 or epochs < 1:
        raise ValueError("Provide samples and positive batch/epoch counts")
    device = next(model.parameters()).device
    rng, reports = random.Random(seed), []
    model.train()
    for epoch in range(epochs):
        shuffled = list(samples)
        rng.shuffle(shuffled)
        losses = []
        for offset in range(0, len(shuffled), batch_size):
            batch = shuffled[offset:offset + batch_size]  # Keep the final partial batch.
            count = len(batch)
            width = max(len(s.policy) for s in batch)
            encoder, decoder = LearnInput(), LearnInput()
            target_scores, masks = [], []
            for sample in batch:
                encoder.add(sample.encoder)
                decoder.add(sample.decoder)
                padding = width - len(sample.policy)
                target_scores.append(sample.policy + [0.0] * padding)
                masks.append([float(sample.policy_valid)] * len(sample.policy) + [0.0] * padding)
                decoder.offset.extend([len(decoder.index)] * padding)

            def tensors(sparse):
                return (torch.tensor(sparse.index, dtype=torch.int64, device=device),
                        torch.tensor(sparse.value, dtype=torch.float32, device=device),
                        torch.tensor(sparse.offset, dtype=torch.int64, device=device))
            values, scores = model(*tensors(encoder), *tensors(decoder))
            targets = torch.tensor([s.value for s in batch], dtype=torch.float32, device=device).view(count, 1)
            score_targets = torch.tensor(target_scores, dtype=torch.float32, device=device)
            mask = torch.tensor(masks, dtype=torch.float32, device=device)
            value_loss = torch.nn.functional.huber_loss(values, targets, delta=0.2)
            score_errors = torch.nn.functional.huber_loss(scores, score_targets, delta=0.1, reduction="none")
            score_loss = ((score_errors * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
            loss = value_loss + score_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((value_loss.item(), score_loss.item()))
        reports.append({"epoch": epoch + 1, "updates": len(losses),
                        "value_loss": sum(v for v, _ in losses) / len(losses),
                        "score_loss": sum(p for _, p in losses) / len(losses)})
    model.eval()
    return reports


def save_checkpoint(path, model, optimizer, vocabulary, deck, iteration):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": model.config, "vocabulary": vocabulary,
                "deck": list(deck), "iteration": iteration, "schema": "kaggle-transformer-v1"}, path)
    return path


def export_agent(checkpoint, destination, max_candidates=32):
    destination = Path(destination)
    (destination / "weights").mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).with_name("model.py"), destination / "model.py")
    for name in ("LICENSE", "NOTICE"):
        notice = Path(__file__).resolve().parents[2] / name
        if notice.is_file():
            shutil.copy2(notice, destination / name)
    inference = torch.load(checkpoint, map_location="cpu", weights_only=True)
    inference.pop("optimizer", None)
    torch.save(inference, destination / "weights/model.pt")
    (destination / "requirements.txt").write_text(f"torch=={torch.__version__.split('+')[0]}\n")
    (destination / "agent.py").write_text('''"""Exported Transformer move-score policy, without MCTS at inference."""
from pathlib import Path
import torch
from model import MyModel, configure_vocabulary, as_view, candidate_actions, get_encoder_input, get_decoder_input, eval_nn

HERE = Path(__file__).resolve().parent
checkpoint = torch.load(HERE / "weights/model.pt", map_location="cpu", weights_only=True)
configure_vocabulary(**checkpoint["vocabulary"])
MODEL = MyModel(**checkpoint["model_config"])
MODEL.load_state_dict(checkpoint["model"])
MODEL.eval()
torch.set_num_threads(1)
OWN_DECK = list(checkpoint["deck"])

def start_game(own_deck):
    global OWN_DECK
    OWN_DECK = list(own_deck)

@torch.inference_mode()
def agent(observation):
    view = as_view(observation)
    actions = candidate_actions(view.select, LIMIT)
    encoder = get_encoder_input(view, OWN_DECK)
    decoder = get_decoder_input(view, actions)
    _, scores = eval_nn(encoder, decoder, MODEL)
    return actions[max(range(len(actions)), key=lambda i: scores[i])]
'''.replace("LIMIT", str(max_candidates)))
    (destination / "README.md").write_text(
        "# Exported RL sample agent\n\nSelect Python script in arena.ipynb and point to this folder. "
        "The callable is agent(observation). Install requirements.txt in the chosen Python environment. "
        "Weights and model code are included. This export uses the learned move-score head without MCTS; "
        "training uses MCTS-guided self-play. A small smoke run does not establish playing strength.\n")
    return destination
