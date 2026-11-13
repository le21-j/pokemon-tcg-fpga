"""Arena entry point. Replace DemoModel with your own trained model loader."""

from pathlib import Path
import math

from model import DemoModel


HERE = Path(__file__).resolve().parent
# Load once when the arena starts this game's Python worker.
MODEL = DemoModel.load(HERE / "weights" / "demo_weights.json")
OWN_DECK = []


def start_game(own_deck):
    """Optional arena hook: reset history and receive your own 60-card deck."""
    global OWN_DECK
    OWN_DECK = list(own_deck)
    # Reset any per-game history or recurrent state here.


def agent(observation):
    """Return option INDICES, not card IDs, scores, or an action name."""
    selection = observation["select"]
    options = selection["option"]
    minimum, maximum = selection["minCount"], selection["maxCount"]
    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("Invalid selection bounds in the observation")

    # Replace this with your preprocessing + model inference. One score per option.
    scores = MODEL.score_options(observation)
    if len(scores) != len(options) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("The model must produce one finite score per available option")

    # This example chooses maxCount. Your policy may choose any permitted count,
    # including zero when minCount is zero. Keep the order if the choice needs it.
    ranked = sorted(range(len(options)), key=lambda i: (-float(scores[i]), i))
    return [int(index) for index in ranked[:maximum]]
