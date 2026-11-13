# Team agent template

Copy this entire folder, rename it for your model, and keep your inference code and weights together.

```text
my_agent/
  agent.py                  # exported agent(observation) entry point
  model.py                  # your model definition and loading code
  requirements.txt          # your inference dependencies
  check_agent.py            # small interface check
  weights/
    demo_weights.json       # replace with your trained weights
```

The included demo runs immediately. It ranks legal options using hand-set scores loaded from JSON. It demonstrates the integration contract and makes no claim of learned playing strength.

## Connect your model

1. Replace `DemoModel` in `model.py` with your model definition and weight loader. Preserve the preprocessing, feature order, vocabulary, and normalization used during training.
2. Load your model once at module scope in `agent.py`. Use paths based on `Path(__file__).resolve().parent` so files work from any notebook directory.
3. Replace `MODEL.score_options(observation)` with your inference. A policy can rank the supplied options; a value model needs a search/action-selection wrapper before it can choose a move.
4. Return a Python `list[int]` of distinct option indices with length between `minCount` and `maxCount`. Indices must be within the current option list. Return `[]` only when the minimum permits it. Selection order may matter for the game effect. Cast NumPy/PyTorch scalar indices with `int(...)`.
5. Install your requirements, run `python check_agent.py`, then test complete native games in the sample-agent notebook or team arena. The synthetic check does not construct the full observation your real model may need; replace its fixtures when adding richer preprocessing.

For PyTorch, the loader typically constructs your model, loads its matching `state_dict` with `torch.load(path, map_location="cpu", weights_only=True)`, calls `load_state_dict`, and switches to `eval()`. Wrap inference in `torch.inference_mode()`. Those names are integration guidance; the demo has no PyTorch dependency.

The optional `start_game(own_deck)` hook resets per-game history and receives only your own deck. The arena starts a fresh process each game and preserves it across that game's moves. It runs the script from this folder, so sibling imports and local assets work. The script receives only the acting player's observation; infer hidden information from that view.

## Arena settings

- Source: `Python script`
- Path: this folder, or its `agent.py`
- Entry point: `agent`
- Call style: `observation`
- Python: blank for the notebook interpreter, or the path to your own environment's Python

Send teammates the complete folder, including the model definition, trained weights, requirements, and any vocabulary/config files. Document which deck and feature version the model was trained on. Weight files must be available locally or fetched by your loader.
