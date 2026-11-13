# Kaggle-style RL/MCTS baseline

The [reinforcement-learning notebook](../reinforcement_learning.ipynb) now presents the original supplied code directly in cells, with short notes. All 22 original function/class definitions are unchanged, including the Transformer, feature encoders, MCTS, and batching helpers. It uses the original placeholder hidden-card assumptions, 64-candidate cap, candidate-loss scaling, and evaluation-before-training loop. Run sizes are configurable; small corrections cover the vocabulary bound, draws, partial batches, and native cleanup. `project.sample_sdk` prepares the original `cg.api` helper files with verified hashes.

The modules in this directory retain the earlier refactored adaptation described below. The notebook uses only their optional export helper; its training code is inline. Saved exports from the earlier notebook continue to work.

In Colab, open the notebook and use **Runtime → Run all**. Setup clones the public [project repository](https://github.com/le21-j/pokemon-tcg-fpga), installs Python requirements, and prepares the native runtime and original API helpers. See the [project README](../README.md) for setup instructions.

This adapts Kiyota's [Reinforcement Learning and MCTS sample code](https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code), published under Apache 2.0. The supplied copy has SHA-256 `1c2827a2f3afc73f06cb49a5d4279adc8e8d29335fd1940da7049122a8ee7ad4`. The Transformer encoder/decoder, sparse feature bags, MCTS priors, AdamW learning rate `3e-4`, Huber deltas `0.2`/`0.1`, and outcome/search target mixing follow that example. This project modifies runtime integration, feature bounds, batching, draw handling, and export; see the repository notices for attribution.

`model.py` retains the sample's feature architecture and Transformer value/move-score heads. It uses numeric option/zone definitions checked against the original [Kiyota cg-lib v9](https://www.kaggle.com/datasets/kiyotah/cg-lib/versions/9). Vocabulary sizes now follow native metadata rather than the sample's fixed 22,000 encoder vocabulary. Dictionary-to-view conversion replaces the Kaggle-only dataclass dependency. This model is a software baseline; the compact Deep Sets/FPGA implementation remains in its existing files.

`train.py` uses the project's verified native search binding. It samples compatible public deck priors, uses each leaf's acting-player deck, aligns values to the root player, counts belief/search fallbacks, and always ends native search/game sessions. Unsupported belief contexts use the learned move-score policy and contribute value targets without fabricated search-score targets.

Candidates span permitted selection counts, with combinations and reversed orders up to the configured cap. They do not exhaust all ordered selections. MCTS supplies bounded move-score advantage targets; completed wins/losses/draws supply terminal returns mixed with search values. This is search-guided self-play regression, rather than PPO or a policy-gradient implementation.

Partial batches are optimized, draw returns are zero, padded candidates are masked, score losses are averaged per decision, and checkpoints are saved after optimization. The exported folder contains a portable model module, inference script, inference weights, and requirements. It selects moves using the learned score head without MCTS, while the training notebook can evaluate the MCTS agent separately.

Run the small deterministic checks from the repository root:

```bash
.venv/bin/python -m project.rl_baseline.check
```

Native game and search interfaces are described in the [CABT documentation](https://matsuoinstitute.github.io/cabt/). Runtime source/version hashes are stored with the notebook's training report. The Python seed does not control native shuffles or coin tosses.
