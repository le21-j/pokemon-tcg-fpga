# Pokémon TCG AI with FPGA inference

Project proposal for EEE 405/591. Updated November 12, 2026.

Planning update, October 29, 2026: the target board is the **DE10-Lite**, and training resources include **ASU Sol**. Read the [architecture and training plan](architecture-and-training-plan.md) for the proposed model, training stages, software/RTL split, and evaluation gates. The [competition research](competition-research.md) distinguishes verified interfaces from data still awaiting inspection.

The November 8 [model comparison research](model-options-research.md) compares PPO, imitation learning, Monte Carlo value learning, search distillation, pooled entity networks, and boosted trees for this task and board.

The current November 12 [NNUE-style Deep Sets plan](openjev-tcg-adaptation-plan.md) defines an **8,865-parameter position evaluator**, incremental updates and saved positions on the DE10-Lite, search on the PC, and completed-match value learning on Sol. The [hardware co-design plan](architecture-and-training-plan.md) summarizes numeric, memory, protocol, and verification budgets. The [small-model comparison](small-model-architecture-research.md) covers value-model alternatives. LSTMs remain excluded. The [Open-Jev source review](openjev-source-research.md) records earlier SDK research.

## Running the Python prototype

The notebooks run independently. **In Colab, open a notebook and select Runtime → Run all.** The first cell clones the public [pokemon-tcg-fpga repository](https://github.com/le21-j/pokemon-tcg-fpga) and installs the notebook requirements automatically. No GitHub token or separate repository upload is required. Game libraries, API helpers, and example decks download when needed.

Custom team agents supply their own inference code, weights, and dependencies. Optional setup fields can point to a checkout or mount Drive. Local runs reuse the checkout/environment unless `INSTALL_LOCALLY` is enabled. Start a fresh Colab runtime to fetch newly published source changes.

Colab uses [requirements-colab.txt](requirements-colab.txt), allowing compatible preinstalled PyTorch/pandas versions without forcing an upgrade or replacing a working CUDA build. The existing exact PyTorch requirement remains available for local reproducibility.

After editing the shared setup helper, synchronize the notebook setup cells:

```bash
.venv/bin/python -m project.build_colab_notebooks
.venv/bin/python -m project.check_colab_setup
```

- [arena.ipynb](arena.ipynb) is the shared team testing notebook. Colab-style forms select each competitor's Python file/folder or a built-in agent, deck, and match settings. Run the forms from top to bottom or use **Run All**. Each first-to-three series saves a JSON report.
- [sample_agent.ipynb](sample_agent.ipynb) inspects the shareable agent template, checks a real native observation, plays a series against Random/Priority, and creates a ZIP for teammates.
- [reinforcement_learning.ipynb](reinforcement_learning.ipynb) presents the supplied Transformer/MCTS sample directly in notebook cells, with brief notes for each section. All 22 sample function/class definitions are unchanged. Small setup, vocabulary, draw, batch, and cleanup fixes allow local/Colab execution. It retains the original evaluation/self-play/optimization flow and adds an optional arena-folder export. Defaults use small game counts.
- [deep_sets_walkthrough.ipynb](deep_sets_walkthrough.ipynb) explains entity tensors, grouped pooling, value prediction, full/incremental attachment updates, parent restoration, and gradients through both learned components.

Share a copy of [agent_template/](agent_template/README.md) with teammates. It includes an inference entry point, a model loader, example weights, requirements, and a runnable interface check. Its hand-set weighted scoring demo runs immediately. Teammates replace the model and weights while keeping the legal-option interface. The RL notebook's export also contains the trained model definition and inference weights; its move-score policy runs without MCTS at inference. The earlier refactored RL helpers and source attribution are documented in [rl_baseline/README.md](rl_baseline/README.md).

The arena accepts external Python scripts, Random, Priority, and saved Deep Sets value checkpoints. For a teammate's model, select **Python script** and enter its `.py` path or folder. The script loads its own architecture, weights, and dependencies. The default interface is `agent(observation) -> list[int]`, returning legal option indices. Integration options support a different exported function name, a second RNG or configuration argument, and a separate Python executable. Install each teammate's requirements in the selected Python environment. Paths may be absolute or relative to this repository; Colab paths refer to uploaded files or mounted Drive files.

Folder selection looks for `agent.py`, then `main.py`, then `__init__.py`, then a sole `.py` file. Supply an explicit file if another entry is needed. Each script runs in a separate process with its folder as the working directory, allowing local imports and relative weight paths. A fresh process starts each game and stays alive across that game's moves. An optional `start_game(own_deck)` hook receives its own selected deck. A standalone training/CLI program needs a small wrapper exposing the policy function.

The notebook offers named decks, random selection from the native-validated pool, and custom JSON files containing 60 card IDs. The underlying Python arena also accepts deck-selection callbacks. Decks stay fixed within a series by default, with an option to reselect each game. Every game starts through a fresh native `battle_start()`. Seats alternate, draws are replayed, and the first agent reaching three wins ends the series. The editable repeated-draw guard reports an incomplete series if reached. Script failures/timeouts stop the run and report the error without assigning a winner.

Script response times include process communication, preprocessing, search/policy work, and inference. They exclude worker startup and import-time weight loading. Reports hash entry scripts, but separately loaded weights are managed by each script; use a competitor name that identifies the model version.

Saved value models choose actions through a native one-step search baseline using compatible-deck priors. Its coverage/fallback counters are displayed. The native search/metadata bindings were verified against the original public Kiyota `cg-lib` v9 wrapper, then tested with the pinned macOS engine. Python seeds control agent/deck/belief sampling; native shuffles have no exposed seed argument. The notebook records first-observed hand IDs/fingerprints for inspecting fresh deals. Source wrappers and example-deck provenance are linked inside the notebook.

Run these commands from the repository root using its existing Python 3.12 environment:

```bash
uv pip install --python .venv/bin/python -r project/requirements.txt
.venv/bin/python -m project.energy_attachment
.venv/bin/python -m project.check_arena
.venv/bin/python project/agent_template/check_agent.py
.venv/bin/python -m project.rl_baseline.check
.venv/bin/python -m project.check_sample_notebook
.venv/bin/python -m project.check_colab_setup
```

The attachment example verifies full/incremental evaluation, exact parent restoration, a second branch, entity permutation, empty/variable-size batches, padding, joint encoder/head gradients, and rejection of a cache after weights change. Floating-point updates use a numerical tolerance. The integer RTL reference is a later milestone.

Generate native simulator data through the same `battle_start`, `battle_select`, and `battle_finish` APIs used by the provided Kaggle notebook:

```bash
.venv/bin/python -m project.collect_matches --download --games 4 --output project/results/demo/matches.jsonl
```

`--download` uses curl to fetch the platform-specific native library and Python game wrapper from pinned Kaggle commit `998b15c14b1b77aafb1843bed63858455035affd`. It preserves the repository license and records downloaded-file hashes. The runtime is stored under ignored `project/.runtime/`. Use `--cg-lib /path/to/cg-lib` to select an existing bundle. Runtime source hashes are saved with the records; an external bundle without a matching pinned manifest has an unverified version. Each worker process should own one active game. The action RNG is seeded; the raw API exposes no native deal/shuffle seed.

The collector writes acting-player observations before each action and observer-relative terminal returns, including zero for a draw. It excludes opaque search serialization. These are individual native games, and the `match_id` equals `game_id` in this collector. Kaggle's BO3 wrapper remains a separate integration. Output paths must be new so an existing dataset/checkpoint is preserved.

When an ordinary attachment is captured, check the actual before/after observations and the complete recorded game traces:

```bash
.venv/bin/python -m project.energy_attachment --before project/results/demo/energy-before.json --after project/results/demo/energy-after.json --dataset project/results/demo/matches.jsonl
.venv/bin/python -m project.train_baseline --data project/results/demo/matches.jsonl --output project/checkpoints/demo-huber.pt --loss huber --epochs 12
```

The learner uses AdamW at `3e-4`. `--loss huber` adapts the notebook's value-loss delta of `0.2`; the default `--loss mse` follows the position-value plan. Both train the encoder and value head jointly on completed-game returns. Train/validation separation groups complete games, retaining both players together. The best validation checkpoint includes model/optimizer/RNG state, feature definitions, dataset hash, and split IDs. A JSON report records all epochs. Add `--checkpoint project/checkpoints/demo-huber.pt` to the attachment command to check that saved model. Checkpoint loading is implemented; interrupted-run resume is a later addition.

The observed-resources-v0 feature schema uses HP, attachment counts, roles, statuses, counts, and selection type. Card identities/mechanics need the audited metadata encoder, and the 32 history slots are currently zero. Native games, floating-point model tests, the first-to-three arena, and one-step model-agent search have run locally on macOS ARM64. Deeper/fixed-observer search, quantized inference, Kaggle BO3 integration, FPGA execution, and Sol execution remain planned. Raw-dictionary native metadata/search bindings are now implemented in `agent/native_search.py`; the complete original `cg.api` dataclass wrapper is separate. See the [API and optimization findings](kaggle-api-implementation-research.md) for source references and recorded evidence.

| Implementation | Purpose |
| --- | --- |
| [features.py](agent/features.py) | Versioned raw-observation feature prototype and stable card instances |
| [model.py](agent/model.py) | Batched differentiable Deep Sets value model |
| [evaluator.py](agent/evaluator.py) | Floating-point incremental cache, transactional updates, save/restore |
| [energy_attachment.py](energy_attachment.py) | Runnable full/incremental/restore assertions and game-trace checks |
| [collect_matches.py](collect_matches.py) | Pinned runtime setup, native games, JSONL observations/outcomes |
| [train_baseline.py](train_baseline.py) | Grouped-split AdamW value regression with MSE/Huber and checkpoints |
| [arena.py](arena.py) | Configurable agents/deck selection, legal native games, first-to-three scorekeeping |
| [script_agent.py](script_agent.py) | File/folder policies in separate Python workers, local assets, signatures, timeouts, cleanup |
| [agent_template/](agent_template/README.md) | Standalone teammate handoff: inference script, model, example weights and checks |
| [rl_baseline/](rl_baseline/README.md) | Portable Transformer sample, MCTS-guided self-play, optimization and agent export |
| [sample_sdk.py](sample_sdk.py) | Original cg.api helper preparation with pinned native bindings and hash checks |
| [check_sample_notebook.py](check_sample_notebook.py) | Verify the notebook retains the original sample's 22 function/class definitions |
| [colab_setup.py](colab_setup.py) | Public Git clone and automatic Colab dependency installation |
| [build_colab_notebooks.py](build_colab_notebooks.py) | Synchronize the small setup cells across notebooks |
| [check_colab_setup.py](check_colab_setup.py) | Direct Git clone, caching, recovery and automatic installation checks |
| [native_search.py](agent/native_search.py) | Verified native search/metadata ABI and compatible hidden-card sampling |
| [check_arena.py](check_arena.py) | Scoring, draw replay, seat mapping, action validation, external script isolation and failure checks |

## Project idea

Develop a learned agent for the Pokémon TCG Kaggle competition, then deploy its model on a physical FPGA. Compare software and FPGA inference to evaluate whether the agent can return useful decisions within a practical time budget for TCG matches.

The project connects two results: an agent evaluated through gameplay and a working hardware implementation measured on the board. Low latency and strong gameplay are separate requirements, so the evaluation must report both.

## Competition target

The working target is the [PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview/description). It continues the original Simulation competition, which has ended. The Playground lists an entry deadline of January 1, 2027 and a final submission deadline of January 8, 2027. The October 29 research note records environment-version differences and inconsistent historical deadline wording on the original page.

The Playground agent receives game observations and available choices, then returns the indices of its selections. Kaggle evaluates agents through matches against other agents. The [cabt engine documentation](https://matsuoinstitute.github.io/cabt/) describes the observation and selection interfaces.

## Proposed scope

- Build a software baseline that completes matches in the competition simulator.
- Train a compact shared card encoder and position-value MLP from completed-match returns, then improve the data through search-guided self-play on Sol.
- Build PC search that compares legal continuations using observer-visible position values and explicit hidden-card/chance assumptions.
- Prepare a software agent for Kaggle submission and record its match performance.
- Quantize the trained model and run its forward pass on a physical FPGA using the same learned weights.
- Maintain cached entity encodings and INT32 group sums on the FPGA, with atomic updates and parent restoration between PC search branches.
- Measure the quality, latency, and hardware resources of the resulting system.

The proposed Kaggle entry uses software inference. The FPGA demonstration runs locally, with the host responsible for the simulator, input encoding, and legal-action handling. Model inference is the initial hardware scope. Camera-based card recognition and automated physical card handling are outside this proposal.

## Inference flow

```text
Player observation and legal root options
  -> PC constructs private simulator search branches
  -> host synchronizes or updates the FPGA position
  -> FPGA updates cached card encodings and group sums
  -> value MLP returns a position evaluation
  -> PC restores the parent and compares alternatives
  -> simulator receives the selected legal root payload
```

Freeze the input encoding, output meaning, and numeric format before implementing the FPGA design. Keep a software reference for the quantized computation so hardware outputs can be checked directly.

## Evaluation plan

| Question | Evidence |
| --- | --- |
| Does the agent play useful games? | Wins, losses, and draws against fixed baseline opponents, with game counts and uncertainty. Record Kaggle rating separately when available. |
| Does quantization preserve quality? | Compare floating-point and quantized software outputs, selected actions, and match results. |
| Does the FPGA compute the intended model? | Require exact integer agreement for full builds, deltas, and restored positions against the software reference. |
| How long does hardware inference take? | Measure accelerator cycles and clock frequency, or equivalent board instrumentation. Report execution time separately from host transfers. |
| How long does a complete decision take? | Include hidden-world construction, simulator work, every evaluation/transfer, snapshot/restore, and root selection from observation receipt. |
| Does the design fit the FPGA? | Report board, clock, model size, numeric precision, and implementation resource use. |

Benchmark the same held-out root decisions with integer software, full-rebuild FPGA, and incremental FPGA evaluation. Compare both fixed search work and fixed wall time. Report sample count, median, p95, maximum observed latency, evaluated positions, search coverage, and fallback frequency. Separate startup from steady-state execution and include states with many choices or changed entities.

Before the final experiment, choose and justify a decision-time budget and an acceptable gameplay-quality loss. Report the fraction of decisions exceeding that budget and the accumulated decision time per game. Judge match-time feasibility using the complete decision pipeline. A simulator demonstration establishes feasibility for encoded game states; physical match use also depends on how those states enter the system.

## Milestones and deliverables

1. Confirm the competition variant, course requirements, available FPGA board, toolchain, and project deadline.
2. Run a baseline through complete matches and freeze a repeatable evaluation workload.
3. Train the position evaluator, establish CPU search, and run search-guided collection/training rounds on Sol.
4. Validate the integer evaluator and deploy full, incremental, and save/restore operations on the FPGA.
5. Demonstrate PC search using FPGA position values, prepare the software Kaggle package, and collect complete-decision measurements.
6. Deliver the model, training and inference code, FPGA sources and bitstream, benchmark procedure, and a report comparing gameplay quality, latency, and resource use.

The model comprises `16 -> 16 -> 16` shared entity layers, four 16-channel sums, and a `128 -> 64 -> 1` value MLP. Its Python implementation is available above. Proposed INT8 weights and INT32 biases occupy 9,156 bytes before state, snapshots, and control. Feature dimensions, hardware capacity, numeric precision, search budget, and latency require the planned experiments.

## Sources

- [Original PTCG AI Battle Challenge Simulation and timeline](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/evaluation)
- [PTCG AI Battle Challenge Playground and timeline](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview/description)
- [cabt engine documentation](https://matsuoinstitute.github.io/cabt/)
