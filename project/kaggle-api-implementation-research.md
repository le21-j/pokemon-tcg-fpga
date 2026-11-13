# Kaggle/CABT implementation research

Checked November 12, 2026. Primary evidence is the supplied notebook and official CABT documentation/source. The reviewed Kaggle source remains commit `998b15c14b1b77aafb1843bed63858455035affd`. This note distinguishes available APIs from demonstrated execution.

Latest implementation checks include 25 completed native games and 1,814 decision records, including a final provenance check of the normal collector entry point. Recorded file hashes are validated against the pinned runtime manifest. External `--cg-lib` bundles without a matching manifest are labeled with unknown revision/version. Using the MSE checkpoint, full and incremental evaluation agreed across all 1,420 positions of the 20-game pilot with maximum absolute value difference `2.24e-7`. The real Energy attachment's value difference was zero and saved-parent restoration passed exactly. These are floating-point correctness/training-pipeline results.

## Native search and arena update

The original author's public `kiyotah/cg-lib` version 9 dataset is accessible through its versioned download endpoint. Its `cg/sim.py` and `cg/api.py` establish the native signatures for `AgentStart`, `SearchBegin`, `SearchStep`, `SearchEnd`, `SearchRelease`, `AllCard`, and `AllAttack`. The inspected source hashes are `e83ca04300990267cd04385fb98ded670b517dfceb481a2bfa63ff433e2946ff` for `sim.py` and `ff163d90fc7f0feb61beb037f8931b887f98e78a7e891bdfc46b2f9c8abbfbff` for `api.py`. The project implements a small raw-dictionary binding from those ABI facts in `agent/native_search.py`, retaining the pinned platform-specific native library. [Original cg-lib v9](https://www.kaggle.com/datasets/kiyotah/cg-lib/versions/9), [versioned download](https://www.kaggle.com/api/v1/datasets/download/kiyotah/cg-lib?datasetVersionNumber=9)

The local binding loaded 1,431 card and 1,755 attack metadata records and executed native search branches. The arena completed a first-to-three series between two saved Deep Sets checkpoints, with custom and random deck selectors. The MSE and Huber agents invoked search at 135 and 98 decisions respectively in that test. This verifies gameplay integration with the prototype models. Those series are smoke checks for the runner, not estimates of comparative playing strength.

Original example deck lists are fetched by dataset version, then validated by native `battle_start`. Mega Lucario v1, Mega Abomasnow v1, Dragapult v2, and the existing sample deck all passed validation. Random deck mode samples from this legal pool. The notebook default Random/Priority series and all walkthrough cells executed successfully. Deterministic checks separately verified draws, alternating seat mapping, first-to-three stopping, incomplete draw guards, and cleanup. [Mega Lucario](https://www.kaggle.com/datasets/kiyotah/mega-lucario-ex-deck), [Mega Abomasnow](https://www.kaggle.com/datasets/kiyotah/mega-abomasnow-ex-deck), [Dragapult](https://www.kaggle.com/datasets/kiyotah/dragapult-ex-deck)

Arena value search uses one compatible hidden-deck sample and one native step per candidate. It evaluates the acting player's leaf view, aligning signs to the root player, and counts unsupported selection/belief fallbacks. This prototype's leaf convention differs from the planned fixed-observer search. Native shuffles remain uncontrolled by the Python seed, and each game initializes a fresh battle. Complete original-notebook reproduction, Kaggle BO3, Sol, and FPGA execution remain separate milestones.

## Verified notebook entry points

The supplied [reinforcement-learning-and-mcts-sample-code.ipynb](https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code) has SHA-256 `1c2827a2f3afc73f06cb49a5d4279adc8e8d29335fd1940da7049122a8ee7ad4`. Code-line references below count lines in its single code cell.

- Lines 12–30 append the first `/kaggle/input/**/cg-lib` match to `sys.path`, then import `cg.api` helpers and `cg.game.battle_start`, `battle_select`, and `battle_finish`. This path assumes Kaggle-mounted input files. It is not a standalone local installation step.
- Metadata helpers are `all_card_data()` and `all_attack()`. `to_observation_class()` converts raw dictionaries into dataclasses. Search uses `search_begin`, `search_step`, and `search_end`. The official reference documents their signatures, including explicitly predicted hidden-card inputs to `search_begin`. [CABT API functions](https://matsuoinstitute.github.io/cabt/api.html#functions)
- The game loop starts with `obs, start_data = battle_start(deck0, deck1)`, checks initialization errors, repeatedly passes selected index lists to `battle_select`, and stops when `obs["current"]["result"] >= 0`. `battle_finish()` releases native game memory. This can produce player-visible decision/action/outcome records when instrumented.
- Lines 552–557 create `MyModel(128, 2, 256, 1, 1)`, choose CUDA when available, and use `AdamW(lr=3e-4)`. Value loss is `HuberLoss(delta=0.2)`; candidate-score loss uses `HuberLoss(reduction="none", delta=0.1)`. These optimizer/loss settings can be independently adapted to a small model; copying the transformer's architecture is unnecessary.
- The training loop runs five configured iterations, with 50 random-opponent evaluation games and 100 self-play games per iteration. MCTS makes ten configured search iterations. Search-derived score targets are clipped to `[-1,1]`; the value labels combine terminal outcomes with search values using `LAMBDA=0.9`. This is search-guided value/score regression, not PPO. The candidate loss sums valid candidate errors and divides by batch size, so decisions with more candidates can contribute more loss.

The notebook contains saved outputs and execution metadata from June 16, 2026. They demonstrate that this copy records an earlier run, not that it executes in our current local environment or that its recorded win rates predict project performance.

## Official raw game API versus Kaggle episodes

The pinned public `cg` directory contains `game.py`, `sim.py`, `__init__.py`, and native libraries, including legacy `v1` libraries. It does **not** contain the notebook's `cg/api.py`. Raw dictionary matches can use the public game wrapper without that helper module; reproducing the exact notebook's metadata/search/dataclass path needs its separate `cg-lib` bundle or an independently implemented adapter. [Pinned cg inventory](https://github.com/Kaggle/kaggle-environments/tree/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cg)

`battle_start(deck0, deck1, reverse_player=False)` requires 60 card IDs per deck and returns an observation plus native `StartData`. An invalid native pointer returns a missing observation. `battle_select(list[int])` forwards selections to the engine and reports errors. The wrapper exposes no explicit random-seed parameter. Python RNG seeding alone therefore does not establish reproducible native shuffles or coin tosses. [game.py, lines 19–74](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cg/game.py#L19-L74)

The high-level route is `make("cabt", configuration=...)` followed by `env.run([agent0, agent1])`. The official example demonstrates this interface. Current configuration defaults to engine version 2 and BO3; legacy version 1 forces BO1. The Kaggle interpreter performs game-result aggregation, subsequent-game startup, and timer resets. Calling raw `battle_start` runs one game, even when the loaded native engine is version 2. [Official example](https://matsuoinstitute.github.io/cabt/#example-test-script), [configuration](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cabt.json#L7-L22), [BO3 interpreter](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cabt.py#L175-L238)

Capture the acting player's observation before selection, the ordered options, action, player index, and eventual outcome. The Kaggle serializer stores episode `steps`; CABT visualization construction aligns an observation with the next step's actions. Do not assume same-index action/observation alignment or train on privileged visualization data. [Episode serialization](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/core.py#L515-L543), [CABT replay construction](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cabt.py#L88-L115)

## Platform and lifecycle constraints

The source selects `cg.dll` on Windows, `libcg.dylib` on macOS, `libcg-arm64.so` on ARM Linux, and `libcg.so` otherwise. Those filenames exist in the pinned repository. Import loads a native library and calls `GameInitialize()`. File presence does not establish ABI compatibility, permitted redistribution, or successful execution on a particular Sol node. [sim.py, lines 23–72](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cg/sim.py#L23-L72)

`Battle` holds process-global library, pointer, and observation state. Use one active game per worker process rather than concurrent native games in threads. Ensure cleanup after every successful start. `battle_finish()` does not itself clear the Python pointer, so a wrapper must avoid duplicate frees and stale-pointer use. [Global state](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cg/sim.py#L75-L83), [cleanup](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/kaggle_environments/envs/cabt/cg/game.py#L47-L49)

## Sample limitations to fix when adapting

1. `create_node`, lines 404–416, enumerates at most 64 increasing-index combinations of exactly `maxCount` options. It misses smaller valid cardinalities and ordered alternatives, and truncates large action sets. Audit coverage instead of treating this as a complete legal-action generator. CABT explicitly provides minimum and maximum selection counts. [SelectData](https://matsuoinstitute.github.io/cabt/api.html#api.SelectData)
2. Lines 623–624 assign `+1` to the winning player and `-1` otherwise. A draw result of 2 therefore labels both players `-1`, despite the search terminal handler using zero for draws. Correct draw handling in the adapted learner.
3. Search substitutes Snorlax for hidden opponent deck/active cards and Basic Energy for hidden hand/prizes. It independently samples the player's remaining deck and prizes from the full known deck list. These are placeholder beliefs, not verified hidden-state reconstruction. Respect public observations and card-count consistency when constructing better beliefs.
4. Checkpoints save model weights before each iteration, without optimizer/RNG state. They do not provide exact interrupted-training recovery. Add resumable state in the new training driver.

## Implementation recommendations and evidence limits

Start with raw-dictionary native matches and an explicit supported-choice policy. Save completed raw game shards before training. Adapt AdamW and Huber regression only with matching value/return semantics. Keep PPO as a separate learner rather than renaming the sample loop.

For Deep Sets full/incremental agreement, keep unrounded pooled accumulators and apply `-phi(old)+phi(new)` for changed entities. Test insertion, deletion, mutation, group moves, empty pools, padding/bias masking, candidate reordering, and model/cache version changes. Requantize after updating the full accumulator. Cached entity features are reusable only when their inputs and model weights are unchanged. Global/history or candidate changes can still require rescoring every candidate. These are new project invariants, not features supplied by CABT.

The Kaggle repository carries Apache-2.0 terms. The original RL/MCTS sample is also confirmed as Apache 2.0 on its [Kaggle page](https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code). Card/native-asset terms still follow their respective sources. Record the notebook identity, official API sources, and independently adapted optimizer/loss choices. Preserve applicable notices for any copied source; preserve the sample's Apache-2.0 attribution. [Repository license](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/LICENSE)

## Recorded local runtime evidence

The coordinating agent subsequently installed PyTorch 2.15.0 in the existing `.venv`, downloaded the pinned public `cg` subset and its macOS native library into an ignored local runtime directory with a provenance manifest, and ran:

```bash
.venv/bin/python -m project.collect_matches --download --games 4 --output project/results/api-smoke/matches.jsonl
```

The run completed on macOS 26.2, ARM64, with Python 3.12.13 using raw `battle_start`, `battle_select`, and `battle_finish`. Four games completed with results `0, 1, 0, 1` and respective decision counts `147, 70, 31, 33`, totaling 281 recorded decisions. This research agent independently inspected the saved JSONL and confirmed those counts and outcomes. The records identify the pinned engine revision, engine version 2, and `single_native_game` format. [Saved game decisions](results/api-smoke/matches.jsonl)

The run also captured a real Energy-card before/after observation pair for subsequent full-versus-incremental model tests. Those fixtures are game observations, not a different interpretation of the model's scalar value output. Their existence alone does not establish that the model tests passed. [Before observation](results/api-smoke/energy-before.json), [after observation](results/api-smoke/energy-after.json)

This establishes successful native loading, complete local raw-API games, and generation of real decision records on this machine. It is an API smoke test, not evidence of trained model quality or reproducible native randomness. The exact notebook's `cg.api` metadata/search path, full Kaggle BO3 wrapper, and ASU Sol execution remain unexecuted. This research agent installed nothing and executed no simulator or training code.

### Additional game collection and optimization baseline

A subsequent native pilot completed 20 games and recorded 1,420 decisions, with seven player-0 wins and thirteen player-1 wins. Together with the first smoke run, the saved datasets contain 24 completed games and 1,701 decisions. This research agent checked the records and counts. [Pilot decisions](results/api-pilot/matches.jsonl)

The coordinating agent also compared full and incremental evaluation on a real Energy-card attachment pair. The reported values were identical for that pair; the update removed one cached hand Energy entity, encoded one changed Pokémon, and reused eight entities. Longer complete-game delta traces were still being checked when this update was written. This establishes the tested transition's cache consistency, not that all transitions or integer arithmetic are verified.

The new [train_baseline.py](train_baseline.py) independently adapts the sample's `AdamW(lr=3e-4)` and optional `HuberLoss(delta=0.2)`. It trains an 8,865-parameter Deep Sets **state-value** model on completed-game returns; it does not train an action-ranking policy or reproduce the notebook's search-score loss. The initial four-game dataset split by game into 248 training positions and 33 validation positions.

Across 12 epochs, the MSE run's validation loss changed from approximately 1.095391 to 0.961481. The Huber run's training loss changed from approximately 0.175902 to 0.169860, but its best validation checkpoint remained the untrained epoch-0 model at approximately 0.184001. The saved reports confirm these results. They establish that the optimization pipeline runs; the small random-play dataset and one validation game do not support a gameplay-strength or generalization claim. [MSE report](checkpoints/api-smoke-mse.json), [Huber report](checkpoints/api-smoke-huber.json)

The collector now explicitly records an action RNG seed without claiming control of native randomness, and clears the Python `Battle` pointer after native cleanup. These address the corresponding provenance and lifecycle limits for the implemented path. The notebook search helpers, BO3 wrapper, Sol execution, and an action-selection learner still need separate evidence.
