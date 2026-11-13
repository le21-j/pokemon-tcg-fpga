# Small position evaluators for Pokémon TCG

Architecture note updated November 12, 2026. The selected design is an NNUE-style Deep Sets position evaluator with incremental updates on the DE10-Lite, search on the PC, and direct training on ASU Sol. The [main plan](openjev-tcg-adaptation-plan.md) owns implementation decisions. Model quality, board fit, simulator branching, and complete-decision latency remain experimental. LSTMs and recurrent variants remain excluded.

## Prediction task and comparison

The model predicts the expected final match return for a fixed observer from visible state and permitted history. The PC uses position evaluations to compare simulated legal continuations. Win, loss, and draw targets are `+1`, `-1`, and `0`.

| Model | Representation | Planned role | Hardware concern |
| --- | --- | --- | --- |
| Fixed-summary value MLP | 128 deterministic state/history features, `128 -> 64 -> 1` | First trained value baseline | Handcrafted summaries may lose relevant relationships |
| Deep Sets value model | Shared entity encoder, four sums, global/history inputs, value MLP | Main architecture and incremental evaluator | Feature collisions, pooled precision, cache validity, state restoration |
| Linear value regressor | Explicit tactical state features and interactions | Minimum learned control | Limited learned interactions |
| Small boosted-tree value regressor | Bounded ensemble over the same permitted state facts | Optional software control | Different threshold/comparison RTL if later selected |

These are engineering choices for comparison. None has an established project win rate or latency. The two neural baselines should use the same source information, history summaries, match splits, training budget, search, and opponent schedule.

## Deep Sets and incremental state

Deep Sets uses shared element transformations and permutation-invariant pooling. For this project, `phi` maps each entity's 16 features to 16 learned values. Four group sums provide 64 values, concatenated with 32 global and 32 history features. The value head is `128 -> 64 -> 1`. The proposed complete evaluator has 8,865 parameters. [Deep Sets](https://arxiv.org/abs/1703.06114)

Maintain owner/zone and active/bench roles in the input. Attachments belong in the parent Pokémon's features. Repeated cards contribute repeated entities, and empty groups contribute zero. Mask padding after encoding so learned bias terms also vanish for padded rows. Candidate targets and selection context need sufficient representation at the evaluated decision boundary. Audit boards that collapse to identical encodings despite different useful continuations.

Retain each quantized entity vector and each INT32 group sum. Remove a card by subtracting its cached contribution. Replace a Pokémon by subtracting its old vector and adding its newly encoded vector. Update every entity whose facts change, plus global/history inputs. Requantize pools only for the value head. Full rebuilding provides the reference for checking incremental agreement.

Stockfish maintains cached neural accumulators and updates changed feature contributions during search. Its features, learned vector tables, and search implementation provide an architectural reference for this project adaptation. [Stockfish 19 accumulator source](https://github.com/official-stockfish/Stockfish/blob/edb0d9db6731067ec50ce619ff372b463bc4dd5d/src/nnue/nnue_accumulator.cpp)

Saved evaluator positions must include entity cache, group sums, globals/history, and position identity. Restore the simulator branch on the PC alongside the FPGA state. Model/schema changes invalidate cached vectors and saved positions. The current design chooses full snapshots first and leaves an undo journal as a measured extension.

## Other model and training families

A small linear value model checks whether additional nonlinear representation improves gameplay. A boosted-tree regressor tests a separate function family over structured state facts. XGBoost describes gradient-boosted tree learning, and published FPGA tree implementations demonstrate comparison/leaf aggregation hardware. Resource use and MAX 10 compatibility require evaluation for the actual model. [XGBoost](https://arxiv.org/abs/1603.02754), [FPGA boosted-tree inference](https://arxiv.org/abs/2002.02534)

The earlier candidate MLP is an optional auxiliary policy. It can support actor-sampled PPO training with a shared encoder/value head. PPO requires behavior action probabilities consistent with the rollout policy. The current main loop instead collects completed-match returns under frozen search agents and regresses the value network, then measures the next checkpoint with the same deployed search procedure. [PPO](https://arxiv.org/abs/1707.06347)

## Evaluation questions

Train on player-visible positions with match-grouped splits and retained wins/losses/draws. Quantization calibration uses training positions. Include the kinds of positions generated by the selected search, preserve observer perspective, and reserve final-test seeds throughout later collection.

Measure full and incremental integer agreement, restoration correctness, fitted resources, root-search coverage, complete-match outcomes, and total observation-to-selection latency. Include belief sampling, simulation, snapshots, all evaluations, and host transfers. Compare fixed search work and fixed wall time. These experiments determine whether learned entity representation and incremental updates improve the quality/cost tradeoff.
