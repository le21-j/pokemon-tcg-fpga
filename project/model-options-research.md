# Model options for Pokémon TCG and DE10-Lite inference

Research note, November 8, 2026, with project direction updated below. The ranking is an engineering judgment for a student software/hardware project. The November 12 Python prototype now runs full/incremental attachment checks, native-game data collection, and an AdamW value-training baseline. Gameplay-strength comparison, quantization, and board fit remain experiments. See the [implementation evidence](kaggle-api-implementation-research.md).

Direction update, November 12: the [current plan](openjev-tcg-adaptation-plan.md) trains a Deep Sets position-value model from completed-match returns and search-guided self-play on Sol. The DE10-Lite maintains incremental evaluations and saved positions while the PC searches. PPO is an optional auxiliary-policy experiment. The [small-model research note](small-model-architecture-research.md) covers current evaluator alternatives. LSTMs and recurrent variants remain outside the project scope. Candidate-scoring methods below record earlier alternatives and possible auxiliary policies.

## Recommendation and terminology

Start with a fixed-summary position-value MLP and the compact Deep Sets value model. Fit observed positions to terminal match returns, then collect further training matches using frozen evaluator-guided search. Keep linear value regression and small boosted-tree value regression as inexpensive controls. The [current model plan](openjev-tcg-adaptation-plan.md) defines the selected learning loop and [hardware design](architecture-and-training-plan.md).

Separate two decisions. Architecture determines the computation deployed on the FPGA: an MLP, pooled entity encoder, tree ensemble, recurrent network, or transformer. A learning procedure determines how parameters are obtained: behavior cloning, DAgger, PPO, Q-learning, or Monte Carlo return regression. Different learning procedures can produce the same RTL-compatible MLP. Conversely, choosing reinforcement learning does not require copying the supplied transformer's architecture.

The selected evaluator computes `V(player_observation, permitted_history)` for a fixed observer. PC search compares legal root payloads using terminal results and leaf values. An auxiliary candidate policy computes `score(observation, history, candidate)` with shared weights across candidates. Candidate indices change meaning between observations. Host handlers preserve cardinality and ordered-selection semantics and report search coverage. The [competition research](competition-research.md) documents those interface constraints.

## What the supplied notebook actually contains

Verified by reading [reinforcement-learning-and-mcts-sample-code.ipynb](https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code), without executing it:

- The introduction calls it reinforcement learning and MCTS with Transformer-like models. `MyModel` uses `EmbeddingBag`, a transformer encoder, an attention decoder, and separate value and candidate-score outputs. The configured model is `MyModel(128, 2, 256, 1, 1)` with 24 encoder words.
- `SEARCH_COUNT=10`. Search uses CABT's search API and chooses the most visited root child. Self-play produces training samples. The optimizer is AdamW; the losses are Huber regression for the value and candidate-score targets. This implementation is not PPO or DQN.
- Its outer loop runs five iterations, each configured for 50 evaluation games against a random opponent and 100 self-play games. Those constants are not evidence that the notebook ran successfully or achieved any reported performance.
- `create_node` enumerates at most 64 increasing-index combinations containing exactly `maxCount` choices. Training pads these candidate combinations to 64. This deserves a selection-coverage audit because the interface also permits smaller cardinalities and ordered selections. Copying that enumeration would impose a policy limitation independently of model quality.

The notebook already supplies a learning/search example, but no course originality rule was provided. A distinct contribution could be compact-model design, audited choice coverage, learning-method comparisons, integer export, RTL inference, and measured quality/resource/latency tradeoffs. That is a proposal to discuss against the rubric, not a claim about grading eligibility.

There is also firsthand evidence from this competition. The author of the 15th-place write-up reports a 7.5-million-parameter recurrent actor-critic trained through population self-play, using direct action scoring at deployment. The author reports approximately 5.5 billion policy decisions during training. These are participant-reported results for that system. They support investigating direct learned policies in this environment, while leaving the quality of a much smaller FPGA model to experiment. [Participant's solution write-up](https://www.kaggle.com/c/pokemon-tcg-ai-battle/writeups/15th-place-solution)

## Learning procedures using a compact scorer

### 1. Behavior cloning, then DAgger if useful

Behavior cloning fits demonstrated selections using supervised learning. For one-choice decisions, normalize candidate scores over the available choices and train cross-entropy against the teacher choice. This gives a direct debugging signal and reuses the planned dense arithmetic. Its limitation is teacher quality and coverage: imitation does not turn random trajectories into good strategy.

DAgger addresses the distribution mismatch between expert demonstrations and states visited by the learned policy. It collects learner-induced states, queries the expert, and aggregates the resulting examples. The original paper studies this sequential-prediction problem and gives guarantees under its assumptions. [Ross, Gordon, and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html)

Project judgment for an auxiliary policy: begin with cloning, then add one measured DAgger round if a teacher can label new states affordably. Keep held-out test matches outside aggregation and record label cost and teacher identity. Measure the resulting policy in complete matches.

### 2. PPO with an MLP actor and critic

PPO alternates environment sampling with optimization of a policy objective designed to limit overly large updates. It can reuse a compact candidate-scoring actor and a value critic. The selected project can export the encoder/value subset for PC search after training. The original experiments concern other benchmark tasks. [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)

Legal-action masking excludes invalid actions from the policy distribution. Research provides theoretical justification for policy-gradient masking and evaluates its effect as invalid-action spaces grow. In this project, candidate lists already encode legal options; padding still needs masking, and compound selection rules remain the host's responsibility. [Huang and Ontañón, 2022](https://arxiv.org/abs/2006.14171)

Project judgment: PPO is an optional direct-RL comparison with a training policy head. Store the exact candidate set, masks, behavior log-probabilities, and action when collecting actor-sampled rollouts. Preserve them when computing update ratios. Export the encoder/value subset for the selected FPGA search evaluator. A search-selected action needs a training design that accounts for its actual behavior policy. Fixed validation opponents help detect narrow adaptation.

### 3. Candidate-conditioned DQN

Deep Q-learning fits action values using reward and a bootstrapped estimate of future value. The original DQN work combines neural value approximation with experience replay. It demonstrates Atari results, not superiority for hidden-information card games. [Mnih et al., 2013](https://arxiv.org/abs/1312.5602)

Project judgment: replace a fixed action-output vector with `Q(observation, candidate)` and maximize only over the next legal candidates. Store the next candidate set with each transition. The RTL remains a compact scorer, but the learner gains replay management and bootstrapped-target failure modes. Define transitions across opponent decisions carefully; blindly negating or discounting every simulator subdecision can change the intended return. Rank this below PPO and Monte Carlo for the first implementation because it adds training complexity without a demonstrated project-specific benefit.

### 4. Deep Monte Carlo return regression

DouZero combines action encoding, neural networks, parallel actors, and Monte Carlo learning in DouDizhu, a different imperfect-information card game with variable legal actions. Its algorithm regresses action values toward observed episode returns rather than a next-state Bellman bootstrap. Its published network and training budget are not a DE10-Lite design. [Zha et al., 2021, algorithm and experiments](https://proceedings.mlr.press/v139/zha21a/zha21a.pdf)

Project judgment: the selected project variant regresses a state-value model from each observer-visible position to that observer's completed-match return. DouZero's action-value model is background for the use of sampled complete returns. Define draws and BO3 outcomes explicitly, retain losses, and collect fresh positions with the actual search policy. Sparse returns and policy/opponent changes require validation. The cited DouDizhu results supply no Pokémon win-rate forecast.

### 5. Search teacher and compact student

Expert Iteration separates planning from generalization: tree search produces improved policies, and a network learns from them before guiding subsequent search. The paper evaluates Hex. [Anthony, Tian, and Barber, 2017](https://arxiv.org/abs/1705.08439)

An optional distillation experiment can use a verified search agent on Sol to label rankings or policy targets. Compare a compact policy with the search source and heuristic baseline. Audit the acting-player information boundary in the search source. Report the runtime system actually deployed and include online search in its complete-decision latency. The selected main plan deploys PC search with the small value evaluator.

## Architectural alternatives

Deep Sets provides a permutation-invariant set-processing design based on shared element transformations and pooling. The selected project uses a small entity MLP, grouped sums, and a position-value MLP. Preserve owner/zone, counts, active status, and attachment facts. Maintain exact cached vectors and wide pool sums to support incremental updates. The current hardware plan gives the value-model resource and numeric budget. [Zaheer et al., 2017](https://arxiv.org/abs/1703.06114)

For richer alternatives, the proposal judgments are conditional. A GNN could explicitly represent Pokémon, attached cards, and targeting relationships, at the cost of graph construction and message buffers. A transformer could model interactions between encoded entities, with attention buffers and additional arithmetic to implement. The supplied notebook already demonstrates that direction in software. These are background options; the current scope uses dense layers and grouped entity sums, with deterministic visible-history summaries.

A small boosted-tree scorer is a real hardware comparator. XGBoost is a tree-boosting learning system. [Chen and Guestrin, 2016](https://arxiv.org/abs/1603.02754) Published FPGA implementations demonstrate boosted trees using on-chip comparison and selection logic. [Summers et al., 2020](https://arxiv.org/abs/2002.02534) Project judgment: bound tree count and depth, compare on the same state-candidate features, and estimate threshold storage and comparator resources. The cited implementation does not establish MAX 10 tool compatibility or our latency. Trees deserve a measured comparison, not blanket rejection because they branch.

## Scope and experiment ranking

NFSP combines reinforcement learning with supervised learning of an average strategy, demonstrated in poker. [Heinrich and Silver, 2016](https://arxiv.org/abs/1603.01121) CFR minimizes counterfactual regret in extensive games and relates that process to equilibrium computation under stated assumptions. [Zinkevich et al., 2007](https://martin.zinkevich.org/publications/regretpoker.pdf) These are relevant imperfect-information methods, but introduce information-set modeling and training scope beyond the first accelerator demonstration. Neither is obligatory merely because the game hides cards.

Recommended order, based on implementation risk rather than proven strength:

1. Fixed-summary and Deep Sets position-value models trained on completed-match returns.
2. PC search-guided self-play and iterative value regression with frozen collection checkpoints.
3. Incremental FPGA evaluation and saved-position restoration under the same search.
4. Linear/boosted-tree value controls and optional PPO with a training policy head.
5. Search-target distillation, candidate DQN, and equilibrium methods after evidence warrants their scope.

Use Sol for independent rollouts, teacher generation, and training comparisons, subject to verified allocations. Freeze decks, opponents, seeds, supported contexts, and training budget. Compare complete-match results with uncertainty, quantization-induced action changes, integer-reference agreement, fitted resources, accelerator cycles, and end-to-end decision time. Only those experiments can decide which model is best for this project.
