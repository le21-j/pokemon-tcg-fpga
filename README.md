# Pokémon TCG AI with FPGA inference

Pokémon TCG agents, a shared testing arena, and a compact Deep Sets position evaluator for a planned DE10-Lite implementation. The Python prototypes run native CABT games; FPGA deployment and ASU Sol training remain project milestones.

## Run in Google Colab

Open a notebook and select **Runtime → Run all**. Setup clones this public repository, installs compatible Python dependencies, and downloads the native game runtime/API helpers when needed.

| Notebook | Purpose | Colab |
| --- | --- | --- |
| [Team arena](project/arena.ipynb) | Compare Python agents or checkpoints in first-to-three series | [Open](https://colab.research.google.com/github/le21-j/pokemon-tcg-fpga/blob/main/project/arena.ipynb) |
| [Sample agent](project/sample_agent.ipynb) | Test and package the teammate inference template | [Open](https://colab.research.google.com/github/le21-j/pokemon-tcg-fpga/blob/main/project/sample_agent.ipynb) |
| [Reinforcement learning](project/reinforcement_learning.ipynb) | The supplied Transformer/MCTS sample, with short section notes | [Open](https://colab.research.google.com/github/le21-j/pokemon-tcg-fpga/blob/main/project/reinforcement_learning.ipynb) |
| [Deep Sets walkthrough](project/deep_sets_walkthrough.ipynb) | Inspect the model and full/incremental Energy attachment updates | [Open](https://colab.research.google.com/github/le21-j/pokemon-tcg-fpga/blob/main/project/deep_sets_walkthrough.ipynb) |

Agents can be a Python file or folder exposing `agent(observation) -> list[int]`. They load their own architecture and weights and return legal option indices. Start with the [agent template](project/agent_template/README.md). Custom model dependencies and weight assets are supplied with each agent.

## Run locally

Use Python 3.12 or another compatible Python version, create an environment, then install and check the project:

```bash
python -m pip install -r requirements.txt
python -m project.energy_attachment
python -m project.check_arena
python -m project.check_sample_notebook
python -m project.check_colab_setup
python -m project.rl_baseline.check
python project/agent_template/check_agent.py
```

Select that environment as the Jupyter kernel. Local notebook runs use the existing checkout. Native game libraries download automatically on first use. For independent native workers, keep one active game per process.

## Project details

Read the [project guide](project/README.md), [architecture and hardware plan](project/architecture-and-training-plan.md), and [NNUE-style Deep Sets plan](project/openjev-tcg-adaptation-plan.md). Generated matches, checkpoints, and runtime downloads are ignored. Published notebooks have cleared outputs. The notebook defaults verify the workflow using short runs; they do not establish competitive playing strength.

## Attribution and license

The RL baseline is based on [Kiyota's Reinforcement Learning and MCTS sample code](https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code), published under Apache 2.0. The original 22 function/class definitions remain intact in the RL notebook; local integration changes are documented there. See [NOTICE](NOTICE), [third-party notices](THIRD_PARTY_NOTICES.md), and [LICENSE](LICENSE).

CABT runtime assets and example decks download from their upstream sources. Pokémon-related names and game assets belong to their respective rights holders.
