# Project Overview

This repository builds upon the research presented in the paper **"Online Planning for F1 Race Strategy Identification"** (Piccinotti et al., 2021, Politecnico di Milano / Università di Bologna). The paper frames F1 pit-stop strategy as a sequential decision-making problem modeled as a Markov Decision Process (MDP):

- **State space:** The state has `3 + K*(R+D)` variables total (per the paper's explicit formula), where K = number of drivers, D = number of tire compounds, R = 5. The 3 race-level variables are: remaining laps, SC flag, VSC flag. Per-driver variables (R=5) are: previous lap time, cumulative race time, tire compound in use, current tire age, and a **compound-change compliance flag** (boolean: whether the driver has already satisfied the "must use at least two compounds" regulation). Additionally, each driver has D variables tracking remaining tire sets per compound.
- **Action space:** Discrete choices — stay on track, or pit and select a new tire compound (one action per compound). Constraints apply: max 2 pit stops per race (+1 extra allowed per active FCY event, only while the FCY remains active); 5-lap lockout after each pit stop; mandatory compound change enforced at the penultimate lap if compliance has not been met.
- **Reward:** Negative lap time (the objective is to minimize cumulative race time). Reward is dense (one signal per lap), not position-based. The paper notes this can cause the agent to sacrifice positions while still minimizing time — an acknowledged limitation.

The core algorithm introduced is **QL-OL-UCT**: Open-loop UCT where the **backup propagation** uses Q-learning TD updates instead of raw Monte Carlo returns. Critically, the rollout still runs to produce an initial leaf value estimate; the QL update is applied in the backup phase at each node using the max operator over visited children (unexplored actions are excluded from the max). The open-loop variant is explicitly used to circumvent the memory overhead of saving the simulator state at each tree node.

**Thesis Goal:** This thesis extends the core research by introducing interpretability methods for the learned policies. The goal is to demystify the "black box" decisions evaluated by the MCTS, validating them through attention mechanisms on state features, SHAP values, and policy visualizations.

# Repository Structure

- `agents/`: Wrappers connecting agents to the logic (e.g., `mcts_agent.py`, `dng_agent.py`).
- `envs/`: Defines the MDP gym-like environments tracking state and coordinating actions. Contains F1 logic via `race_strategy.py`, `race_strategy_event.py`, and `race_strategy_full.py`.
- `race_simulation/`: The backend F1 simulator based on Heilmeier et al. (2020), comprising modules `racesim/` and `racesim_basic/`, simulating race dynamics lap-by-lap.
- `interpretable_mcts/`: The refactored and modularized codebase for tree-search implementation designed specifically to hook interpretability methods into the nodes (such as DPW or Q-learning).
- `pure_mcts/`: The legacy set of MCTS configurations. 
- `run_*` shell scripts: Execution blueprints to quickly launch specific experiments (e.g. `run_full_race.sh`).
- Main entry points: `alphazero.py` (which oversees initialization, looping limits, and logging runs) and `agent.py` (which wires up the actual search vs environment routines).

How to run experiments: 
Use shell scripts mapped in the root directory (e.g. `./run_full_race.sh`) which invokes `alphazero.py` and subsequently routes into `agent.py` setting off iterative rollouts against the `envs/` MDPs.

# Current State

- **What is working:** The F1 Simulator modifications tracking state determinism/stochastic event generation, the rollout models, basic baselines, and Q-learning integrated backups in the tree search.
- **Recent cleanup:** The codebase underwent major cleanup. The tree-search scripts were aggressively decluttered and abstracted into a new unified `interpretable_mcts/` directory. Equivalence was aggressively asserted against older versions using `test_mcts_equivalence.py`, meaning all experiments are technically ready to migrate purely to the new architecture. 
- **Technical Debt:** Legacy components like `pure_mcts` need to be formally deprecated and removed. Parts of `alphazero.py` have old code block comments reflecting unpolished data-saving loops, and there are potentially inactive dependencies. 

# Architectural Decisions

- **Open-Loop over Progressive Widening:** F1 simulators natively manage huge amounts of internal variable state. Saving/restoring the exact physics/timing state at each tree node to perform pure Progressive Widening (DPW-UCT) requires exorbitant memory — the paper explicitly states they "quickly ran out of memory during the tree search, even for small search budgets" when attempting DPW-UCT. Open-Loop UCT instead treats each node as an action sequence (not a state), re-simulating from root at each iteration.
- **Simulator Integration:** Heilmeier's base simulator was augmented significantly. It is structured to halt, return control back to the MDP wrapper lap-by-lap, and most importantly inject stochastic SC/VSC events at runtime. This guarantees the planner has to deal with variance and does not simply "predict the future". FCY events injected at lap `l` only become effective at lap `l+1`; new events must be at least 1 lap apart from the previous; randomly-generated events merge with a running event by extending its duration.
- **Q-Learning Backups:** Because standard Monte Carlo returns accumulate all stochasticity down an entire rollout (SC events, slow pits), the variance back-propagated to the tree root destabilizes branch evaluations. The QL TD backup at each node is: `Q_t(N, a) = (1 - α_t) * Q_t(N, a) + α_t * (r + γ * max_a' Q_t(child, a'))`. The learning rate `α_t` is **exponentially decaying**. The discount factor is **γ = 1** (undiscounted, to focus on cumulative race time). When not all children have been visited, the max operator is applied only over visited children.
- **Opponent Modeling:** During tree search, all drivers (including opponents) are controlled by the **rollout policy** — not their real race strategies. Real race strategies are used only as the "True" baseline at evaluation time. This prevents the planner from having privileged knowledge of opponent behavior.

# Algorithm Configuration (from Paper)

These details are explicitly specified in the paper but were absent from prior documentation. They are critical for reproducing experiments and for reasoning about the interpretability extensions.

- **Planning start:** Lap 8 (not lap 1). The first 7 laps use the real-race strategy.
- **Discount factor:** γ = 1 (undiscounted returns).
- **Computational budget:** 10,000 simulator samples per lap decision.
- **Learning rate:** Exponentially decaying `α_t`.
- **UCB exploration constant (C / Cp): NOT REPORTED IN THE PAPER.** The paper describes UCB1 selection as `argmax_a [Q(N,a) + C * sqrt(2 ln N.n / C(N,a).n)]` and notes C regulates exploration-exploitation, but the actual tuned value is never stated. All hyperparameters were tuned via Bayesian optimization (Mango framework) on the 2017 Australian GP race — the value of C is therefore experiment-dependent and must be recovered from the code or config files.
- **Hyperparameter tuning:** Mango (Bayesian optimization), tuned once on 2017 Australian GP and applied to all 10 races without further adjustment.
- **Rollout policy:** ESPN durability-based policy. A "compound durability" is pre-computed from ESPN predicted strategies. At each lap, if the current tire has reached its expected durability: pit with probability 0.9, defer by 1 lap with probability 0.1. Next compound is chosen deterministically: softest compound whose durability covers remaining race distance; if none qualifies, softest available. The simpler stochastic rollout (90% stay, 10% random pit) was tried first and discarded as it produced unreasonable strategies.
- **Tire availability assumptions:** For races with 2 available compounds: 3 sets soft, 2 sets hard. For races with 3+ compounds: 2 soft, 2 medium, 1 hard.

# Thesis Backlog

- [ ] Define and implement interpretability methods (candidates: attention maps on state features, SHAP values on action selection, policy visualization across race laps)
- [ ] Design evaluation protocol for interpretability
- [ ] Write thesis chapters

# Thesis Writing Log

**[2026-04-14] — Refactored MCTS module — Cleaned up old tree search legacy scripts into `interpretable_mcts/` and formally asserted equivalence with `test_mcts_equivalence.py` — Establishes cleaner codebase allowing easier hooks/node-level logging for interpretability methods (SHAP / Attachments)**
