# Thesis: Interpretable Online Planning for F1 Race Strategy

**Author:** Andrea Pettenon
**Institution:** [to fill]
**Base paper:** Piccinotti et al. (2021), *"Online Planning for F1 Race Strategy Identification"*

---

## Research Question

> *Can the decisions of an MCTS-based F1 race strategy agent be made interpretable — and do the explanations align with what experienced strategists would expect?*

This thesis extends the Piccinotti et al. paper in one targeted direction: **interpretability**. The algorithm is taken as given; the contribution is a principled framework for understanding *why* it makes the decisions it does.

---

## 1. Problem Formulation

### 1.1 F1 Race Strategy as an MDP

An F1 race is modeled as a finite-horizon Markov Decision Process:

- **State space** (`3 + K*(R+D)` variables):
  - 3 race-level: remaining laps, SC flag, VSC flag
  - Per driver (R=5): lap time, cumulative race time, current compound, tire age, compound-change compliance flag
  - Per driver (D): remaining tire sets per compound
  - In practice, for single controlled agent: 30-dimensional observation vector

- **Action space:** Discrete — `{STAY, PIT_compound_1, PIT_compound_2, ...}`
  - Constraints: 5-lap lockout after each pit; max 2 pits (+1 during active FCY); mandatory compound variety by penultimate lap

- **Reward:** `1 - lap_time / 300` per decision step (positive, bounded in [0,1]). Dense signal, not position-based. Terminal penalty of −10,000 if race ends without pit or without using two different compounds (F1 regulation).

- **Horizon:** Planning starts at lap 8 (first 7 laps use real strategy). Maximum 70 decision steps.

- **Discount:** γ = 1 (undiscounted — objective is minimize total race time, not discounted future).

### 1.2 Why this is hard

- **Stochastic transitions:** Safety Car / Virtual SC events inject unpredictable timing shifts. A single SC can invalidate a planned two-stop strategy.
- **Long horizon with delayed consequences:** Tire choice at lap 15 constrains all subsequent decisions and only shows its cost in lap 40+.
- **Discrete + combinatorial constraints:** Not all actions are available at every step (lockout, compound availability, regulatory compliance).
- **Opponent behavior:** Competitor pit strategies affect track position independently of the controlled agent's choices.

---

## 2. The Algorithm: QL-OL-UCT

### 2.1 Core idea

Monte Carlo Tree Search (MCTS) builds a search tree by repeatedly simulating forward from the current state, using accumulated statistics to bias exploration toward promising action sequences. The paper's specific variant combines three design choices that are each motivated by the F1 problem:

**Open-Loop UCT (OL-UCT):** Standard MCTS (closed-loop) saves the simulator state at every tree node to avoid re-simulation. The F1 simulator carries so much internal state (tire physics, timing, event queues) that this immediately exhausts memory at any useful budget. OL-UCT instead treats each node as an *action sequence*, re-simulating the entire sequence from the root at every iteration. Memory cost is O(depth) not O(budget × depth).

**Q-Learning backup (QL):** Standard MCTS backup propagates the full Monte Carlo return from a leaf to the root:
```
R ← leaf.V
for each (state, action) on path from leaf to root:
    R ← state.r + γ * R
    action.Q ← (action.W + R) / (action.n + 1)    # running average
```
This is unbiased but high-variance in stochastic environments: a single unlucky SC event near the leaf can corrupt Q estimates throughout the tree. The QL backup bootstraps instead:
```
R ← leaf.V
for each (state, action) on path from leaf to root:
    td_target ← state.r + γ * R
    α_t ← α_0 * decay^(action.n)                   # decaying learning rate
    action.Q ← action.Q + α_t * (td_target - action.Q)
    R ← max_a Q(parent_state, a)                    # bootstrap, not accumulated return
```
The key difference: `R` is reset at each level to the *current best Q estimate* of the parent state, not the accumulated return from the leaf. This clips the propagation of leaf-level variance.

**ESPN durability rollout:** When a new node is first created, a rollout policy estimates its value. A uniform random rollout often produces illegal or nonsensical strategies (pitting too early, never pitting). The ESPN policy instead mirrors what a real-world analyst would do: if the current tire has exceeded its expected compound lifespan, pit with probability 0.9 (defer 1 lap with probability 0.1). Next compound: softest that covers remaining race distance, else softest available. This produces meaningful initial value estimates, guiding early exploration.

### 2.2 Algorithm structure

```
Input: current env state s, budget B, exploration constant C
Output: chosen action a

1. Initialize root node with V ← ESPN_rollout(s)
2. Repeat until budget exhausted:
   a. Deepcopy env → fresh simulation from root (open-loop)
   b. Selection: follow tree by UCT until unexpanded action
        UCT(a) = Q(a) + C * sqrt(ln(N) / n(a))
   c. Expansion: simulate action → new child node + ESPN rollout → V
   d. Backup (QL): propagate from leaf to root
        Q(a) += α_t * (r + γ * max_Q(next_state) - Q(a))
        α_t = α_0 * lr_decay^n(a)
3. Return action = argmax_a Q(root, a)
```

### 2.3 Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| γ | 1.0 | Paper |
| Budget | 10,000 per decision step | Paper |
| C (UCB constant) | 1.2 | Config file `race_strategy_full_ol.json` |
| α_0 (initial LR) | 1.0 | Not stated in paper — implementation default |
| lr_decay | 0.99 | Not stated in paper — implementation default |
| ESPN pit probability | 0.9 / defer 0.1 | Paper |
| Planning start | Lap 8 | Paper |

### 2.4 Why each choice matters

| Choice | What breaks without it |
|--------|----------------------|
| Open-Loop | Memory exhaustion: DPW-UCT ran out even at small budgets (stated in paper) |
| QL backup | High-variance MC returns corrupt Q estimates when SC events hit near leaf |
| ESPN rollout | Random rollout yields nonsensical strategies; random Q_init destabilizes early UCT selection |
| γ = 1 | Discounting would undervalue late-race pit windows, biasing toward early pits |
| skip_steps | During 5-lap lockout the agent has one action — fast-forwarding avoids wasting budget on trivial steps |

---

## 3. Implementation

### 3.1 Module map (`f1_mcts_lean/`)

```
f1_mcts_lean/
├── f1_env/race_strategy_event.py   MDP wrapper (gym.Env subclass)
├── mcts/mcts_ol_uct.py             QL-OL-UCT — THE PAPER'S ALGORITHM
├── mcts/mcts_dpw.py                DPW-UCT (baseline, MC backup)
├── race_simulation/                Heilmeier et al. (2020) simulator
├── validate_ql_ol_uct.py           Full-race validation script
└── validate_lean.py                Quick smoke test (DPW)
```

### 3.2 State vector (30 dimensions)

```
Index   Feature                         Notes
[0]     lap / race_length               Global progress
[1:7]   FCY flag one-hot (6 types)      G, Y, R, FCY, SC, VSC
[7]     lap / race_length (per-driver)  Same as [0] in single-agent
[8]     lap_time / 300                  Normalized current lap time
[9]     cumulative_time / 7200          Race time progress (2h max)
[10]    tire_age / race_length          Key feature for pit decisions
[11]    pit_count / 5                   Normalized pit stop counter
[12]    changed_compound (bool)         Regulatory compliance flag
[13]    overtake_available (bool)       DRS / track position flag
[14:22] current compound one-hot        A1–A6, I, W
[22:30] available compounds one-hot     Remaining tire set availability
```

### 3.3 Notable implementation bugs fixed (not in original paper)

1. **Terminal penalty bug:** `self._pit_counts == 0` compared list to int (always False). Fixed to `self._pit_counts[i] == 0 or len(self.used_compounds[i]) <= 1`.
2. **FCY deep-copy bug:** `get_signature()` stored `simulator_state` as a shallow reference. Restoring it during DPW state sampling corrupted the live simulator's timing state, crashing inside FCY lap fraction validation. Fixed to `deepcopy(self._race_sim.get_simulation_state())`.

---

## 4. Thesis Structure — Proposed Chapters

### Chapter 1: Introduction
- F1 strategy as a research problem: high stakes, time pressure, stochasticity
- Why interpretability matters: teams need to trust and override AI recommendations
- Thesis scope and contributions

### Chapter 2: Background
- MDP formalism
- MCTS family: UCT, DPW-UCT, OL-UCT
- Q-learning and TD methods
- Interpretability in RL: taxonomy (post-hoc vs intrinsic, local vs global)

### Chapter 3: The Planning Algorithm (QL-OL-UCT)
- Based on Section 2 above
- Implementation details and bug fixes
- Validation: reproduce paper results on 2–3 races

### Chapter 4: Interpretability Framework
- *See Section 5 below*

### Chapter 5: Experiments and Analysis
- Per-race explanation quality
- Alignment with domain knowledge (do explanations match what strategists expect?)
- Failure modes: when does the agent make surprising choices, and are the explanations informative?

### Chapter 6: Conclusions
- What the explanations reveal about the algorithm's implicit strategy model
- Limitations and future work

---

## 5. Interpretability Methods — Candidate Framework

### 5.1 My recommendation: do NOT skip validation

Before adding interpretability, run the algorithm on 2–3 paper races and verify the total race time is within ~30s of Table 1. This is necessary for one reason: **interpretability of a wrong policy is meaningless**. You cannot claim SHAP values explain good F1 strategy if the agent is actually playing poorly. One validation chapter (not exhaustive — 2 races is enough) anchors the whole contribution.

### 5.2 Interpretability methods suited to this architecture

The critical property of the architecture is that at every decision step you have:
- A 30-dim state vector `s`
- Q values for each action: `Q(STAY)`, `Q(PIT_A3)`, `Q(PIT_A4)`
- The full MCTS tree (action sequences + visit counts + Q estimates at each node)

This gives you four distinct attachment points for interpretability:

---

#### Method 1: SHAP Feature Attribution (per decision)

**What it answers:** "Which state features drove the Q difference between STAY and PIT at this lap?"

**How:** Treat MCTS as a black-box function `f(s) → Q(a)`. For each decision point, run KernelSHAP or a surrogate:
- Perturb features of `s`, re-run MCTS (or use surrogate model), measure change in `Q(PIT) - Q(STAY)`
- SHAP values assign each feature a signed contribution to the action gap

**Output:** Bar chart per decision: "tire_age contributed +0.8 to choosing PIT; laps_remaining contributed −0.3"

**Strength:** Theoretically grounded (Shapley values = unique fair attribution). Publishable.

**Challenge:** Full re-run of MCTS for each perturbation is expensive (~10k budget each). Solution: train a lightweight surrogate (linear model or small neural net) on (state, Q_gap) pairs collected during a full race run, then apply SHAP to the surrogate.

---

#### Method 2: Policy Decision Map (global)

**What it answers:** "For which combinations of tire_age and laps_remaining does the agent choose to pit?"

**How:** Fix all other state features at typical values. Sweep (tire_age, laps_remaining) on a grid. For each grid point, run MCTS and record chosen action. Plot 2D heatmap colored by action.

**Output:** A "strategy map" — readable by F1 engineers. Shows exactly the agent's pit window as a function of tire age and race position.

**Strength:** Directly connects to domain knowledge. F1 strategists think in these terms. Very visual.

**Variants:** Add a third axis (SC flag = 0/1) to show how the agent adapts to safety car situations.

---

#### Method 3: Q-Value Evolution over Race

**What it answers:** "How does the agent's confidence in each action evolve lap by lap?"

**How:** At each decision step, record `[Q(STAY), Q(PIT_A3), Q(PIT_A4)]`. Plot as time series over the race.

**Output:** Line chart where you can visually see the moment Q(PIT) crosses Q(STAY) — the agent's internal "pit window".

**Strength:** Requires zero extra computation (data is already there). Immediately intuitive. Good for Chapter 5 narrative.

---

#### Method 4: Decision Tree Policy Extraction (global surrogate)

**What it answers:** "Can we summarize the agent's full race policy as human-readable rules?"

**How:** Run the agent across many race episodes. Collect (state, chosen_action) pairs. Fit a shallow CART decision tree on this dataset.

**Output:** Rules like:
```
if tire_age > 22 AND laps_remaining < 25 AND compound == A3 → PIT
else → STAY
```

**Strength:** Most directly useful for practitioners — gives them something they can apply without running the AI. Also reveals the most important features structurally.

**Weakness:** Surrogate fidelity (how well the tree approximates the real policy) must be reported honestly. A fidelity of 85% means 15% of decisions are unexplained.

---

#### Method 5: Counterfactual Explanations (per decision)

**What it answers:** "The agent chose STAY — what would have had to be different for it to choose PIT?"

**How:** For each STAY decision, find the minimum perturbation to `s` that flips the action. Can be done by gradient/search on the surrogate model.

**Output:** "The agent would have pitted if the tire were 4 laps older" or "if there were 8 fewer laps remaining".

**Strength:** Closest to how humans think about decisions. Narratively compelling for the thesis.

---

### 5.3 Recommended combination for the thesis

| Method | Chapter | Scope | Effort |
|--------|---------|-------|--------|
| Q-value evolution | Ch 5 | Local (per race) | Minimal — already available |
| Policy decision map | Ch 4 | Global | Low — grid sweep |
| SHAP on surrogate | Ch 4 | Local (per decision) | Medium — needs surrogate training |
| Decision tree extraction | Ch 4 | Global | Medium — needs multi-race data |

Start with Q-value evolution and policy maps (low effort, high visual impact). Add SHAP once you have the surrogate. Decision tree extraction is the most novel claim and should be the centerpiece.

**Counterfactual explanations** are optional — add if time allows, they strengthen the narrative.

---

### 5.4 Alignment validation: how to know if the explanations are good

Interpretability is not just about producing numbers — it's about checking whether the explanation matches domain knowledge. For F1 strategy, ground truth is known:

| What the explanation should show | Why we know this |
|----------------------------------|-----------------|
| `tire_age` is the dominant feature for PIT decisions | Every real strategy is governed by tire degradation |
| `laps_remaining` negatively correlates with pit probability | Pitting late in the race is wasteful |
| SC flag increases Q(PIT) sharply | Pitting under SC reduces time loss — well-known F1 tactic |
| `changed_compound` (compliance flag) spikes Q(PIT) near the end | Regulatory penalty of −10,000 makes late compliance mandatory |

If SHAP values or the decision tree contradict these expectations, it signals either a problem with the algorithm or a limitation of the explanation method — both are publishable findings.

---

## 6. Open Questions and Risks

- **UCB constant C:** The exact Mango-tuned value is not in the paper. C=1.2 (from the config file) is used, but may not reproduce Table 1 exactly.
- **ESPN durability table:** The paper derives per-compound durability from ESPN data not available in the repo. Approximate values are used.
- **α_0 and lr_decay:** Not stated in the paper. Sensitivity analysis recommended.
- **Surrogate fidelity:** SHAP on a surrogate is only as good as the surrogate. Must report fidelity explicitly.
- **Single-agent assumption:** The implementation controls only one driver; opponents follow the rollout policy. This diverges from real races where opponent strategy is a key input to pit decisions.
