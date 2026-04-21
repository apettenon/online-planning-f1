"""
QL-OL-UCT: Open-Loop UCT with Q-learning TD backup.

Based on: Piccinotti et al. (2021),
          "Online Planning for F1 Race Strategy Identification"

Key differences vs mcts_dpw.py:
  - Open-Loop: each action has ONE child_state (not a list with DPW).
    The tree is re-simulated from root on every iteration.
    No simulator state is saved at nodes — this is the whole point of OL.
  - Q-Learning backup: instead of MC average over full returns, TD update
    with max bootstrap:
        Q(a) += α_t * (r + γ * max_a' Q(s', a') − Q(a))
    Learning rate decays exponentially: α_t = α_0 * decay^(n_visits(a))
  - ESPN rollout: pit near the compound's expected durability (p=0.9),
    defer 1 lap (p=0.1). Next compound: softest that covers remaining
    distance, else softest available.

Paper hyper-parameters (γ=1, budget=10,000 per decision step, no DPW).
α_0 and lr_decay are not explicitly stated — sensible defaults provided.
"""

import copy
import json

import numpy as np

from utils.helpers import argmax, max_Q, stable_normalizer

# ------------------------------------------------------------------
# ESPN rollout constants
# ------------------------------------------------------------------

# Expected stint length (laps) per compound — used by rollout only.
# Softer compounds have shorter expected stints.
# Approximate Pirelli-era dry values; wet/inter figures are nominal.
COMPOUND_DURABILITY = {
    "A1": 15, "A2": 20, "A3": 25, "A4": 30, "A5": 35, "A6": 40,
    "I": 30, "W": 50,
}

# Softness order (index 0 = softest).  Used to choose next compound.
COMPOUND_SOFTNESS = ["A1", "A2", "A3", "A4", "A5", "A6", "I", "W"]

# Compound list matching the one-hot encoding in race_strategy_event.py
COMPOUNDS = ["A1", "A2", "A3", "A4", "A5", "A6", "I", "W"]

# State vector layout (30 features, single-agent, from RaceEnv.get_state())
# [0]      : global lap / race_length
# [1:7]    : FCY flag one-hot  (6 values)
# [7]      : per-driver lap / race_length  (= [0] for single agent)
# [8]      : lap_time / 300
# [9]      : cumulative_time / 7200
# [10]     : tire_age / race_length
# [11]     : pit_count / 5
# [12]     : changed_compound (bool)
# [13]     : overtake_available (bool)
# [14:22]  : current compound one-hot  (8 values)
# [22:30]  : available compounds one-hot  (8 values)
_IDX_GLOBAL_LAP = 0
_IDX_TIRE_AGE   = 10
_IDX_COMPOUND   = slice(14, 22)


# ==================================================================
# Tree nodes
# ==================================================================

class OLAction:
    """
    Action node in QL-OL-UCT.

    Open-Loop variant: at most ONE child state per action (no list,
    no progressive widening).  Q value is maintained via QL updates.
    """

    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.n = 0
        self.Q = Q_init
        self.child_state = None  # single child — open-loop property

    def ql_update(self, td_target, alpha_0, lr_decay):
        """
        Q-learning update with exponentially decaying learning rate.
            α_t = α_0 * lr_decay^n
            Q  += α_t * (td_target − Q)
        """
        alpha_t = alpha_0 * (lr_decay ** self.n)
        self.Q += alpha_t * (td_target - self.Q)
        self.n += 1


class OLState:
    """
    State node in QL-OL-UCT.

    On first creation a rollout (ESPN policy) is run to initialise V.
    Child actions share that V as their initial Q estimate.
    """

    def __init__(self, index, r, terminal, parent_action, na,
                 budget, env=None, max_depth=200):
        self.index = index
        self.r = r
        self.terminal = terminal
        self.parent_action = parent_action
        self.na = na
        self.n = 0
        self.remaining_budget = budget

        if terminal or env is None:
            self.V = 0.0
            self.remaining_budget -= 1
        else:
            self.V, self.remaining_budget = _espn_rollout(
                copy.deepcopy(env), budget, max_depth
            )

        # Only create child actions for currently available actions.
        # Unavailable actions (lockout, depleted compound) are masked with
        # Q = -inf so UCT never selects them.  This mirrors race_ol_uct.py
        # in the big repo and prevents corrupted Q signals from STAY rewards
        # being back-propagated onto illegal-action Q values.
        if env is not None and hasattr(env, 'get_available_actions') and hasattr(env, 'get_next_agent'):
            available = set(env.get_available_actions(env.get_next_agent()))
        else:
            available = set(range(na))

        self.child_actions = [
            OLAction(a, parent_state=self,
                     Q_init=self.V if a in available else -np.inf)
            for a in range(na)
        ]

    def select(self, c=1.5):
        """UCT action selection — masked actions (Q=-inf) are never selected."""
        uct = np.array([
            -np.inf if ca.Q == -np.inf            # masked: lockout / depleted
            else (ca.Q + c * np.sqrt(np.log(self.n) / ca.n) if ca.n > 0
                  else np.inf)
            for ca in self.child_actions
        ])
        return self.child_actions[argmax(uct)]

    def update(self):
        self.n += 1

    def to_json(self):
        return json.dumps({
            "V": str(self.V),
            "n": self.n,
            "r": float(self.r),
            "terminal": self.terminal,
        })


# ==================================================================
# ESPN rollout helpers  (module-level, not methods, for speed)
# ==================================================================

def _espn_rollout(env, budget, max_depth):
    """
    Run an ESPN durability-based rollout from the current env state.

    At each decision step:
      - If the tire age has reached the compound's expected durability,
        pit with p=0.9 (choosing the best compound), else stay.
      - Calls env.step() which handles the skip_steps fast-forward.
    Returns (cumulative_reward, remaining_budget).
    """
    done = False
    ret = 0.0
    depth = 0
    while depth < max_depth and budget > 0 and not done:
        a = _espn_action(env)
        _, r, done, _ = env.step(a)
        ret += r
        depth += 1
        budget -= 1
    return ret, budget


def _espn_action(env):
    """Choose an action according to the ESPN durability policy."""
    agent = env.get_next_agent() if hasattr(env, "get_next_agent") else 0
    available = (
        env.get_available_actions(agent)
        if hasattr(env, "get_available_actions")
        else list(range(env.action_space.n))
    )

    if len(available) == 1:
        return available[0]  # only STAY available

    # --- read state -------------------------------------------------------
    state_vec = env.get_state()
    if state_vec is None or len(state_vec) < 22:
        # fallback: random among available
        return int(np.random.choice(available))

    race_len = getattr(env, "race_length", 70) or 70
    tire_age_laps = float(state_vec[_IDX_TIRE_AGE]) * race_len
    remaining_laps = (1.0 - float(state_vec[_IDX_GLOBAL_LAP])) * race_len

    compound_oh = state_vec[_IDX_COMPOUND]
    ci = int(np.argmax(compound_oh))
    current_compound = COMPOUNDS[ci] if ci < len(COMPOUNDS) else "A3"
    durability = COMPOUND_DURABILITY.get(current_compound, 25)

    # --- pit decision -----------------------------------------------------
    if tire_age_laps >= durability:
        if np.random.random() < 0.9:  # pit with p=0.9
            return _choose_compound(env, agent, available, remaining_laps)
        # defer with p=0.1 → fall through to STAY
    return 0  # STAY


def _choose_compound(env, agent, available, remaining_laps):
    """
    Next compound selection: softest whose durability >= remaining_laps.
    If none qualifies, pick the softest available.
    """
    pit_actions = [a for a in available if a > 0]
    if not pit_actions:
        return 0

    options = []
    for a in pit_actions:
        try:
            comp = env.map_action_to_compound(a)
        except Exception:
            comp = "A3"
        dur = COMPOUND_DURABILITY.get(comp, 25)
        soft = COMPOUND_SOFTNESS.index(comp) if comp in COMPOUND_SOFTNESS else 3
        options.append((a, comp, dur, soft))

    # Softest compound that still covers remaining distance
    covering = [(a, c, d, s) for a, c, d, s in options if d >= remaining_laps]
    if covering:
        covering.sort(key=lambda x: x[3])   # ascending softness = softest first
        return covering[0][0]

    # No compound covers the distance: fall back to softest available
    options.sort(key=lambda x: x[3])
    return options[0][0]


# ==================================================================
# QL-OL-UCT planner
# ==================================================================

class QL_OL_UCT:
    """
    Open-Loop UCT with Q-learning TD backup.

    Arguments
    ---------
    root        : OLState or None — existing root to continue from.
    root_index  : np.ndarray     — current environment observation.
    na          : int            — number of discrete actions.
    gamma       : float          — discount factor (paper uses γ=1).
    alpha_0     : float          — initial Q-learning rate.
    lr_decay    : float          — per-visit multiplicative decay of α.
                  α_t = alpha_0 * lr_decay^(n_visits).
    depth_based_bias : bool      — whether to scale UCB constant by depth.
    """

    def __init__(self, root, root_index, na,
                 gamma=1.0, alpha_0=1.0, lr_decay=0.99,
                 model=None, depth_based_bias=False):
        self.root = root
        self.root_index = root_index
        self.na = na
        self.gamma = gamma
        self.alpha_0 = alpha_0
        self.lr_decay = lr_decay
        self.depth_based_bias = depth_based_bias

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200):
        """
        Perform QL-OL-UCT search from the root.

        Each iteration:
          1. Deepcopy Env → re-simulate from root (open-loop).
          2. Follow tree: if action has child → re-simulate that step.
             if no child → expand (run rollout), break.
          3. QL backup from leaf to root.
        """
        if self.root is None:
            env_copy = copy.deepcopy(Env)
            self.root = OLState(
                self.root_index, r=0.0, terminal=False,
                parent_action=None, na=self.na,
                budget=budget, env=env_copy, max_depth=max_depth,
            )
            budget = self.root.remaining_budget
        else:
            self.root.parent_action = None

        if self.root.terminal:
            raise ValueError("Can't search from a terminal state")

        while budget > 0:
            state = self.root
            mcts_env = copy.deepcopy(Env)   # open-loop: re-simulate each iter
            mcts_env.seed()
            st = 0

            # --- Selection / Expansion ------------------------------------
            while not state.terminal:
                bias = (
                    c * self.gamma ** st / (1 - self.gamma)
                    if self.depth_based_bias else c
                )
                action = state.select(c=bias)
                st += 1

                if action.child_state is not None:
                    # Already expanded: re-simulate the step and follow
                    s1, r, t, _ = mcts_env.step(action.index)
                    state = action.child_state
                    state.r = r          # update with fresh stochastic reward
                    state.terminal = t
                    if t:
                        budget -= 1
                        break
                else:
                    # Expand: simulate step, create new child with rollout
                    s1, r, t, _ = mcts_env.step(action.index)
                    budget -= 1
                    child = OLState(
                        s1, r, t, action, self.na,
                        budget=budget, env=mcts_env,
                        max_depth=max_depth - st,
                    )
                    budget = child.remaining_budget
                    action.child_state = child
                    state = child
                    break

            # --- QL Backup ------------------------------------------------
            # R starts as the leaf's rollout estimate V.
            # At each level: td_target = r + γ * R
            #   action.Q += α_t * (td_target − Q)
            # Then R is reset to max_Q of the parent state (bootstrap).
            R = state.V
            state.update()

            while state.parent_action is not None:
                if not state.terminal:
                    td_target = state.r + self.gamma * R
                else:
                    td_target = state.r

                action = state.parent_action
                action.ql_update(td_target, self.alpha_0, self.lr_decay)

                state = action.parent_state
                state.update()

                # Bootstrap: value of parent state = max Q among its children
                R = max(ca.Q for ca in state.child_actions)

    def return_results(self, temp, on_visits=False):
        """
        Compute action distribution and estimated value at root.

        on_visits=False → argmax policy (as in paper).
        on_visits=True  → temperature-softened visit counts.
        """
        counts = np.array([ca.n for ca in self.root.child_actions])
        Q = np.array([ca.Q for ca in self.root.child_actions])
        if on_visits:
            pi = stable_normalizer(counts, temp)
        else:
            pi = max_Q(Q)
        total = np.sum(counts)
        # Only include visited actions in V — 0 * (-inf) = nan for masked actions
        visited = counts > 0
        V = (np.sum((counts[visited] / total) * Q[visited])[None]
             if total > 0 and visited.any()
             else np.array([0.0]))
        return self.root.index, pi, V

    def forward(self, a, s1, r):
        """
        Advance root after the real environment step.

        OL-UCT does NOT reuse the subtree: the open-loop assumption means
        the sub-tree built from a single root state may not match the new
        root after a stochastic transition.  Reset root to None each step.
        """
        self.root = None
        self.root_index = s1
