"""
Validation script for QL-OL-UCT (the paper's algorithm).

Runs a full race episode with QL-OL-UCT at each decision step and prints:
  - per-lap: action chosen, raw reward, cumulative reward, elapsed time
  - final: total episode reward, laps, pit stops

Paper hyper-parameters:
  budget = 10,000 per decision step
  γ      = 1.0  (undiscounted)
  C_UCB  = tuned via Mango on 2017 Australian GP (value not published)
  ESPN rollout policy (durability-based)

Run from the f1_mcts_lean/ directory:
    ../venv/bin/python3 validate_ql_ol_uct.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from f1_env.race_strategy_event import RaceEnv
from mcts.mcts_ol_uct import QL_OL_UCT

# --- Hyper-parameters ---------------------------------------------------
BUDGET    = 10000    # simulator calls per decision step (paper spec)
GAMMA     = 1.0      # undiscounted (paper)
C_UCB     = 1.2      # exploration constant (paper doesn't publish tuned value)
MAX_DEPTH = 70       # tree depth cap (= race horizon)
ALPHA_0   = 1.0      # initial Q-learning rate
LR_DECAY  = 0.99     # per-visit decay: α_t = α_0 * 0.99^n
HORIZON   = 70       # max decision steps

ACTION_NAMES = {0: "STAY", 1: "PIT_A3", 2: "PIT_A4"}


def run(race_pars_file="pars_Suzuka_2016.ini"):
    print("=" * 60)
    print(f"QL-OL-UCT VALIDATION — {race_pars_file}")
    print(f"  budget={BUDGET}, γ={GAMMA}, C={C_UCB}")
    print(f"  α_0={ALPHA_0}, lr_decay={LR_DECAY}")
    print("=" * 60)

    np.random.seed(42)
    env = RaceEnv(horizon=HORIZON, skip_steps=True, race_pars_file=race_pars_file)
    env.seed(42)
    s = env.reset()

    mcts = QL_OL_UCT(
        root=None, root_index=s, na=env.action_space.n,
        gamma=GAMMA, alpha_0=ALPHA_0, lr_decay=LR_DECAY,
    )

    total_reward = 0.0
    lap = 0
    pit_count = 0
    done = False

    print(f"\n{'Step':>4}  {'Action':>8}  {'Reward':>8}  {'CumRew':>9}  {'Time':>7}")
    print("-" * 50)

    while not done:
        lap += 1
        t0 = time.time()

        mcts.search(n_mcts=BUDGET, c=C_UCB, Env=env, mcts_env=env,
                    budget=BUDGET, max_depth=MAX_DEPTH)
        elapsed = time.time() - t0

        _, pi, v = mcts.return_results(temp=0)
        a = int(np.argmax(pi))

        s1, r, done, _ = env.step(a)
        total_reward += r

        action_str = ACTION_NAMES.get(a, f"PIT_{a}")
        if a > 0:
            pit_count += 1
        print(f"{lap:>4}  {action_str:>8}  {r:>8.4f}  {total_reward:>9.4f}  [{elapsed:.1f}s]")

        mcts.forward(a, s1, r)
        s = s1

        if lap >= HORIZON:
            break

    print("-" * 50)
    print(f"Episode done — steps: {lap}, pits: {pit_count}, "
          f"total reward: {total_reward:.4f}")
    return total_reward, lap, pit_count


if __name__ == "__main__":
    run()
