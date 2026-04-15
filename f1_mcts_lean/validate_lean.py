"""
Validation script for f1_mcts_lean.
Runs a full race episode with MCTS at each lap decision and prints:
  - per-lap: action chosen, raw reward, cumulative reward
  - final: total episode reward, laps completed, number of pit stops
Run from the f1_mcts_lean/ directory: python validate_lean.py
"""

import os
import sys
import numpy as np
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from f1_env.race_strategy_event import RaceEnv
from mcts.mcts_dpw import MCTSStochastic

BUDGET    = 500
GAMMA     = 1.0
C_UCB     = 1.2
MAX_DEPTH = 15
ALPHA     = 0.6
HORIZON   = 70

ACTION_NAMES = {0: "STAY", 1: "PIT_A3", 2: "PIT_A4"}

def run():
    print("=" * 60)
    print("LEAN REPO VALIDATION — Catalunya 2017, single agent")
    print("=" * 60)

    np.random.seed(42)
    env = RaceEnv(horizon=HORIZON, skip_steps=True, race_pars_file="pars_Suzuka_2016.ini")
    env.seed(42)
    s = env.reset()

    mcts = MCTSStochastic(root_index=s, root=None, model=None,
                          na=env.action_space.n, gamma=GAMMA, alpha=ALPHA)

    total_reward = 0.0
    lap = 0
    pit_count = 0
    done = False

    print(f"\n{'Lap':>4}  {'Action':>8}  {'Reward':>8}  {'CumRew':>9}")
    print("-" * 40)

    while not done:
        lap += 1
        n_iters = max(1, BUDGET // MAX_DEPTH)

        t0 = time.time()
        mcts.search(n_mcts=n_iters, c=C_UCB, Env=env, mcts_env=env,
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

    print("-" * 40)
    print(f"Episode complete — laps: {lap}, pits: {pit_count}, total reward: {total_reward:.4f}")
    approx_race_time = sum([(1 - (total_reward / lap)) * 300]) * lap if lap > 0 else float('nan')
    print(f"Approx cumulative race time: ~{approx_race_time:.1f} s")
    return total_reward, lap, pit_count

if __name__ == "__main__":
    run()
