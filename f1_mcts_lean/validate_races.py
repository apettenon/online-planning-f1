"""
Multi-race validation for QL-OL-UCT.

Runs the paper's algorithm (QL-OL-UCT) on a set of 2017 races and
reports actual race time (seconds) for the controlled driver (Vettel #5),
suitable for comparison against paper Table 1.

Race time is read directly from the simulator's cumulative time register
at race end — not derived from the reward signal.

Run from f1_mcts_lean/:
    ../venv/bin/python3 validate_races.py [--races clean|all|<name>]

Examples:
    ../venv/bin/python3 validate_races.py                  # clean races only
    ../venv/bin/python3 validate_races.py --races all      # all 2017 races
    ../venv/bin/python3 validate_races.py --races Monza    # single race
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from f1_env.race_strategy_event import RaceEnv
from mcts.mcts_ol_uct import QL_OL_UCT

# ------------------------------------------------------------------
# Hyper-parameters (paper spec)
# ------------------------------------------------------------------
BUDGET    = 10000
GAMMA     = 1.0
C_UCB     = 1.2      # from race_strategy_full_ol.json config
ALPHA_0   = 1.0
LR_DECAY  = 0.99
HORIZON   = 70
MAX_DEPTH = 70

# ------------------------------------------------------------------
# Race catalogue
# ------------------------------------------------------------------
# 2017 races with NO FCY events — safest for first validation pass.
CLEAN_RACES = [
    "pars_Melbourne_2017.ini",     # Australian GP — paper tuning race
    "pars_Monza_2017.ini",         # Italian GP
    "pars_Spielberg_2017.ini",     # Austrian GP
    "pars_KualaLumpur_2017.ini",   # Malaysian GP
    "pars_Austin_2017.ini",        # US GP
    "pars_YasMarina_2017.ini",     # Abu Dhabi GP
]

# 2017 races WITH FCY events (deepcopy fix should handle these).
FCY_RACES = [
    "pars_Catalunya_2017.ini",     # Spanish GP   — VSC laps 33–36
    "pars_Sakhir_2017.ini",        # Bahrain GP   — SC laps 12–15
    "pars_Shanghai_2017.ini",      # Chinese GP   — VSC+SC at start
    "pars_Silverstone_2017.ini",   # British GP   — SC laps 1–3
    "pars_Spa_2017.ini",           # Belgian GP   — SC laps 29–32
    "pars_Budapest_2017.ini",      # Hungarian GP — SC lap 1
    "pars_Sochi_2017.ini",         # Russian GP   — SC lap 1
    "pars_SaoPaulo_2017.ini",      # Brazilian GP — SC laps 1–4
    "pars_MexicoCity_2017.ini",    # Mexican GP   — VSC laps 31–33
    "pars_Singapore_2017.ini",     # Singapore GP — 3× SC
    "pars_MonteCarlo_2017.ini",    # Monaco GP    — SC lap 60
    "pars_Baku_2017.ini",          # Azerbaijan GP— 2× SC
    "pars_Suzuka_2017.ini",        # Japanese GP  — SC+VSC
]

def get_action_name(env, a):
    """Human-readable action name using actual compound mapping."""
    if a == 0:
        return "STAY"
    try:
        comp = env.map_action_to_compound(a)
        return f"PIT_{comp}"
    except Exception:
        return f"PIT_{a}"


def get_final_race_time(env):
    """
    Read the race time (seconds) for the controlled driver from the simulator.

    Returns (total_s, planning_s) where:
      total_s    — full cumulative race time from lap 0
      planning_s — time from planning start lap (lap 8) to finish
                   This matches the paper's Table 1 metric (cumulative return
                   from lap 8 = sum of lap times laps 8..end).
    Returns (None, None) if unavailable.
    """
    try:
        sim_state = env._race_sim.get_simulation_state()
        lap = sim_state[0]  # current lap (= race_length at end)
        cum_times = sim_state[2]  # shape (race_length+1, n_drivers)
        start_lap = getattr(env, 'start_lap', 7)  # default = 7 (planning from lap 8)

        for driver_entry in sim_state[10]:
            carno = driver_entry[2]
            if carno in env._active_drivers:
                sim_idx = env._race_sim.drivers_mapping[carno]
                total_s = float(cum_times[lap][sim_idx])
                planning_s = total_s - float(cum_times[start_lap][sim_idx])
                return total_s, planning_s
    except Exception as e:
        print(f"  [WARN] Could not read race time from sim state: {e}")
    return None


def run_race(race_pars_file, seed=42, verbose=True):
    """Run one full-race episode with QL-OL-UCT. Returns result dict."""
    np.random.seed(seed)

    env = RaceEnv(
        horizon=HORIZON,
        skip_steps=True,
        race_pars_file=race_pars_file,
    )
    env.seed(seed)
    s = env.reset()

    mcts = QL_OL_UCT(
        root=None, root_index=s, na=env.action_space.n,
        gamma=GAMMA, alpha_0=ALPHA_0, lr_decay=LR_DECAY,
    )

    total_reward = 0.0
    step = 0
    pit_count = 0
    done = False
    wall_start = time.time()

    if verbose:
        print(f"\n  {'Lap':>4}  {'Step':>5}  {'MCTSAct':>10}  {'ExecAct':>10}  {'Reward':>8}  {'CumRew':>9}")
        print("  " + "-" * 58)

    while not done:
        step += 1
        lap_before = env._lap

        mcts.search(n_mcts=BUDGET, c=C_UCB, Env=env, mcts_env=env,
                    budget=BUDGET, max_depth=MAX_DEPTH)

        _, pi, _ = mcts.return_results(temp=0)
        a_mcts = int(np.argmax(pi))

        # Determine which action env will actually execute (lockout → STAY)
        agent = env.get_next_agent()
        available = env.get_available_actions(agent)
        a_exec = a_mcts if a_mcts in available else 0

        s1, r, done, _ = env.step(a_mcts)
        total_reward += r
        if a_exec > 0:
            pit_count += 1

        if verbose:
            mcts_name = get_action_name(env, a_mcts)
            exec_name = get_action_name(env, a_exec)
            flag = " *" if a_exec != a_mcts else ""
            print(f"  {lap_before:>4}  {step:>5}  {mcts_name:>10}  {exec_name:>10}  "
                  f"{r:>8.4f}  {total_reward:>9.4f}{flag}")

        mcts.forward(a_mcts, s1, r)
        s = s1

        if step >= HORIZON:
            break

    wall_elapsed = time.time() - wall_start

    times = get_final_race_time(env)
    total_s, planning_s = times if times else (None, None)

    return {
        "race": race_pars_file.replace("pars_", "").replace(".ini", ""),
        "total_reward": total_reward,
        "steps": step,
        "pits": pit_count,
        "race_time_total_s": total_s,
        "race_time_planning_s": planning_s,  # from lap 8, matches paper Table 1
        "wall_time_s": wall_elapsed,
    }


def print_summary(results):
    print("\n" + "=" * 80)
    print(f"{'Race':<25} {'Total(s)':>10} {'FromLap8(s)':>12} {'Paper':>10} "
          f"{'Pits':>5} {'WallTime':>10}")
    print("-" * 80)
    # Paper Table 1 QL-OL UCT values (2017 races only; others N/A)
    PAPER = {
        "Australia_2017": 4459.71,
        "Spain_2017":     5188.05,
        "Austria_2017":   4465.85,
        "Belgium_2017":   4246.09,
        "Russia_2017":    4421.54,
    }
    for r in results:
        tot = f"{r['race_time_total_s']:.1f}" if r['race_time_total_s'] else "N/A"
        pl  = f"{r['race_time_planning_s']:.1f}" if r['race_time_planning_s'] else "N/A"
        paper = PAPER.get(r['race'], "N/A")
        paper_s = f"{paper:.2f}" if isinstance(paper, float) else paper
        print(f"{r['race']:<25} {tot:>10} {pl:>12} {paper_s:>10} "
              f"{r['pits']:>5} {r['wall_time_s']:>9.0f}s")
    print("=" * 80)
    print("FromLap8(s) = time from planning start (lap 8) — comparable to paper Table 1.")
    print("Paper values: QL-OL UCT column, averaged over 100 runs (lower is better).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--races", default="clean",
        help="Race set: 'clean' (no FCY, default), 'all' (all 2017), "
             "or a location name substring e.g. 'Monza'"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.races == "clean":
        races = CLEAN_RACES
        label = "clean (no FCY)"
    elif args.races == "all":
        races = CLEAN_RACES + FCY_RACES
        label = "all 2017"
    else:
        all_races = CLEAN_RACES + FCY_RACES
        races = [r for r in all_races if args.races.lower() in r.lower()]
        if not races:
            print(f"No race found matching '{args.races}'")
            sys.exit(1)
        label = f"matching '{args.races}'"

    print("=" * 70)
    print(f"QL-OL-UCT MULTI-RACE VALIDATION  ({label})")
    print(f"  budget={BUDGET}, γ={GAMMA}, C={C_UCB}, α_0={ALPHA_0}, decay={LR_DECAY}")
    print(f"  driver: Vettel #5 (2017)")
    print("=" * 70)

    results = []
    for race_file in races:
        race_name = race_file.replace("pars_", "").replace(".ini", "")
        print(f"\n{'=' * 60}")
        print(f"Running: {race_name}")
        print(f"{'=' * 60}")
        try:
            result = run_race(race_file, seed=args.seed, verbose=True)
            results.append(result)
            tot = f"{result['race_time_total_s']:.1f}s" if result['race_time_total_s'] else "N/A"
            pl  = f"{result['race_time_planning_s']:.1f}s" if result['race_time_planning_s'] else "N/A"
            print(f"\n  -> Total: {tot}  FromLap8: {pl}  |  Pits: {result['pits']}  |"
                  f"  Wall: {result['wall_time_s']:.0f}s")
        except Exception as e:
            print(f"  [ERROR] {race_name} failed: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
