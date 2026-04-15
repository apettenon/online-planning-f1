import os
import sys
import numpy as np
import time

# Ensure python can import from local standard folders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f1_env.race_strategy_event import RaceEnv
from mcts.mcts_dpw import MCTSStochastic

def run():
    print("Initializing F1 Race Environment...")
    env = RaceEnv(horizon=70, skip_steps=True)
    
    budget = 1000
    gamma = 1.0
    c_dpw = 1.2
    max_depth = 15
    alpha = 0.6
    
    np.random.seed(42)
    env.seed()
    s = env.reset()
    
    mcts = MCTSStochastic(root_index=s, root=None, model=None, na=env.action_space.n, gamma=gamma, alpha=alpha)
    n_iters = int(budget / max_depth)
    
    print(f"\nStarting pure MCTS run with budget {budget}...")
    start_time = time.time()
    
    mcts.search(n_mcts=n_iters, c=c_dpw, Env=env, mcts_env=env, budget=budget, max_depth=max_depth)
    
    print(f"Search completed in {time.time() - start_time:.2f} seconds.")
    root_index, pi, v = mcts.return_results(temp=1.0)
    
    print("Action Probabilities:", pi)
    print("Expected Value:", v)

if __name__ == '__main__':
    run()
