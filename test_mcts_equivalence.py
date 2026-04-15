import numpy as np
import random
from envs.river_swim_continuous import RiverSwimContinuous
from pure_mcts.mcts_dpw import MCTSStochastic as OriginalMCTS
from interpretable_mcts.mcts_dpw import MCTSStochastic as NewMCTS
from pure_mcts.mcts import MCTS as OriginalPureMCTS
from interpretable_mcts.mcts import MCTS as NewPureMCTS

def run_test(mcts_class):
    np.random.seed(42)
    random.seed(42)
    env = RiverSwimContinuous()
    env.seed(42)
    s = env.reset()
    if not isinstance(s, np.ndarray):
        s = np.array([s])
    
    # Patch env.seed so it doesn't break determinism
    env.seed = lambda *args: None
    
    mcts = mcts_class(root_index=s, root=None, model=None, na=env.action_space.n, gamma=0.99, alpha=0.44)
    
    budget = 100
    max_depth = 10
    n_iters = int(budget / max_depth)
    
    # MCTS signature requires (n_mcts, c, Env, mcts_env, budget, max_depth)
    mcts.search(n_mcts=n_iters, c=1.2, Env=env, mcts_env=env, budget=budget, max_depth=max_depth)
    root_index, pi, v = mcts.return_results(temp=1.0)
    
    return pi, v

if __name__ == '__main__':
    print('Testing MCTSStochastic (DPW)...')
    pi_orig, v_orig = run_test(OriginalMCTS)
    pi_new, v_new = run_test(NewMCTS)
    
    print('Original pi:', pi_orig, 'v:', v_orig)
    print('New pi:     ', pi_new, 'v:', v_new)
    
    assert np.allclose(pi_orig, pi_new), 'Policies differ!'
    assert np.allclose(v_orig, v_new), 'Values differ!'
    print('=> MCTSStochastic Equivalence Validated!\n')
    
    print('Testing Pure MCTS...')
    pi_orig2, v_orig2 = run_test(OriginalPureMCTS)
    pi_new2, v_new2 = run_test(NewPureMCTS)
    
    print('Original pi:', pi_orig2, 'v:', v_orig2)
    print('New pi:     ', pi_new2, 'v:', v_new2)
    
    assert np.allclose(pi_orig2, pi_new2), 'Policies differ!'
    assert np.allclose(v_orig2, v_new2), 'Values differ!'
    print('=> Pure MCTS Equivalence Validated!')
