import sys as system
system.path.insert(0, system.path[0] + '/..')
import src.conf as conf
np = conf.np
from src.utils import plot_numpy_dict, eval
from src.models import foo
from src.models import andrea as ada
#from src.models.sim import simulate_system
from src.sls_cvx import sls, dro
#from src.sls import sls as slsreg
#from src.sls import dro

import seaborn as sns
import matplotlib.pyplot as plt
import cvxpy as cp

if __name__ == '__main__':
    
    sys = ada
    T = 5
    patterns = ["gaussian", "uniform0.5", "uniform", "constant",
                "sine", "sawtooth", "step", "stairs", "worst"]
    
    # We can use zeros for the trajectory because the system is LTI
    sls.set(sys, np.linspace(0, 1, T), np.zeros((sys.n, T)),
               cost=(np.eye(T*sys.n), np.eye(T*sys.m)), axis=1)


    # Build constraints lists
    _mkconsx = lambda phi : sls.mkcons(phi, sys.H, sys.h,
                                       sys.Hw, 0*sys.hw)
                                       
    _mkcons = lambda phi : sls.mkcons(phi, sys.H, sys.h,
                                      sys.Hw, sys.hw)
                        
    # Evaluate identity map to get just the patterns
    train_pattern = eval(np.eye((sls.n+sls.p)*sls.T),
                         sls.T, patterns, 20, average=False)
    dro.train(sys.wb*np.real(train_pattern["uniform"]))
    _drmkcons = lambda phi : dro.mkcons(phi, sys.H, sys.h)

        
    # Solve the h2, hinf and regret problems
    pxu, pxuv, puu, puuv = sls.min('h2', constraints=_mkconsx)
    px2, px2v, pu2, pu2v = sls.min('h2', constraints=_mkcons)
    pxd, pxdv, pud, pudv = dro.min('dro', constraints=_drmkcons)
    
    # stack v and w maps
    pxu, puu = np.hstack((pxu, pxuv)), np.hstack((puu, puuv))
    px2, pu2 = np.hstack((px2, px2v)), np.hstack((pu2, pu2v))
    pxd, pud = np.hstack((pxd, pxdv)), np.hstack((pud, pudv))
    print(pud.shape)
                
    e_hu = eval(np.vstack((pxu, puu)), sls.T, patterns, 1000)
    e_h2 = eval(np.vstack((px2, pu2)), sls.T, patterns, 1000)
    e_hd = eval(np.vstack((pxd, pud)), sls.T, patterns, 1000)
                
    print(" ".ljust(16), "[no noise, bounded, dro]")
    for p in patterns:
        e_hu[p] = np.linalg.norm(e_hu[p])
        e_h2[p] = np.linalg.norm(e_h2[p])
        e_hd[p] = np.linalg.norm(e_hd[p])
        e = np.array([e_h2[p], e_hd[p]])
        best = np.min(e)
        print(p.ljust(16), np.round((e/best - 1)*10000)/100)
    
    """
    plot_numpy_dict(
        {"h2" : np.sum((pvc @ v + pwc @ w).reshape((10, 2)), axis=1),
         "regret" : np.sum((pvr @ v + pwr @ w).reshape((10, 2)), axis=1),
         "x" : sol.t })
    """

