import sys as system
system.path.insert(0, system.path[0] + '/..')
import src.conf as conf
np = conf.np
from src.utils import plot_numpy_dict, eval, eval_infty
from src.models import foo
from src.models import andrea as ada
from src.models import double_integrator as din
from src.sls_cvx import sls, dro
from tqdm import tqdm
from scipy.linalg import sqrtm

import seaborn as sns
import matplotlib.pyplot as plt
import cvxpy as cp

if __name__ == '__main__':
    
    sys = din
    #sys.prm = 1.1
    #sys.prm = 0.2
    T, test_T = 7, 100
    n_train, n_test = 20, 500
    # TODO: understand why constant is a problem
    # Constant is a problem if T is too small (controllability)
    patterns = ["gaussian", "uniform", "uniform0.5", "constant",
                "sine", "sawtooth", "step", "stairs"]
    #patterns = ["gaussian", "uniform0.5", "sine"]
    
    # We can use zeros for the trajectory because the system is LTI
    sls.set(sys, np.linspace(0, 1, T), np.zeros((sys.n, T)),
               cost=(np.eye(sys.n), np.eye(sys.m)), axis=1)
               
    # Evaluate identity map to get just the patterns
    train_pattern = eval_infty(np.eye(sls.n+sls.p), sys, sls.T,
                               patterns, n_train)
    # Use more samples for Gaussian mean and variance
    mean_pattern = eval_infty(np.eye(sls.n+sls.p), sys, sls.T,
                              patterns, n_test)
               
    # Deal with each pattern separately for training
    e_g, e_e, e_r = dict(), dict(), dict()
    for pat in tqdm(patterns):

#        dro.eps = np.max(np.linalg.svd(train_pattern[pat],
#                         compute_uv=False))/np.sqrt(n_train)
#        dro.train(0*train_pattern[pat][:, [0]])
                
#        px, pxv, pu, puv = sls.min('h2 infty', constraints=None)
#        pxg, pug = np.hstack((px, pxv)), np.hstack((pu, puv))
        
        # Gaussian centered on mean and with same std as data
        # can be implemented as dro with empirical = 2nd moment.
        # This includes bias and variance, unlike h2 that assumes
        # zero-mean noise.
        dro.eps = 1e-4*sys.wb
        dro.train(sqrtm((mean_pattern[pat] @ mean_pattern[pat].T)/n_test
                        + 1e-8*np.eye(mean_pattern[pat].shape[0])))
        px, pxv, pu, puv = \
            dro.min('dro infty', constraints=lambda phi :
                    dro.mkcons(phi, sys.H, sys.h, repeat=False))
        pxg, pug = np.hstack((px, pxv)), np.hstack((pu, puv))
        
        # Empirical distribution
        dro.eps = 1e-4*sys.wb
        dro.train(train_pattern[pat])
        px, pxv, pu, puv = \
            dro.min('dro infty', constraints=lambda phi :
                    dro.mkcons(phi, sys.H, sys.h, repeat=False))
        pxe, pue = np.hstack((px, pxv)), np.hstack((pu, puv))
        
        # DRO
        dro.eps = 0.2*sys.wb
        px, pxv, pu, puv = \
            dro.min('dro infty', constraints=lambda phi :
                    dro.mkcons(phi, sys.H, sys.h, repeat=False))
        pxr, pur = np.hstack((px, pxv)), np.hstack((pu, puv))
                
        # Evaluate policies
        e_g[pat] = eval_infty(np.vstack((pxg, pug)), sys,
                              test_T, [pat], n_test, cost=sls.Ss2)[pat]
        e_e[pat] = eval_infty(np.vstack((pxe, pue)), sys,
                              test_T, [pat], n_test, cost=sls.Ss2)[pat]
        e_r[pat] = eval_infty(np.vstack((pxr, pur)), sys,
                              test_T, [pat], n_test, cost=sls.Ss2)[pat]
    
    # Display results
    print(" ".ljust(16), "[h2".ljust(16), "emp".ljust(15),
          "dro".ljust(15), "]")
    
    err_plot = {"x": np.arange(test_T+1)}
    for pat in patterns:
        if pat == "step":
            err_plot["g"] = np.mean(np.abs(e_g[pat]), axis=1)
            err_plot["e"] = np.mean(np.abs(e_e[pat]), axis=1)
            err_plot["r"] = np.mean(np.abs(e_r[pat]), axis=1)
            
        e_g[pat] = np.linalg.norm(np.mean(np.abs(e_g[pat]),
                                          axis=1))/test_T
        e_e[pat] = np.linalg.norm(np.mean(np.abs(e_e[pat]),
                                          axis=1))/test_T
        e_r[pat] = np.linalg.norm(np.mean(np.abs(e_r[pat]),
                                          axis=1))/test_T
        e = np.array([e_g[pat], e_e[pat], e_r[pat]])
        e = np.round((e/np.min(e) - 1)*10000)/100
        
        s = "["
        for err in e:
            s += str(err).ljust(16) if err > 0 else conf.ANSI["GREEN"] \
                + str(1).ljust(16) + conf.ANSI["RESET"]
        s += "]"
        print(pat.ljust(16), s)
    
    plot_numpy_dict(err_plot, x_label="x", y_label="cost")

