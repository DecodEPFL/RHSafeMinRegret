import sys as system
system.path.insert(0, system.path[0] + '/..')
import src.conf as conf
np = conf.np
from src.utils import plot_numpy_dict, eval
#from src.models import foo, vdp
from src.models import andrea as ada
#from src.models.sim import simulate_system
from src.sls_cvx import sls
#from src.sls import sls as slsreg
#from src.sls import dro

import seaborn as sns
import matplotlib.pyplot as plt
import cvxpy as cp

if __name__ == '__main__':
    """
    sol = simulate_system(np.linspace(0, 1, 10), [1.0, 0.0], vdp)
    slsreg.set(vdp, sol.t, sol.x, cost=np.diag([1]*10 + [5]*10), axis=1)
    sls.set(vdp, sol.t, sol.x, cost=(np.diag([1]*10 + [5]*10), None), axis=1)
        
    pwc, pvc = slsreg.min('noncausal')
    pwcv, pvcv, _, _ = sls.min('noncausal')
    np.set_printoptions(precision=2)
        
    pwc, pvc = slsreg.min('causal')
    pwcv, pvcv, _, _ = sls.min('h2')
    np.set_printoptions(precision=2)
    
    sns.heatmap(np.block([[pvc, pwc], [pvcv, pwcv]]))
    plt.show()
    sns.heatmap(np.block([[pvc - pvcv, pwc - pwcv]]))
    plt.show()

    
    pwr, pvr, _, _ = sls.min('worst case regret')
    assert(pvr.shape == slsreg.C.T.shape)
    assert(pwr.shape == slsreg.A.shape)
    
    sns.heatmap(np.block([[pvc, pwc], [pvr, pwr]]))
    plt.show()
    sns.heatmap(np.block([[pvc - pvr, pwc - pwr]]))
    plt.show()
    """
                           
    
    patterns = ["gaussian", "uniform0.5", "uniform", "constant",
                "sine", "sawtooth", "step", "stairs", "worst"]
    
    # We can use zeros for the trajectory because the system is LTI
    sls.set(ada, np.linspace(0, 1, 10), np.zeros((3, 10)),
               cost=(np.eye(30), np.eye(20)), axis=1)
    
    def _mkcons(phi):
        # State and input constraints
        H = np.kron(np.vstack((np.eye(ada.n+ada.m),
                               -np.eye(ada.n+ada.m))), sls.I)
        h = np.array(([3]*ada.n + [2]*ada.m)*2*sls.T)
                
        # disturbance constraint
        Hw = np.kron(np.vstack((np.eye(ada.n),
                                -np.eye(ada.n))), sls.I)
        hw = np.array([1]*ada.n*2*sls.T)

        # Dual variable
        Z = cp.Variable((Hw.shape[0], H.shape[0]))
        
        # List of additional constraints
        return [Z.T @ hw <= h, H @ phi == Z.T @ Hw, Z >= 0]
        
    # Solve the h2, hinf and regret problems
    px2, _, pu2, _ = sls.min('h2', constraints=_mkcons)
    pxi, _, pui, _ = sls.min('hinf', constraints=_mkcons)
    pxr, _, pur, _ = sls.min('worst case regret', constraints=_mkcons)
        
    e_h2 = eval(np.vstack((px2, pu2)), sls.T, patterns, 1000)
    e_hi = eval(np.vstack((pxi, pui)), sls.T, patterns, 1000)
    e_reg = eval(np.vstack((pxr, pur)), sls.T, patterns, 1000)
    
    for p in patterns:
        e_h2[p] = np.linalg.norm(e_h2[p])
        e_hi[p] = np.linalg.norm(e_hi[p])
        e_reg[p] = np.linalg.norm(e_reg[p])
        e = np.array([e_h2[p], e_hi[p], e_reg[p]])
        best = np.min(e)
        print(p.ljust(16), np.round((e/best - 1)*10000)/100)
    
    """
    plot_numpy_dict(
        {"h2" : np.sum((pvc @ v + pwc @ w).reshape((10, 2)), axis=1),
         "regret" : np.sum((pvr @ v + pwr @ w).reshape((10, 2)), axis=1),
         "x" : sol.t })
    """

