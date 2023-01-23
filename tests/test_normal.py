import sys as system
system.path.insert(0, system.path[0] + '/..')
import src.conf as conf
np = conf.np
from src.utils import plot_numpy_dict
from src.models import vdp
from src.models.sim import simulate_system
from src.sls import sls, dro

if __name__ == '__main__':
    sol = simulate_system(np.linspace(0, 1, 10), [1.0, 0.0], vdp)
    sls.set(vdp, sol.t, sol.x, cost=np.diag([1]*10 + [5]*10), axis=1)
        
    pwc, pvc = sls.min('causal')
    assert(pvc.shape == sls.C.T.shape)
    assert(pwc.shape == sls.A.shape)

    dro.eps = 1e-1
    pwr, pvr = dro.min('regret')
    assert(pvr.shape == sls.C.T.shape)
    assert(pwr.shape == sls.A.shape)
    
    pattern = "sine"
    if pattern == "sine":
        v = np.sin(np.pi*sol.t)
        w = np.repeat(v, 2)
    else:
        pass
    
    plot_numpy_dict(
        {"h2" : np.sum((pvc @ v + pwc @ w).reshape((10, 2)), axis=1),
         "regret" : np.sum((pvr @ v + pwr @ w).reshape((10, 2)), axis=1),
         "x" : sol.t })
    
