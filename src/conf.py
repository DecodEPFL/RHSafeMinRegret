import numpy as np
from scipy.optimize import minimize

# Sampling time for discrete time
ts = 0.0025 # 0.1

# Easily define objects with dot access
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Column vectorization function, and add to np
def custom_vec(x):
    return x.reshape((-1, 1), order="F")
    
# Least absolute deviations, and add to np
def custom_least_ad(_a, _b, opt=None):
    # Make b 2 dimensional
    _b = _b if len(_b.shape) == 2 else _b[:, None]

    # Get initial guess
    x = np.linalg.lstsq(_a, _b, rcond=None)[0]

    # L1 norm function
    c = lambda p, a, b: np.sum(np.abs(b - a @ p))

    # Solve each regression (each column of _b)
    for i in range(_b.shape[1]):
        x[:, i] = minimize(c, x[:, i], options=opt,
                            args=(_a, _b[:,i])).x

    return x

# Add custom functions to np for convenience
np.vec = custom_vec
np.linalg.lstad = custom_least_ad

# Set print settingq
np.set_printoptions(precision=3, linewidth=10000)

# Set seed for reproductibility
np.random.seed(123)
