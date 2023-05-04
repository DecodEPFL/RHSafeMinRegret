import numpy as np
from scipy.optimize import linprog
import cvxpy as cp


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
def custom_least_ad(_a, _b):
    m, n = _a.shape
    
    # Make b 2 dimensional
    _b = _b if len(_b.shape) == 2 else _b[:, None]
    
    # Placeholder for opt variables
    # There is a slack variable for |x|
    # to keep the general LP formulation
    x = np.zeros((n, _b.shape[1]))
    
    # Create problem matrices
    c = np.concatenate((np.zeros(n), np.ones(m)))
    A = np.block([[_a, -np.eye(m)],
                  [-_a, -np.eye(m)]])
    b = np.kron([[1], [-1]], _b)
    
    # Solve x col by col
    for i in range(_b.shape[1]):
        x[:, i] = linprog(c, A_ub=A, b_ub=b[:, i],
                          bounds=(None, None)).x[:n]

    return x

# Add custom functions to np for convenience
np.vec = custom_vec
np.linalg.lstad = custom_least_ad

# Set print settingq
np.set_printoptions(precision=3, linewidth=10000)

# Set seed for reproductibility
np.random.seed(123)
