import numpy as np

# Sampling time for discrete time
ts = 0.01

# Easily define objects with dot access
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Column vectorization function, and add to np
def vec(x):
    return x.reshape((-1, 1), order="F")

np.vec = vec

# Set seed for reproductibility
np.random.seed(1234)
