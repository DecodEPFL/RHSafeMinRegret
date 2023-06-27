import numpy as np
from scipy import linspace
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete as c2d

import src.conf as conf

#############################################################
#   DUMMY
#############################################################

m = 1
n = 2
p = 1

# Damping parameter for the Van der Pol oscillator
prm = 0.2

# Polytopic constraints
xb, ub, wb = 5, 5, 0.02
H = np.vstack((np.eye(n+m), -np.eye(n+m)))
h = np.array(([xb]*n + [ub]*m)*2)
Hw = np.vstack((np.eye(n+p), -np.eye(n+p)))
hw = np.array([wb]*(n+p)*2)


def dyn(t, z, u=None, q=None):
#############################################################
#   Function for the dynamics of the dummy with parameter q.
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: parameter for matrix A = I - p11, default is foo.prm
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    # Unpack values
    x, y = z
    
    uk = np.array(list(u.keys())) if u is not None else None
    u = [0] if u is None else u[uk[uk <= t].max()]
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    # Return results of dynamics equation
    return np.array([[1.0 - q, -q], [-q, 1.0 - q]]) @ [x, y] \
        + np.array([[1.0], [0.0]]) @ u
    

def obs(t, z, u=None, q=None):
#############################################################
#   Function for the output of the dummy.
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: parameter (not used).
#   :return: list of dx/dt and dy/dt.
#############################################################
            
    return z[0]
    

def lin(t, z, u=None, q=None):
#############################################################
#   Function for the linear dynamics of the dummy
#   with parameter q.
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: parameter for matrix A = I - p11, default is foo.prm
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    # Discretize
    sys = c2d((np.array([[1.0 - q, -q], [-q, 1.0 - q]]),
               np.array([[1], [0]]), np.array([[1, 0]]),
               np.array([[0]])), conf.ts)
        
    # Return linearization matrix of dynamics and output at z
    return sys[0], sys[1], sys[2]
