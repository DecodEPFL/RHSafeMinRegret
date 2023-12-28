import numpy as np
from scipy import linspace
from scipy.integrate import solve_ivp
from scipy.linalg import logm, matrix_balance

import src.conf as conf

#############################################################
#   VAN DER POL OSCILLATOR
#############################################################

m = 1
n = 2
p = 1

# Damping parameter for the Van der Pol oscillator
prm = 1.0

# Polytopic constraints
xb, ub, wb = 50, 50, 0.2
H = np.vstack((np.eye(n+m), -np.eye(n+m)))
h = np.array(([xb]*n + [ub]*m)*2)
Hw = np.vstack((np.eye(n+p), -np.eye(n+p)))
hw = np.array([wb]*(n+p)*2)


def dyn(t, z, u=None, q=None):
#############################################################
#   Function for the dynamics of Andrea's system
#   with parameter mu = p.
#
#   :param t: time instant.
#   :param z: tuple containing state (x).
#   :param u: system input (u).
#   :param q: damping parameter mu, default is vdp.prm.
#   :return: list of dx/dt and dy/dt.
#############################################################

    # Deal with optional input
    uk = np.array(list(u.keys())) if u is not None else None
    u = [0, 0] if u is None else u[uk[uk <= t].max()]
    
    # Unpack values
    x, i = np.array(list(z)), np.array(list(u))
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    # Continuous dynamics. Slow but needed for compatibility
    M = np.block([[np.array([[1, q], [0, 1]]),
                   np.array([[0], [1]])],
                  [np.array([[1, 0]]), np.eye(m)]])
    Ms, (sca, _) = matrix_balance(M, permute=0, separate=1)
    eM = logm(M)*(sca[:, None]*np.reciprocal(sca))*(1/conf.ts)
    
    # Return results of dynamics equation
    return eM[:n, :n] @ x + eM[:n, n:] @ i
    
    
def obs(t, z, u=None, q=None):
#############################################################
#   Function for the output of Andrea's system
#   with parameter mu = p.
#
#   :param t: time instant.
#   :param z: tuple containing state (x).
#   :param u: system input (u).
#   :param q: damping parameter mu (not used).
#   :return: list of dx/dt and dy/dt.
#############################################################
            
    return z[1]
    
    
def lin(t, z, u=None, q=None):
#############################################################
#   Function for the linearized dynamics of Andrea's system
#   with parameter mu = p.
#
#   :param t: time instant.
#   :param z: tuple containing state (x).
#   :param u: system input (u).
#   :param q: damping parameter mu, default is vdp.prm.
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    # Return linearization matrix of dynamics and output at z
    return np.array([[1, q], [0, 1]]), \
        np.array([[0], [1]]), np.array([[1, 0]])
