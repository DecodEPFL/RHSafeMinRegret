import numpy as np
from scipy import linspace
from scipy.integrate import solve_ivp
from scipy.linalg import logm, matrix_balance

import src.conf as conf

#############################################################
#   VAN DER POL OSCILLATOR
#############################################################

m = 2
n = 3
p = 0

# Damping parameter for the Van der Pol oscillator
prm = 0.7


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
    M = np.block([[q*np.array([[0.7, 0.2, 0], [0.3, 0.7, -0.1],
                               [0, -0.2, 0.8]]),
                   np.array([[1, 0.2], [2, 0.3], [1.5, 0.5]])],
                  [np.zeros((m, n)), np.eye(m)]])
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
            
    return z
    
    
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
    return q*np.array([[0.7, 0.2, 0], [0.3, 0.7, -0.1],
                       [0, -0.2, 0.8]]), \
        np.array([[1, 0.2], [2, 0.3], [1.5, 0.5]]), np.ones((0, 3))
