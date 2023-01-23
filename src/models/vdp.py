import numpy as np
from scipy import linspace
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete as c2d

import src.conf as conf

#############################################################
#   VAN DER POL OSCILLATOR
#############################################################

m = 0
n = 2
p = 1

# Damping parameter for the Van der Pol oscillator
prm = 1


def dyn(t, z, u=None, q=None):
#############################################################
#   Function for the dynamics of the Van der Pol oscillator
#   with parameter mu = p.
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: damping parameter mu, default is vdp.prm.
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    # Unpack values
    x, y = z
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    # Return results of dynamics equation
    return np.array([y, q*(1 - x**2)*y - x])
    
    
def obs(t, z, u=None, q=None):
#############################################################
#   Function for the output of the Van der Pol oscillator
#   with parameter mu = p.
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: damping parameter mu (not used).
#   :return: list of dx/dt and dy/dt.
#############################################################
            
    return z[0]
    
    
def lin(t, z, u=None, q=None):
#############################################################
#   Function for the linearized dynamics of the Van der Pol
#   oscillator with parameter mu = p.
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: damping parameter mu, default is vdp.prm.
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    # Unpack values
    x, y = z
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    # Discretize
    sys = c2d((np.array([[0, 1],[-2*q*x*y - 1, q*(1 - x**2)]]),
               np.array([[0], [0]]), np.array([[1, 0]]),
               np.array([[0]])), conf.ts)
        
    # Return linearization matrix of dynamics and output at z
    return sys[0], sys[1], sys[2]
    # TODO: change this to ZOH model with conf.ts
