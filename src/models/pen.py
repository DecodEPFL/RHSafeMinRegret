import numpy as np
from scipy import linspace
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete as c2d

import src.conf as conf

#############################################################
#   INVERTED PENDULUM
#############################################################

m = 0
n = 4
p = 2

# polytopic constraints
H, h = None, None
Hw, hw = None, None

# Parameters for the inverted pendulum
# Length of pendulum [m], mass of bob [kg], mass of cart [kg],
# damping for bob [Ns/rad], damping for cart [Ns/m],
# and P and I control gains.
prm = 2.0, 0.1, 0.1, 1.0, 1.0, 10.0, 2.0

# Integrator for PI
integral = conf.AttributeDict({'x' : 0, 'v' : 0})

def dyn(t, z, u=None, q=None):
#############################################################
#   Function for the dynamics of the inverted pendulum
#   with parameter mu (global variable).
#   This function modifies the controller integral !!!
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: tuple contating the parameters of the pendulum.
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    # Unpack values
    v, w, x, y = z
    
    # Update integral controller
    integral.x += x
    integral.v += v
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    g = 9.8 # Gravitational Acceleration
    L, m, M, damping_theta, damping_x, pg, ig = q
        
    # Controller input
    u = -pg * x - ig*integral.x
    
    # Compute dynamics
    x_ddot = (L * y*y * np.cos(x)  -  g * np.cos(x) *  np.sin(x)) \
        * m / ( m* np.sin(x)* np.sin(x) - M -m ) + u

    theta_ddot = -g/L * np.cos(x) - 1./L * np.sin(x) * x_ddot

    return np.array([ w, x_ddot - damping_x*w,
                      y, theta_ddot - damping_theta*y ])
    
def obs(t, z, u=None, q=None):
#############################################################
#   Function for the output of the inverted pendulum
#   with parameter mu (global variable).
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: tuple contating the parameters of the pendulum
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    return np.array([z[0], z[2]])

def lin(t, z, u=None, q=None):
#############################################################
#   Function for the linearized dynamics of the inverted
#   pendulum with parameter mu (global variable).
#
#   :param t: time instant.
#   :param z: tuple containing position (x) and speed (y).
#   :param u: system input (empty).
#   :param q: tuple contating the parameters of the pendulum
#   :return: list of dx/dt and dy/dt.
#############################################################
    
    # Unpack values
    v, w, x, y = z
    
    # Use default parameters if none given
    q = prm if q is None else q
    
    g = 9.8 # Gravitational Acceleration
    L, m, M, damping_theta, damping_x, pg, ig = q
        
    # Controller input
    u = -pg * x - ig*integral.x
    
    # Don't forget ZOH
    raise NotImplementedError("linearization of inverted \
                               pendulum not implemented!")

    return
