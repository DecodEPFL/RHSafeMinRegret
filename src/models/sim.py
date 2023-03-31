import numpy as np
from scipy import linspace
from scipy.integrate import solve_ivp

import src.conf as conf


def simulate_system(t, x0, sys, u=None, w=None):
#############################################################
#   Function for the simulating the trajectory of
#   the Van der Pol oscillator.
#
#   :param t: list of time instants used in the simulation.
#   :param x0: initial point at min(t).
#   :param sys: function returning the dynamics dx/dt of
#       the system for a given state x.
#   :param u: (optional) zoh input sequence at each time in t.
#       This argument is set to None if its shape doesn't fit.
#       To use different times as t, give a dictionary where
#       the keys are the corresponding instants.
#   :param w: (optional) disturbance sequence at each time step.
#       This argument is set to None if its shape doesn't fit.
#       To use different times as t, give a dictionary where
#       the keys are the corresponding instants.
#   :return: object with attributes
#       t: the list of time instants
#       y: 2D array with y[0] being the list of positions
#          and y[1] the list of speeds.
#############################################################

    # Make array
    t, x0 = np.array(t), np.array(x0)

    # If u is a ndarray, not a dictionary containing times
    if type(u) is np.ndarray:
        # Manage different axis
        u = u if u.shape[0] == len(t)-1 else \
            (u.T if u.shape[1] == len(t)-1 else None)
            
        # Make a dictionary linking time to inputs
        u = dict(zip(t[:-1], u.tolist()))
            
    if type(w) is np.ndarray:
        # Manage different axis
        w = w if w.shape[0] == len(t)-1 else \
            (w.T if w.shape[1] == len(t)-1 else None)
            
        # Make a dictionary linking time to disturbances
        w = dict(zip(t[:-1], w.tolist()))
    else:
        w = dict(zip(t[:-1], (0*t[:-1]).tolist()))
    
    def ct2d(_x, _t):
        _b = np.array(list(_x.keys())) <= _t
        return _x[np.array(list(_x.keys()))[_b].max()]

    # Use scipy integration to simulate system
    sol = solve_ivp(lambda t, y : sys.dyn(t, y, u)
                    + ct2d(w, t),  [np.min(t), np.max(t)],
                    x0, t_eval=t)
    
    # Adjust notations
    sol.x = sol.y.copy()
    sol.y = sys.obs(sol.t, sol.x)
    
    # Return the solution
    return sol
