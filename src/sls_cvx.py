from src.conf import np
from tqdm import tqdm
import src.conf as conf
from scipy.linalg import sqrtm
import cvxpy as cp

class sls:
#############################################################
#   SYSTEM LEVEL SYNTHESIS OBSERVER DESIGN
#############################################################
    T, I, Z, A, C, S, Ss2, verbose = [0] + [None] * 7
    m, n, p, Im, In, Ip = [0] * 3 + [None] * 3
    
    _mkvar = lambda x: cp.Variable(x) if x[0] > 0 \
        and x[1] > 0 else np.zeros(x)

    @staticmethod
    def set(sys, t, z, cost=None, axis=0):
    #############################################################
    #   Build the system level synthesis based on state space.
    #
    #   :param sys: state space description of the system. It
    #               it needs the functions dyn, obs and lin.
    #   :param t: time instants of the samples in the trajectory
    #   :param z: state trajectory
    #   :param cost: quadratic cost matrix for error (default 1)
    #   :param axis: time axis for z (default 0)
    #############################################################
        
        # Deal with transposed trajectories
        if axis > 1 or axis < 0 or len(t) != z.shape[axis] \
            or z.shape[1 - axis] != sys.n:
            raise IndexError("index out of range")
        elif axis == 1:
            z = z.copy().T
        
        # length of time horizon
        sls.T, sls.m, sls.n, sls.p = z.shape[0], sys.m, sys.n, sys.p

        # Handy matrices
        sls.I, sls.Im = np.eye(sls.T), np.eye(sls.T * sls.m)
        sls.In, sls.Ip = np.eye(sls.T * sls.n), np.eye(sls.T * sls.p)
        sls.Z = np.eye(sls.T * sls.n, k=-sls.n)
        
        # Init System matrices
        sls.A = (0*sls.In).copy()
        sls.B = (0*sls.In[:, :sls.m*sls.T]).copy()
        sls.C = (0*sls.In[:sls.p*sls.T, :]).copy()
        for i in range(sls.T):
            sls.A[i*sls.n:(i+1)*sls.n, i*sls.n:(i+1)*sls.n], \
            sls.B[i*sls.n:(i+1)*sls.n, i*sls.m:(i+1)*sls.m], \
            sls.C[i*sls.p:(i+1)*sls.p, i*sls.n:(i+1)*sls.n] = \
                sys.lin(t[i], z[i, :])
        
        # Index matrices for closed loop maps
        iw = [True] * (sls.n*sls.T) + [False] * (sls.p*sls.T)
        ix = [True] * (sls.n*sls.T) + [False] * (sls.m*sls.T)
        
        sls.ixw = np.ix_(ix, iw)
        sls.ixv = np.ix_(ix, ~np.array(iw))
        sls.iuw = np.ix_(~np.array(ix), iw)
        sls.iuv = np.ix_(~np.array(ix), ~np.array(iw))
                            
        # Error cost
        if cost is not None and len(cost) == 2:
            if cost[0] is not None and cost[1] is not None:
                o = np.zeros((cost[0].shape[0], cost[1].shape[1]))
                sls.S = np.block([[cost[0], o], [o.T, cost[1]]])
            elif cost[0] is None:
                sls.S = cost[1]
            elif cost[1] is None:
                sls.S = cost[0]
        else:
            sls.S = cost
        
        sls.Ss2 = 1 if cost is None else sqrtm(sls.S)
        
        # Weighted least squares solver
        sls._solver = cp.MOSEK
        
    @staticmethod
    def min(objective='causal', _solver=None, constraints=None):
    #############################################################
    #   Build the error maps minimizing the given objective.
    #   Use sls.set before to initialize the system!
    #
    #   :param objective: string parameter:
    #       - causal for causal h2 observer
    #       - noncausal for clairvoyant observer
    #       - regret for minimal regret observer
    #   :param _solver: (optional) solver from cvxpy,
    #       default is cp.Mosek
    #   :param constraints: (optional) additional constraints,
    #       must be a function taking the closed loop map (phi)
    #       as input and returning a list of cvxpy constraints
    #   :return: sls maps of distrubance and noise to
    #       state and input errors
    #############################################################
    
        if sls.I is None:
            raise ValueError("sls object is not set with a system")
                
        # Make a progress bar if verbose is on
        pb = tqdm if sls.verbose else lambda x: x

        # Deal with optional parameters
        _solver = sls._solver if _solver is None else _solver
        if constraints is None:
            constraints = lambda x: []
        
        # Build noncausal noise and disturbance to error sls maps
        if objective != 'h2' and objective != 'hinf':
            # Full variables for unconstrained noncausal problem
            _phi = sls._opt_variables()
            
            # H2 or Hinf objectives are equivalent without causality
            obj = cp.norm(sls.Ss2 @ _phi, 'fro')
            
            # Achievability constraints
            cons = sls._achievability(_phi) + constraints(_phi)
            
            # Solve the problem. Results are directly stored in vars
            cp.Problem(cp.Minimize(obj), cons).solve(solver=_solver)
            
            # Only keep the values and return if needed
            _phi = _phi.value
            if objective == 'noncausal':
                return _phi[sls.ixw], _phi[sls.ixv], \
                       _phi[sls.iuw], _phi[sls.iuv]
        
        # If causal result is asked, move on to that problem
                
        # Lower triangular variables for causal problem
        _pc = sls._opt_variables(True)
        
        # Achievability constraint
        cons = sls._achievability(_pc) + constraints(_pc)
        
        if objective == "h2" or objective == "hinf":
            # H2 optimizes fro norm and hinf spectral norm
            obj = cp.norm(sls.Ss2 @ _pc,
                          2 if objective == "hinf" else 'fro')
                                      
            # Solve the problem.
            cp.Problem(cp.Minimize(obj),
                       cons).solve(solver=_solver)
                                   
            # Return values
            _pc = _pc.value
            return _pc[sls.ixw], _pc[sls.ixv], \
                   _pc[sls.iuw], _pc[sls.iuv]
        
        elif objective == "expected regret":
            raise NotImplementedError(objective + \
                                      " is the same as h2.")
        
        elif objective == "worst case regret":
            # Get the regret problem
            obj, sdpc = sls._opt_regret(_pc, _phi)
                                      
            # Solve the problem.
            cp.Problem(cp.Minimize(obj), cons
                       + sdpc).solve(solver=_solver)
                       
            # Return values
            _pc = _pc.value
            return _pc[sls.ixw], _pc[sls.ixv], \
                   _pc[sls.iuw], _pc[sls.iuv]
        
        elif objective == "expected immitation":
            raise NotImplementedError(objective + \
                                      " is the same as h2.")
        
        elif objective == "worst case immitation":
            # Get the regret problem
            obj, sdpc = sls._opt_immitation(_pc, _phi)
                                      
            # Solve the problem.
            cp.Problem(cp.Minimize(obj), cons
                       + sdpc).solve(solver=_solver)
                       
            # Return values
            _pc = _pc.value
            return _pc[sls.ixw], _pc[sls.ixv], \
                   _pc[sls.iuw], _pc[sls.iuv]
        
        # Wrong objective passed
        raise ValueError(str(objective) + " is not a valid " \
                         + "objectve. Valid objectives are: " \
                         + "'causal', 'noncausal', and 'regret'.")
        
    @staticmethod
    def _opt_variables(causal=False):
    #############################################################
    #   Builds the Phi matrices as optimization variables.
    #
    #   :param causal: sets the upper block-triangle to zero
    #   :return: Optimization variables for cvxpy
    #############################################################
    
        # Make each 4 blocks of the output feedback matrix
        # Some are [] if sls.m or sls.p are zero
        if causal:
            _xv = [ [sls._mkvar((sls.n, sls.p*i)),
                     np.zeros((sls.n, sls.p*(sls.T - i)))]
                   for i in range(1, sls.T)] + \
                   [[sls._mkvar((sls.n, sls.p*sls.T))]]
            _xw = [ [sls._mkvar((sls.n, sls.n*i)),
                     np.zeros((sls.n, sls.n*(sls.T - i)))]
                   for i in range(1, sls.T)] + \
                   [[sls._mkvar((sls.n, sls.n*sls.T))]]
                           
            _uv = [ [sls._mkvar((sls.m, sls.p*i)),
                     np.zeros((sls.m, sls.p*(sls.T - i)))]
                   for i in range(1, sls.T)] + \
                   [[sls._mkvar((sls.m, sls.p*sls.T))]]
            _uw = [ [sls._mkvar((sls.m, sls.n*i)),
                     np.zeros((sls.m, sls.n*(sls.T - i)))]
                   for i in range(1, sls.T)] + \
                   [[sls._mkvar((sls.m, sls.n*sls.T))]]
        else:
            _xv = [[sls._mkvar(sls.C.T.shape)]]
            _xw = [[sls._mkvar(sls.A.shape)]]
            _uv = [[sls._mkvar((sls.m*sls.T, sls.p*sls.T))]]
            _uw = [[sls._mkvar(sls.B.T.shape)]]
        
        # Build the block matrix
        if sls.m == 0 and sls.p == 0:
            raise IndexError("Why use this package without" \
                              + " input or ouput?")
        elif sls.m == 0:
            _phi = cp.bmat([[cp.bmat(_xw), cp.bmat(_xv)]])
        elif sls.p == 0:
            _phi = cp.bmat([[cp.bmat(_xw)], [cp.bmat(_uw)]])
        else:
            _phi = cp.bmat([[cp.bmat(_xw), cp.bmat(_uw)],
                            [cp.bmat(_xw), cp.bmat(_uw)]])

        return _phi
    
    
    def _achievability(phi):
    #############################################################
    #   Builds the achievability constraints on the Phi matrices.
    #
    #   :param phi: full closed loop map
    #   :return: Optimization variables for cvxpy
    #############################################################
    
        # Handy matrices
        oI = np.block([[sls.In], [np.zeros_like(sls.B.T)]])
        iO = np.block([[sls.In, np.zeros_like(sls.C.T)]])
        
        # make constraint
        cons = []
    
        # If the system has an input
        if sls.m > 0:
            cons += [np.block([[sls.In - sls.Z @ sls.A,
                                -sls.Z @ sls.B]]) @ phi == iO]
                                    
        # If the system has an output
        if sls.p > 0:
            cons += [phi @ np.block([[sls.In - sls.Z @ sls.A],
                                     [sls.C @ sls.Z]]) == oI]
                   
        return cons
            
            
    @staticmethod
    def _opt_regret(phi, phi_c):
    #############################################################
    #   Builds the semi-definite programming problem for regret
    #
    #   :param phi: full closed loop map (optimization variable)
    #   :param phi_c: full closed loop map of the clairvoyant
    #   :return: objective and constraint for optimization
    #############################################################
    
        # Define useful variables
        i1, i2 = np.eye(phi.shape[0]), np.eye(phi.shape[1])
        J_c, eig = phi_c.T @ sls.S @ phi_c, cp.Variable()
        
        # Full SDP constraint matrix
        P = cp.bmat([[i1, sls.Ss2 @ phi],
                     [phi.T @ sls.Ss2, eig * i2 + J_c]])
        
        # Return new cost and constraint
        return eig, [P >> 0]
                    
            
    @staticmethod
    def _opt_immitation(phi, phi_c):
    #############################################################
    #   Builds the semi-definite programming problem for regret
    #
    #   :param phi: full closed loop map (optimization variable)
    #   :param phi_c: full closed loop map of the clairvoyant
    #   :return: objective and constraint for optimization
    #############################################################
    
        # Define useful variables
        i1, i2 = np.eye(phi.shape[0]), np.eye(phi.shape[1])
        eig = cp.Variable()
        
        # Full SDP constraint matrix
        P = cp.bmat([[i1, sls.Ss2 @ (phi - phi_c)],
                     [(phi - phi_c).T @ sls.Ss2, eig * i2]])
        
        # Return new cost and constraint
        return eig, [P >> 0]
