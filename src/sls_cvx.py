from src.conf import np
from tqdm import tqdm
import src.conf as conf
from scipy.linalg import sqrtm
import cvxpy as cp

class sls:
#############################################################
#   SYSTEM LEVEL SYNTHESIS CONTROL DESIGN
#############################################################
    T, I, Z, A, C, S, Ss2, verb = [0] + [None] * 6 + [False]
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
        sls.iw = [True] * (sls.n*sls.T) + [False] * (sls.p*sls.T)
        sls.iv = [False] * (sls.n*sls.T) + [True] * (sls.p*sls.T)
        sls.ix = [True] * (sls.n*sls.T) + [False] * (sls.m*sls.T)
        sls.iu = [False] * (sls.n*sls.T) + [True] * (sls.m*sls.T)
        
        sls.ixw = np.ix_(sls.ix, sls.iw)
        sls.ixv = np.ix_(sls.ix, sls.iv)
        sls.iuw = np.ix_(sls.iu, sls.iw)
        sls.iuv = np.ix_(sls.iu, sls.iv)
                            
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
    #       - causal for causal h2 cost
    #       - noncausal for clairvoyant cost
    #       - regret for minimal regret cost
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
        pb = tqdm if sls.verb else lambda x: x

        # Deal with optional parameters
        _solver = sls._solver if _solver is None else _solver
        if constraints is None:
            constraints = lambda x: []
        
        # Build noncausal or infinite horizon
        # noise and disturbance to error sls maps
        if objective == 'h2 infty' or objective == 'hinf infty':
            # Variables with only the last block-row to optimize
            _phi = sls._opt_variables("infty")
            
            # H2 optimizes fro norm and hinf spectral norm
            obj = cp.norm(sls.Ss2 @ _phi,
                          2 if objective == "hinf infty" else 'fro')
            
            # Achievability constraints
            cons = sls._achievability_infty(_phi) + constraints(_phi)
                                      
            # Solve the problem. Results are directly stored in vars
            cp.Problem(cp.Minimize(obj),
                       cons).solve(solver=_solver, verbose=sls.verb)
                                   
            # Return values
            _pc = _phi.value
            return _pc[:sls.n, sls.iw], _pc[:sls.n, sls.iv], \
                   _pc[sls.n:, sls.iw], _pc[sls.n:, sls.iv]
                   
        elif objective != 'h2' and objective != 'hinf':
            # Full variables for unconstrained noncausal problem
            _phi = sls._opt_variables()
            
            # H2 or Hinf objectives are equivalent without causality
            obj = cp.norm(sls.Ss2 @ _phi, 'fro')
            
            # Achievability constraints
            cons = sls._achievability(_phi) + constraints(_phi)
            
            # Solve the problem. Results are directly stored in vars
            cp.Problem(cp.Minimize(obj),
                       cons).solve(solver=_solver, verbose=sls.verb)
            
            # Only keep the values and return if needed
            _phi = _phi.value
            if objective == 'noncausal':
                return _phi[sls.ixw], _phi[sls.ixv], \
                       _phi[sls.iuw], _phi[sls.iuv]
        
        # If causal result is asked, move on to that problem
                
        # Lower triangular variables for causal problem
        _pc = sls._opt_variables("causal")
        
        # Achievability constraint
        cons = sls._achievability(_pc) + constraints(_pc)
        
        if objective == "h2" or objective == "hinf":
            # H2 optimizes fro norm and hinf spectral norm
            obj = cp.norm(sls.Ss2 @ _pc,
                          2 if objective == "hinf" else 'fro')
                                      
            # Solve the problem.
            cp.Problem(cp.Minimize(obj),
                       cons).solve(solver=_solver, verbose=sls.verb)
                                   
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
                       + sdpc).solve(solver=_solver, verbose=sls.verb)
                       
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
                       + sdpc).solve(solver=_solver, verbose=sls.verb)
                       
            # Return values
            _pc = _pc.value
            return _pc[sls.ixw], _pc[sls.ixv], \
                   _pc[sls.iuw], _pc[sls.iuv]
        
        # Wrong objective passed
        raise ValueError(str(objective) + " is not a valid " \
                         + "objectve. Valid objectives are: " \
                         + "'causal', 'noncausal', and 'regret'.")
                         
                         
    def mkcons(phi, H, h, Hw, hw, repeat=True):
    #############################################################
    #   Builds the constraints H @ [x, u] ≤ h and Hw @ [v, w] ≤ hw
    #   given phi relating [x, u] and [v, w]
    #
    #   The constraints are static (the same at each timestep
    #   To build dynamic constraints, you can directly copy the
    #   last two lines of the function
    #
    #   :param phi: the optimization variable representing sls maps
    #   :param H: state/input constraint multiplier matrix
    #   :param h: state/input constraint comparison vector
    #   :param Hw: noise constraint multiplier matrix
    #   :param hw: noise constraint comparison vector
    #   :param repeat: Determines if the constraints H[x, u] ≤ h
    #       must be repeated for all time steps (True),
    #       or if it is a constraint on the trajectory (False).
    #       For inifite horizon use False.
    #   :return: Optimization variables for cvxpy
    #############################################################
        # State and input constraints
        HT, hT = (np.kron(H, sls.I), np.kron(np.diag(sls.I), h)) \
            if repeat else (H, h)

        # disturbance constraint
        HwT, hwT = np.kron(Hw, sls.I), np.kron(np.diag(sls.I), hw)

        # Dual variable
        Z = cp.Variable((HwT.shape[0], HT.shape[0]))
        
        # List of additional constraints
        return [Z.T @ hwT <= hT, HT @ phi == Z.T @ HwT, Z >= 0]
        
        
    @staticmethod
    def _opt_variables(type="causal"):
    #############################################################
    #   Builds the Phi matrices as optimization variables.
    #
    #   :param type: sets the sparsity pattern. several choice:
    #       - causal: sets the upper block-triangle to zero
    #       - infty: sets everything to zero besides the last row
    #       - default is dense matrix
    #   :return: Optimization variables for cvxpy
    #############################################################
    
        # Make each 4 blocks of the output feedback matrix
        # Some are [] if sls.m or sls.p are zero
        if type == "infty":
            _xv = [ [sls._mkvar((sls.n, sls.p*sls.T))] ]
            _xw = [ [sls._mkvar((sls.n, sls.n*sls.T))] ]
            _uv = [ [sls._mkvar((sls.m, sls.p*sls.T))] ]
            _uw = [ [sls._mkvar((sls.m, sls.n*sls.T))] ]
        elif type == "causal":
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
            _phi = cp.bmat([[cp.bmat(_xw), cp.bmat(_xv)],
                            [cp.bmat(_uw), cp.bmat(_uv)]])

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
            
    def _achievability_infty(phi):
    #############################################################
    #   Builds the achievability constraints on the Phi matrices
    #   in infinite horizon (FIR).
    #
    #   :param phi: full closed loop map
    #   :return: Optimization variables for cvxpy
    #############################################################
    
        # Handy matrices
        _ab = np.hstack((sls.A[:sls.n, :sls.n],
                         sls.B[:sls.n, :sls.m]))
        _ac = np.vstack((sls.A[:sls.n, :sls.n],
                         sls.C[:sls.p, :sls.n]))

        # make constraints
        cons = [phi[:sls.n, sls.n*(sls.T-1):sls.n*sls.T]
                == 0]
        cons += [phi[:sls.n, sls.n*(sls.T-2):sls.n*(sls.T-1)]
                 == np.eye(sls.n)]

        if sls.m > 0:
            cons += [phi[sls.n:, sls.n*(sls.T-1):sls.n*sls.T] == 0]

        if sls.p > 0:
            cons += [phi[:sls.n, -sls.p:] == 0,
                     phi[:sls.n, -2*sls.p:-sls.p]
                     == sls.B[:sls.n, :sls.m] @ phi[sls.n:, -sls.p:],
                     phi[sls.n:, sls.n*(sls.T-2):sls.n*(sls.T-1)]
                     == phi[sls.n:, -sls.p:] @ sls.C[:sls.p, :sls.n]]

        for t in range(sls.T-1):
            # Select phis at t and t-1 /!\ Flipped
            it = list(range((t-1)*sls.n,t*sls.n)) \
                + list(range(sls.T*sls.n + (t-1)*sls.p,
                             sls.T*sls.n + t*sls.p))
            itp1 = list(range(t*sls.n,(t+1)*sls.n)) \
                + list(range(sls.T*sls.n + t*sls.p,
                             sls.T*sls.n + (t+1)*sls.p))
                             
            # If the system has an input
            if sls.m > 0 and t > 0:
                cons += [_ab @ phi[:, itp1] ==
                         phi[:, it][:sls.n, :] * (t > 0)]

            # If the system has an output
            if sls.p > 0 and t > 0:
                cons += [phi[:, itp1] @ _ac ==
                         phi[:, it][:, :sls.n] * (t > 0)]
            if t == 0:
                cons += [phi[:, itp1] @ _ac == 0,
                         _ab @ phi[:, itp1] == 0]
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


class dro(sls):
#############################################################
#   SYSTEM LEVEL SYNTHESIS DISTRIBUTIONALLY ROBUST OPTIMIZATION
#############################################################
    eps = 0.01
    profiles = None
    tmp = None
    
    def train(vw_profile):
    #   :param profiles: samples of the noise to use as empirical
    #       distribution
        dro.profiles = vw_profile
    
    @staticmethod
    def min(objective='causal', _solver=None, constraints=None):
    #############################################################
    #   Build the error maps minimizing the given objective.
    #   Use sls.set before to initialize the system!
    #
    #   :param objective: string parameter:
    #       - causal/noncausal/regret is inherited
    #       - dro for distributionally robust cost
    #   :param _solver: (optional) solver from cvxpy,
    #       default is cp.Mosek
    #   :param constraints: (optional) additional constraints,
    #       must be a function taking the closed loop map (phi)
    #       as input and returning a list of cvxpy constraints
    #   :return: sls maps of distrubance and noise to
    #       state and input errors
    #############################################################
            
        # Check training and setting
        if sls.I is None:
            raise ValueError("sls object is not set with a system")
        
        if dro.profiles is None:
            raise TypeError("empirical distribution must be trained.")
                
        # Make a progress bar if verbose is on
        pb = tqdm if sls.verb else lambda x: x

        # Deal with optional parameters
        _solver = sls._solver if _solver is None else _solver
        if constraints is None:
            constraints = lambda x: []
                    
        # Lower triangular variables for causal problem
        # or block row for infinite horizon
        _pc = sls._opt_variables("causal" if objective == 'dro'
                                 else "infty")
        
        if objective == 'dro':
            # Achievability constraint
            cons = sls._achievability(_pc) + constraints(_pc)
            
            # Make the problem
            obj, sdpc = dro._opt_prob(_pc)
                                      
            # Solve the problem.
            cp.Problem(cp.Minimize(obj), cons
                       + sdpc).solve(solver=_solver, verbose=sls.verb)
                       
            # Return values
            _pc = _pc.value
            return _pc[sls.ixw], _pc[sls.ixv], \
                   _pc[sls.iuw], _pc[sls.iuv]
        
        elif objective == 'dro infty':
            # Achievability constraint
            cons = sls._achievability_infty(_pc) + constraints(_pc)
            
            # Make the problem
            obj, sdpc = dro._opt_prob(_pc)
                                      
            # Solve the problem.
            cp.Problem(cp.Minimize(obj), cons
                       + sdpc).solve(solver=_solver, verbose=sls.verb)
                       
            # Return values
            _pc = _pc.value
            return _pc[:sls.n, sls.iw], _pc[:sls.n, sls.iv], \
                   _pc[sls.n:, sls.iw], _pc[sls.n:, sls.iv]
        
        return sls.min(objective, _solver, constraints)


    @staticmethod
    def mkcons(phi, H, h, p_fail=5e-2, repeat=True):
    #############################################################
    #   Builds the DR constraints H @ [x, u] ≤ h with empirical
    #   distribution ws = [[v1, ..., vN], [w1, ..., wN]] and
    #   given phi relating [x, u] and ws
    #
    #   To set the empitical distribution use dro.train(profile)
    #
    #   The constraints are static (the same at each timestep
    #   To build dynamic constraints, you can directly copy the
    #   last two lines of the function
    #
    #   :param phi: the optimization variable representing sls maps
    #   :param H: state/input constraint multiplier matrix
    #   :param h: state/input constraint comparison vector
    #   :param p_fail: probability level of CVar constraint
    #   :param p_fail: defines whether this is a state or an
    #       input constraint. Default is True for state constaint.
    #   :param repeat: Determines if the constraints H[x, u] ≤ h
    #       must be repeated for all time steps (True),
    #       or if it is a constraint on the trajectory (False).
    #       For inifite horizon use False.
    #   :return: Optimization variables for cvxpy
    #############################################################
        # Check training
        if dro.profiles is None:
            raise TypeError("empirical distribution must be trained.")
    
        # Dimensionality of the constraint
        N = dro.profiles.shape[1]
        Iw = np.eye(dro.profiles.shape[0])
        # Split phix and phiu
        nx = int(phi.shape[0]/(sls.n+sls.m)*sls.n)
        
        # rename handy variables
        y = p_fail
        px, pu = phi[:nx, :], phi[nx:, :]
        vw = dro.profiles
    
        # State and input constraints
        HT, hT = (np.kron(H, sls.I), np.kron(np.diag(sls.I), -h)) \
            if repeat else (H, -h)

        # Dual variables
        lbdx, lbdu = cp.Variable(), cp.Variable()
        taux, tauu = cp.Variable(), cp.Variable()
        sx, su = cp.Variable(N), cp.Variable(N)
                
        # Need a loop here for now, implementing Proposition 5 in
        # "Capture, Propagate, and Control Distributional Uncertainty"
        # from Liviu and Nicolas
        
        # Constraints on dual varialbes (included the one embeded in J+1)
        constraints = [taux <= sx, tauu <= su,
                       lbdx >= 0, lbdu >= 0]
        
        # CVar cost less than 0
        constraints += [lbdx*dro.eps*N + cp.sum(sx) <= 0,
                        lbdu*dro.eps*N + cp.sum(su) <= 0]
        
        # Then SDP for all i and j
        for i in range(N):
            # build j-independent parts
            lm = np.block([[Iw, vw[:, [i]]], [vw[:, [i]].T,
                           np.dot(vw[:, i], vw[:, i])]])
            sc = np.diag(np.hstack([0*np.diag(Iw), 1]))
            
            for j in range(H.shape[0]):
                ajx, aju = HT[[j], :nx].T/(2*y),  HT[[j], nx:].T/(2*y)
                bjx = (hT[j] + y*taux - taux)/y
                bju = (hT[j] + y*tauu - tauu)/y
                
                constraints += \
                    [cp.bmat([[0*Iw, px.T @ ajx], [ajx.T @ px, [[0]]]])
                     + sc * (sx[i] - bjx) + lbdx * lm >> 0,
                     cp.bmat([[0*Iw, pu.T @ aju], [aju.T @ pu, [[0]]]])
                     + sc * (su[i] - bju) + lbdu * lm >> 0]
                    #[bjx + cp.kron(ajx.T, vw[:, [i]].T) @ cp.vec(px.T) <= sx[i],
                    # cp.SOC(lbd, cp.kron(ajx.T, sls.In) @ cp.vec(px.T)]

        return constraints
            
    @staticmethod
    def _opt_prob(phi):
    #############################################################
    #   Builds the semi-definite programming problem for DRO
    #
    #   :param phi: full closed loop map (optimization variable)
    #   :return: objective and constraint for optimization
    #############################################################
        # dimensionality of the constraint
        N = dro.profiles.shape[1]
        Iw = np.eye(dro.profiles.shape[0])
        vw = dro.profiles
        
        # Dual variables
        lbd, s = cp.Variable(), cp.Variable(N)
        cons = [lbd >= 0, s >= 0]
                    
        # Adding variable for Phi.T Phi to stay convex
        _pcsq = cp.Variable((phi.shape[1], phi.shape[1]))
        sdpc = [cp.bmat([[_pcsq, phi.T @ sls.Ss2],
                         [sls.Ss2 @ phi, np.eye(phi.shape[0])]
                         ]) >> 0]

        # SDP constraint
        for i in range(N):
            sdpc += [cp.bmat([[lbd*Iw - _pcsq,
                               lbd*vw[:, [i]]],
                              [lbd*vw[:, [i]].T,
                               s[[[i]]]+lbd*cp.norm(vw[:, [i]])]
                               ]) >> 0]
        
        # Cost
        obj = lbd*dro.eps*dro.eps + cp.sum(s)/N
        
        # Return new cost and constraint
        return obj, cons + sdpc
