from src.conf import np
from tqdm import tqdm
import src.conf as conf
from scipy.linalg import sqrtm, issymmetric

class sls:
#############################################################
#   SYSTEM LEVEL SYNTHESIS OBSERVER DESIGN
#############################################################
    T, I, Z, A, C, Q, verbose = [0] + [None] * 6
    m, n, p, Im, In, Ip = [0] * 3 + [None] * 3

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
        
        # System matrices
        sls.A = (0*sls.In).copy()
        sls.B = (0*sls.In[:, :sls.m*sls.T]).copy()
        sls.C = (0*sls.In[:sls.p*sls.T, :]).copy()
        for i in range(sls.T):
            sls.A[i*sls.n:(i+1)*sls.n, i*sls.n:(i+1)*sls.n], \
            sls.B[i*sls.n:(i+1)*sls.n, i*sls.m:(i+1)*sls.m], \
            sls.C[i*sls.p:(i+1)*sls.p, i*sls.n:(i+1)*sls.n] = \
                sys.lin(t[i], z[i, :])
                            
        # Error cost
        sls.Q = cost.copy()
        if sls.Q is not None and not issymmetric(sls.Q):
            raise ValueError("Weight matrix must be symmetic!")
        sls.Qsqrt = 1 if cost is None else sqrtm(sls.Q)
        sls.Qinv = np.linalg.inv(sls.Qsqrt)
        
        # Weighted least squares solver
        sls._solver = lambda x : \
            np.linalg.lstsq(x[0], x[1], rcond=None)[0]
        
    @staticmethod
    def min(objective='causal', _solver=None):
    #############################################################
    #   Build the error maps minimizing the h2 criterion.
    #   Use sls.set before to initialize the system!
    #
    #   :param objective: string parameter:
    #       - causal for causal h2 observer
    #       - noncausal for clairvoyant observer
    #       - regret for minimal regret observer
    #   :param _solver: (optional) solver function for least
    #       squares, default is sp.linalg.lstsq
    #       (with A and b packed in a tuple)
    #   :return: distrubance and noise to error sls maps
    #############################################################
    
        if sls.I is None:
            raise ValueError("sls object is not set with a system")
                
        # Make a progress bar if verbose is on
        pb = tqdm if sls.verbose else lambda x: x

        # Deal with optional parameter
        _solver = sls._solver if _solver is None else _solver
    
        # Make transform from Phi_v to Phi_w
        iza = np.linalg.inv(sls.In - sls.Z @ sls.A)
        _v, _w = (0*sls.C.T).copy(), (0*sls.A).copy()
        
        # Build noncausal noise and disturbance to error sls maps
        if objective == 'noncausal' or objective == 'regret':
            # Solve unconsrained h2 problem (unvectorized)
            _v = sls.Qinv @ _solver(sls._lstsq_mats(iza)).T
            _w = (sls.In - _v @ sls.C @ sls.Z) @ iza
            
            if objective == 'noncausal':
                return _w, _v
        
        # Build causal noise and disturbance to error sls maps
        if objective == 'causal' or objective == 'regret':
            # vech(Phi_v) multiplier and bias
            _a, _b = sls._lstsq_mats(iza, _w, _v)
            
            # Solve multiple regression with params set to 0
            for i in pb(range(sls.T)):
                # Solve expectation maximization problem
                sol = _solver((_a[:, :(i+1)*sls.p],
                               _b[:, i*sls.n:(i+1)*sls.n]))
                
                # Save in corresponding columns of _v
                _v[i*sls.n:(i+1)*sls.n, :(i+1)*sls.p] = sol.T
                _v[i*sls.n:(i+1)*sls.n, (i+1)*sls.p:] = 0
            
            _v = sls.Qinv @ _v
            return (sls.In - _v @ sls.C @ sls.Z) @ iza, _v
        
        # Wrong objective passed
        raise ValueError("valid objectives are: 'causal'," + \
            "'noncausal', and 'regret'.")
        
    @staticmethod
    def _lstsq_mats(iza, nc_w=None, nc_v=None):
    #############################################################
    #   Build the data matrix and observation vector for observer
    #   design.
    #
    #   :param iza: matrix inverse of (I - ZA)
    #   :param nc_w: (optional) disturbance to error sls map of
    #                the clairvoyant for regret
    #   :param nc_v: (optional) noise to error sls map of the
    #                clairvoyant for regret
    #   :return: distrubance and noise to error sls maps
    #############################################################
    
        # Check optional parameters
        nc_v = 0*sls.C if nc_v is None else nc_v.T
        nc_w = 0*sls.A if nc_w is None else nc_w.T
        
        # Build matrices for lstsq problem maximizing expectation
        return np.vstack((sls.Ip, iza.T @ sls.Z.T @ sls.C.T)), \
               np.vstack((0*sls.C + nc_v, iza.T + nc_w)) @ sls.Qsqrt
        
    
class dro(sls):
#############################################################
#   DISTRIBUTIONALLY ROBUST OPTIMIZATION SOLVER
#############################################################
    # Use the sls attributes plus the wasserstein radius
    eps = 0
        
    @staticmethod
    def min(objective='causal'):
    #############################################################
    #   Build the error maps minimizing the h2 criterion.
    #   Based on the parameters contained in the sls class.
    #   Use sls.set before to initialize the system!
    #
    #   :param objective: string parameter:
    #       - causal for causal h2 observer
    #       - noncausal for clairvoyant observer
    #       - regret for minimal regret observer
    #   :return: distrubance and noise to error sls maps
    #############################################################
        
        # Transform eps into regularizer for iza deviations
        d = np.sqrt(dro.eps)
        
        # Quick access to necessary shapes
        ss = lambda x : (x[0].shape[1],
                         (x[0].shape[1], x[1].shape[1]))
        
        # Regularized solver
        _solver = lambda x : \
            np.linalg.lstsq(np.vstack((x[0], d*np.eye(ss(x)[0]))),
                            np.vstack((x[1], np.zeros(ss(x)[1]))),
                            rcond=None)[0]
            
        return sls.min(objective, _solver)
