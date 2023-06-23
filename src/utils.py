import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tikzplotlib import save as tikz_save
from scipy.sparse.linalg import eigs
from tqdm import tqdm

matplotlib.rcParams['lines.markersize'] = 2
plt.ion()

def plot_numpy_dict(data, x_label="x", y_label="y",
                    name="plot", save=True, plot="plot"):
    """
    Plots a dictionary of numpy arrays and exports the plot as tikz
    
    :param data: Dictionary of numpy arrays
    :param x_label: Label for x-axis and key for x values
    :param y_label: Label for y-axis
    :param name: Title and filename
    :param save: save plot as tikz
    :param plot: plot function member of matplotlib.pyplot.
    :return: None
    """
    pltf = getattr(plt, plot)
    
    # Generate relative errors
    err, mine = {}, np.inf
    for key, value in data.items():
        if key is not x_label:
            err[key] = np.linalg.norm(value, 2)
            mine = mine if err[key] >= mine else err[key]
    for key, value in err.items():
        err[key] = ": best" if value == mine else \
            ": " + "{:.2f}".format(value / mine * 100 - 100) + "%"
    
    # Plot all elements except x value
    for key, value in data.items():
        if key is not x_label and x_label is not None:
            pltf(data[x_label], value, label=key + err[key],
                 linewidth=1)
        elif x_label is None:
            pltf(value, label=key, linewidth=1)
    plt.ylabel(y_label)
    
    # Set x axis if given
    if x_label is not None:
        plt.xlabel(x_label)
        plt.xticks(data[x_label])
        
    # Add description
    plt.title(name)
    plt.legend(loc='best')
    
    # Plot and save
    if save:
        tikz_save(name + ".tikz") #, figurewidth='\\figurewidth')
        plt.show()
        plt.figure()


def eval(phi, t, disturbances, n=1, mask=None, average=True):
    """
    Evaluate the closed loop map phi with a given noise.
    Give Identity closed loop map to obtains noise patterns.
    
    :param phi: Closed loop map to evaluate.
    :param t: length of the time horizon in phi.
    :param disturbances: list or single disturbance pattern
        This can be names or directly the value for custom ones.
    :param n: number of realizations to average for random
        disturbance patterns.
    :param mask: set elements to 0 to ignore the corresponding
        element in the disturbance. Pay attention to shape with
        custom disturbances.
    :param average: return the average error or the error at
        each time step in a matrix.
    :return: dict with errors for each disturbance.
    """

    if type(disturbances) is str:
        disturbances = [disturbances]
    elif type(disturbances) is np.ndarray:
        if disturbances.shape[0] == phi.shape[1]:
            return {"custom" : phi @
                    (disturbances * (mask if mask is not None else 1))}
        else:
            raise ValueError("Custom disturbance has a wrong shape.")
    elif type(disturbances) is not list:
        raise TypeError("please specify disturbances as str or "
                        + "list of strs.")
    
    w = dict()
    
    for d in disturbances:
        if type(d) is np.ndarray and d.shape[0] == phi.shape[1]:
            w["custom"] = d
        elif "gaussian" in d:  # Gaussian: N(0, 1)
            f = float(d.replace("gaussian", "")) \
                if d != "gaussian" else 1
            w[d] = np.random.normal(0, f, (phi.shape[1], n))
        elif "uniform" in d:  # Uniform: U(f, 1)
            f = float(d.replace("uniform", "")) \
                if d != "uniform" else 1
            w[d] = (np.random.rand(phi.shape[1], n)*f + 1.0 - f)
        elif "constant" in d:  # Constant at 1
            f = float(d.replace("constant", "")) \
                if d != "constant" else 1
            w[d] = f*np.ones((phi.shape[1], 1))
        elif "sine" in d:  # Sinusoid
            f = int(d.replace("sine", "")) if d != "sine" else 1
            w[d] = np.repeat(np.sin(np.linspace(0, 2*f*np.pi, t)),
                             phi.shape[1]/t)[:, None]
        elif "sawtooth" in d:  # Sawtooth
            f = int(d.replace("sawtooth", "")) \
                if d != "sawtooth" else 1
            w[d] = np.repeat(np.linspace(0, f - 1e-10, t) % 1,
                             phi.shape[1]/t)[:, None]
        elif "step" in d:  # Step function
            w[d] = np.repeat(np.hstack((np.zeros(int(np.floor(t/2))),
                                        np.ones(int(np.ceil(t/2))))),
                             phi.shape[1]/t)[:, None]
        elif "stairs" in d:  # Sawtooth
            f = int(d.replace("stairs", "")) if d != "stairs" else 1
            w[d] = np.repeat(np.floor(np.linspace(0, 3*f - 3/t, t)%3)
                             - 1, phi.shape[1]/t)[:, None]
        elif d == "worst":  # Worse case
            _, w[d] = eigs(phi.T @ phi, 1)
        else:  # Error
            raise KeyError(d + " is not a valid disturbance type.")
    
    # Evaluate the map with all elements of w
    for d in w.keys():
        w[d] = phi @ (w[d] * (mask if mask is not None else 1))
        w[d] = np.mean(w[d], axis=1) if average else w[d]
    
    return w
    
    
def eval_infty(phi, sys, t, disturbances, n=1,
               mask=None, average=True, verbose=True):
    """
    Evaluate the closed loop map phi with a given noise.
    Give Identity closed loop map to obtains noise patterns.
    
    :param phi: Closed loop map to evaluate. Pass an identity
        of size sys.n+sys.p to obtain just the noise profiles.
    :param t: tuple containing two elements:
        - the length of the time horizon in phi,
        - the length of the simulation horizon.
    :param disturbances: list or single disturbance pattern
        This must be an str or list of str containing the
        name of the distribution generating the noise.
    :param n: number of realizations to average for random
        disturbance patterns.
    :param mask: set elements to 0 to ignore the corresponding
        element in the disturbance. Size should be sys.n+sys.p.
    :param average: return the average error or the error at
        each time step in a matrix.
    :param verbose: display messages and progress bar.
    :return: dict with errors for each disturbance.
    """
    pprint = print if verbose else (lambda x: x)
    pbar = tqdm if verbose else (lambda x: x)
    
    # If one just wants the noise profile, just send a phi = I
    just_profile = False
    
    # Check arguments
    if type(disturbances) is str:
        disturbances = [disturbances]
    elif type(disturbances) is not list:
        raise TypeError("please specify disturbances as str or "
                        + "list of strs.")
    if phi == np.eye(sys.n+sys.p):
        just_profile = True
    elif phi.shape[0] != sys.n+sys.m:
        raise ValueError("Closed loop map dimension 0 does not \
                          correspond to given system.")
    T = int(phi.shape[1]/(sys.n+sys.p))
    if T != phi.shape[1]/(sys.n+sys.p):
        raise ValueError("Closed loop map dimension 1 does not \
                          correspond to given horizon and system.")
    
    # get LTI system parameters
    _a, _b, _c = sys.lin(0, np.zeros(sys.n))
    npp = sys.n+sys.p
    
    # Generate realizations
    w = dict()
    pprint("Starting the generation of random noise.")
    for d in disturbances:
        pprint(d)
        if "gaussian" in d:  # Gaussian: N(0, 1)
            f = float(d.replace("gaussian", "")) \
                if d != "gaussian" else 1
            w[d] = np.random.normal(0, f, (npp*t, n))
        elif "uniform" in d:  # Uniform: U(f, 1)
            f = float(d.replace("uniform", "")) \
                if d != "uniform" else 1
            w[d] = (np.random.rand(npp*t, n)*f + 1.0 - f)
        elif "constant" in d:  # Constant at 1
            f = float(d.replace("constant", "")) \
                if d != "constant" else 1
            w[d] = f*np.ones((npp*t, 1))
        elif "sine" in d:  # Sinusoid
            f = int(d.replace("sine", "")) if d != "sine" else 1
            w[d] = np.repeat(np.sin(np.linspace(0, 2*f*np.pi, t)),
                             npp)[:, None]
        elif "sawtooth" in d:  # Sawtooth
            f = int(d.replace("sawtooth", "")) \
                if d != "sawtooth" else 1
            w[d] = np.repeat(np.linspace(0, f - 1e-10, t) % 1,
                             npp)[:, None]
        elif "step" in d:  # Step function
            w[d] = np.repeat(np.hstack((np.zeros(int(np.floor(t/2))),
                                        np.ones(int(np.ceil(t/2))))),
                             npp)[:, None]
        elif "stairs" in d:  # Sawtooth
            f = int(d.replace("stairs", "")) if d != "stairs" else 1
            w[d] = np.repeat(np.floor(np.linspace(0, 3*f - 3/t, t)%3)
                             - 1, npp)[:, None]
        #elif d == "worst":  # Worse case
        #    _, w[d] = eigs(phi.T @ phi, 1)
        else:  # Error
            raise KeyError(d + " is not a valid disturbance type.")
    pprint("Done!")
    
    if just_profile:
        return w
    
    # Evaluate the map with all elements of w
    pprint("Computing the state and output trajectories...")
    for d in w.keys():
        pprint(d)
        nd = w[d].shape[1]
        
        # Zero initial states, might be suboptimal
        e = np.zeros((sys.n*T, nd))
        x = np.zeros((sys.n, nd))
        xs = x
        
        y = np.zeros((sys.p*T, nd)) if sys.p > 0 else e
            
        xs, ys = x, y[-sys.p:, :] if sys.p > 0 else x
        for k in pbar(range(t)):
            # Extract noise from traj
            ek = w[d][npp*k:npp*(k+1), :] * \
                (mask if mask is not None else 1)
            vk, wk = ek[sys.n:, :], ek[:sys.n, :]
            
            if sys.p > 0:
                # Get output at k
                y = np.roll(y, -sys.p, axis=0)
                y[-sys.p:, :] = _c @ x + vk
                # Compute control input at k
            
            # Update internal state at k+1
            e = np.pad(e, ((0, sys.n),(0, 0)),
                       mode='constant')[sys.n:, :]
            ey = np.vstack([e, y])  if sys.p > 0 else e
            
            u = (phi[sys.n:, :] @ ey) if sys.m > 0 else 0
            e[-sys.n:, :] = -phi[:sys.n, :] @ ey \
                + (x if sys.p == 0 else 0)
            
            # Get state at k+1
            x = _a @ x + wk + (_b @ u if sys.m > 0 else 0)
            
            # append result
            xs = np.vstack([xs, x])
            ys = np.vstack([ys, (_c @ x) if sys.p > 0 else x])
        
        w[d] = np.mean(ys, axis=1) if average else ys
    pprint("Done!")
    
    print(phi[:sys.n, :])
    print(phi[sys.n:, :])
    
    return w
