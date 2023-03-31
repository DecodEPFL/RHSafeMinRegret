import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tikzplotlib import save as tikz_save
from scipy.sparse.linalg import eigs

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


def eval(phi, t, disturbances, n=1, mask=None):
    """
    Evaluate the closed loop map phi with a given disturbance.
    
    :param phi: Closed loop map to evaluate.
    :param t: length of the time horizon in phi.
    :param disturbances: list or single disturbance pattern
        This can be names or directly the value for custom ones.
    :param n: number of realizations to average for random
        disturbance patterns.
    :param mask: set elements to 0 to ignore the corresponding
        element in the disturbance. Pay attention to shape with
        custom disturbances.
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
        elif d == "constant":  # Constant at 1
            w[d] = np.ones((phi.shape[1], 1))
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
        w[d] = np.mean(phi @ (w[d] * (mask if mask is not None
                                      else 1)), axis=1)
    
    return w
