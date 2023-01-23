import sys as system
system.path.insert(0, system.path[0] + '/..')
from tqdm import tqdm
import src.conf as conf
np = conf.np
from src.models import foo, vdp, andrea
from src.models.sim import simulate_system
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sol = simulate_system(np.linspace(0, 10, 1000), [2.0, 0.0], vdp)
    plt.plot(sol.x[0], sol.x[1])
    plt.show()
    
    inputs = np.array([0*np.linspace(0, 1.0, 99), np.zeros((99,))]).T
    sol = simulate_system(np.linspace(0, 1.0, 100), [2.0, -1.0, 0.0],
                          andrea, inputs)
    plt.plot(sol.t, sol.x[0])
    plt.show()
    
    #sol = simulate_system(np.linspace(0, 10, 1000), [0, 0, 0.1, 0], pen)
    #plt.plot(sol.t, sol.y[0])
    #plt.plot(sol.t, sol.y[2])
    #plt.show()

