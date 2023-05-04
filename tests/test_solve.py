import sys as system
system.path.insert(0, system.path[0] + '/..')
from tqdm import tqdm
import src.conf as conf
np = conf.np
from scipy.optimize import minimize

if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = -np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
    
    res1 = np.linalg.lstad(A, B)

    x = np.linalg.lstsq(A, B, rcond=None)[0]

    # L1 norm function
    c = lambda p, a, b: np.sum(np.abs(b - a @ p))

    # Solve each regression (each column of _b)
    for i in range(B.shape[1]):
        x[:, i] = minimize(c, x[:, i],
                            args=(A, B[:,i])).x
    
    print(x, res1)
    print(c(x, A, B))
    print(c(res1, A, B))
