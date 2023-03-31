import sys as system
system.path.insert(0, system.path[0] + '/..')
import src.conf as conf
np = conf.np
from src.utils import plot_numpy_dict, eval
from src.models import vdp
from src.models.sim import simulate_system
from src.sls import sls, dro
from tqdm import tqdm

if __name__ == '__main__':
    np.random.seed(123)
    np.set_printoptions(precision=2, linewidth=10000)
    pprint, pbar = print, tqdm
    
    # Set system params
    vdp.prm = 10, 9
    
    # Variables
    nmag, sfreq = 0.05, 314
    Ntrain, Ntest = 20, 50
    tH, tsls = 500, 10
    x0 = (2.0, 10.0)
    
    # Create result matrices
    results = {"t": np.linspace(0, (tsls+tH+1)*conf.ts, tsls+tH+1),
               "h2": np.zeros((tH+1, tsls*vdp.n, Ntest)),
               "dr": np.zeros((tH+1, tsls*vdp.n, Ntest)),
               "gt": np.zeros(((tH+tsls+1)*vdp.n, Ntest)),
               "ekf": np.zeros(((tH+tsls+1)*vdp.n, Ntest)),
               "obs": np.zeros(((tH+tsls+1)*vdp.n, Ntest))}

    # Make disturbance patterns
    v = nmag*np.random.uniform(low=-1.0, high=1.0,
                               size=(vdp.p*(tsls+tH), Ntrain+Ntest))
    w = nmag*np.random.uniform(low=-1.0, high=1.0,
                               size=(vdp.n*(tsls+tH), Ntrain+Ntest))
    
    if True:
        v -= nmag*np.kron(np.sin(sfreq*results["t"][:-1, None]),
                          np.ones((1, Ntrain+Ntest)))
        w += nmag*np.kron(np.sin(sfreq*results["t"][:-1, None]),
                          np.ones((2, Ntrain+Ntest)))
    else:
        v = nmag * (v > 0.5*nmag) * \
            np.random.normal(-1.0, 0.25,
                             size=(vdp.p*(tsls+tH), Ntrain+Ntest)) \
            + nmag * (v <= 0.5*nmag) * \
            np.random.normal(2.0, 0.25,
                             size=(vdp.p*(tsls+tH), Ntrain+Ntest))
        w = nmag * (w > 0.5*nmag) * 0.5 * \
            np.random.normal(1.0, 0.25,
                             size=(vdp.n*(tsls+tH), Ntrain+Ntest)) \
            + nmag * (w <= 0.5*nmag) * 0.5 * \
            np.random.normal(-1.0, 0.25,
                             size=(vdp.n*(tsls+tH), Ntrain+Ntest))
            
    
    # System trajectory
    pprint("Computing ground truth trajectories...")
    for i in pbar(range(Ntest)):
        results["gt"][0, :], results["gt"][1, :] = x0
        for t in range(tH+tsls):
            _xi = results["gt"][t*vdp.n:(t+1)*vdp.n, i]
            sol = simulate_system([0, conf.ts], _xi, vdp)
            results["gt"][(t+1)*vdp.n:(t+2)*vdp.n, i] = \
                sol.x[:, 1] + w[t*vdp.n:(t+1)*vdp.n, i]
    pprint("Done!")
                
    # Non horizon filters
    pprint("Applying non-horizon filters...")
    Q, R = nmag*np.eye(vdp.n), nmag*np.eye(vdp.p)
    results["ekf"][:vdp.n, :] = results["gt"][:vdp.n, :]
        
    for i in pbar(range(Ntest)):
        P = Q.copy()
        for t in range(tH+tsls):
            xt = results["ekf"][t*vdp.n:(t+1)*vdp.n, i].copy()
            Ft, _, Ht  = vdp.lin(results["t"][t], xt)
            
            K = P @ Ht.T @ np.linalg.inv(Ht @ P @ Ht.T + R)
            yt = vdp.obs(results["t"][t],
                         results["gt"][t*vdp.n:(t+1)*vdp.n, i]) \
                - vdp.obs(results["t"][t], xt) \
                + v[t*vdp.p:(t+1)*vdp.p, i] * (t > tsls)
            xt = xt + K @ yt
            P = (np.eye(vdp.n) - K @ Ht) @ P
            
            sol = simulate_system([0, conf.ts], xt, vdp)
            results["ekf"][(t+1)*vdp.n:(t+2)*vdp.n, i] = \
                sol.x[:, 1].copy()
            P = Ft @ P @ Ft.T + Q
    pprint("Done!")
            
    # Prefill results with perfect trajectory
    pprint("Applying moving horizon filters...")
    for t in (range(tH+1)):
        results["h2"][t, :, :] = results["gt"]\
            [(t+1)*vdp.n:(t+tsls+1)*vdp.n, :]
        results["dr"][t, :, :] = results["gt"]\
            [(t+1)*vdp.n:(t+tsls+1)*vdp.n, :]
    
    # Add estimation error
    for t in pbar(range(tH)):
        # Slide window over disturbances
        vt = np.vstack((np.zeros((vdp.p, Ntest+Ntrain)),
                        v[(t+2)*vdp.p:(t+tsls+1)*vdp.p, :]))
        wt = -np.vstack((np.zeros((vdp.n, Ntest+Ntrain)),
                         w[(t+2)*vdp.n:(t+tsls+1)*vdp.n, :]))
                             
        for i in (range(Ntest)):
            # Get current estimate
            xh2 = results["h2"][t, :, i]
            xdr = results["dr"][t, :, i]
            
            # Set SLS with new time window
            sls.set(vdp, results["t"][t:t+tsls],
                    np.reshape(xh2, (vdp.n, tsls), "F"),
                    cost=np.diag([nmag]*2*tsls), axis=1)

            # Design H2 observer
            pwh2, pvh2 = sls.min(1)
            
            # Reset SLS for DR
            sls.set(vdp, results["t"][t:t+tsls],
                    np.reshape(xdr, (vdp.n, tsls), "F"),
                    cost=np.diag([nmag]*2*tsls), axis=1)

            # Add sampled disturbance
            sls.train(vt[:, Ntest:], wt[:, Ntest:])
                                 
            # Design DR observer
            dro.eps_v, dro.eps_w = [nmag] * 2
            pwdr, pvdr = dro.min(1, sls._lstsq)
            
            # Recover initial error
            xgt = results["gt"][(t+1)*sls.n:(t+2)*sls.n, i]
            e0h2 = np.concatenate((xh2[:sls.n] - xgt,
                                   np.zeros(sls.n*(sls.T-1))))
            e0dr = np.concatenate((xdr[:sls.n] - xgt,
                                   np.zeros(sls.n*(sls.T-1))))
            
            # Add error to results
            results["h2"][t+1, :, i] += \
                pwh2 @ (wt[:, i] + e0h2) + pvh2 @ vt[:, i]
            results["dr"][t+1, :, i] += \
                pwdr @ (wt[:, i] + e0dr) + pvdr @ vt[:, i]
                
    pprint("Done!")
   
    # Average phase plot
    phase_h2 = {"x": np.mean(results["h2"][1:, -2, :], axis=1),
                "h2": np.mean(results["h2"][1:, -1, :], axis=1)}
    phase_dr = {"x": np.mean(results["dr"][1:, -2, :], axis=1),
                "dr": np.mean(results["dr"][1:, -1, :], axis=1)}
    phase_kf = {"x": np.mean(results["ekf"][-2*tH::2, :], axis=1),
                "ekf": np.mean(results["ekf"][-2*tH+1::2, :], axis=1)}
    phase_gt = {"x": np.mean(results["gt"][-2*tH::2, :], axis=1),
                "gt": np.mean(results["gt"][-2*tH+1::2, :], axis=1)}
            
    plot_numpy_dict(phase_h2, y_label="h2", save=False)
    plot_numpy_dict(phase_dr, y_label="dr", save=False)
    plot_numpy_dict(phase_kf, y_label="ekf", save=False)
    plot_numpy_dict(phase_gt, y_label="gt")
    
    # Error plot
    err_traj = {"h2": np.abs(results["h2"][1:, -2, :]
                             - results["gt"][-2*tH::2, :])
                + np.abs(results["h2"][1:, -1, :]
                         - results["gt"][-2*tH+1::2, :]),
                "dr": np.abs(results["dr"][1:, -2, :]
                             - results["gt"][-2*tH::2, :])
                + np.abs(results["dr"][1:, -1, :]
                         - results["gt"][-2*tH+1::2, :]),
                "ekf": np.abs(results["ekf"][-2*tH::2, :]
                              - results["gt"][-2*tH::2, :])
                + np.abs(results["ekf"][-2*tH+1::2, :]
                         - results["gt"][-2*tH+1::2, :])}
    print(err_traj["h2"].shape)
    for key, val in err_traj.items():
        err_traj[key] = np.mean(val, axis=1)
    print(err_traj["h2"].shape)
    
    err_traj["t"] = results["t"][-tH:]
            
    plot_numpy_dict(err_traj, x_label="t")

    # Noise training samples plot
    noise = {"t": np.linspace(0, (tsls+tH)*conf.ts, tsls+tH)}
    for i in range(Ntrain):
        noise[str(i)] = v[:, i]
    plot_numpy_dict(noise, "t", "noise", plot="scatter")
    
    # Summary statistics of error
    
    # Wait for user to read the plots
    input("Simulation finished, press any key to end.")
