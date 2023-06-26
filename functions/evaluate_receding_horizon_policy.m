function [cum_cost] = evaluate_receding_horizon_policy(sys, opt, sls, ctrl, k, flag, Phi_benchmark, w)

    % Extract information about the receding horizon policy
    N = ctrl.N(k);  % Number of times the finite-horizon control problem is solved online
    s = ctrl.s(k);  % Number of control actions applied before optimizing again
    % Compute the state and input cost matrices used to access the
    % performance of the receding horizon control policy
    Q = blkdiag(kron(eye(s*N), opt.Qt), zeros(sys.n, sys.n)); 
    R = kron(eye(s*N), opt.Rt);
    % Allocate memory for the closed-loop input and state trajectories
    cl_traj.u = zeros(sys.m, s*N); 
    cl_traj.x = zeros(sys.n, s*N + 1);
    cl_traj.x(:, 1) = sys.x0; % Always set the same initial condition
    for n = 1:N % Simulate the closed-loop system under the receding horizon control law
        sys.currentState = cl_traj.x(:, 1 + (n - 1)*s); % Set the current plant state as the initial condition for the optimization
        switch flag
            case 'H2'
                [Phi.x, Phi.u, ~] = h2_constrained(sys, sls, opt);
            case 'Hinf'
                [Phi.x, Phi.u, ~] = hinf_constrained(sys, sls, opt);
            case 'regret'
                [Phi.x, Phi.u, ~] = regret_constrained(sys, sls, opt, Phi_benchmark);
        end
        clear functions
        if s == 1
            % The controller has access to the current state only for feedback
            fb.delta = sys.currentState;
        else
            % Extract the next (s - 1) disturbance realizations
            fb.w = w(:, (1 + (n - 1)*s):(n*s - 1)); 
            fb.delta = [sys.currentState; fb.w(:)];
        end
        % Compute the first ctrl.s control actions to be applied to the plant
        fb.u = Phi.u(1:s*sys.m, 1:s*sys.n)*fb.delta;
        % Compute the corresponding state evolution
        fb.x = Phi.x((sys.n + 1):((s + 1)*sys.n), 1:(s+1)*sys.n)*[fb.delta; w(:, n*s)];
        % Update the closed-loop system trajectory
        cl_traj.u(:, (1 + (n - 1)*s):(n*s)) = reshape(fb.u, sys.m, s);
        cl_traj.x(:, (1 + (n - 1)*s + 1):(n*s + 1)) = reshape(fb.x, sys.n, s);
    end
    cum_cost = [cl_traj.x(:); cl_traj.u(:)]'*blkdiag(Q, R)*[cl_traj.x(:); cl_traj.u(:)];
end