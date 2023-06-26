function [Phi_x, Phi_u, objective] = regret_constrained(sys, sls, opt, Phi_benchmark)
%regret_constrained computes a finite-horizon regret-optimal constrained 
%causal linear control policy with respect to the given benchmark

    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.n*(opt.T + 1), sys.n*(opt.T + 1), 'full');
    Phi_u = sdpvar(sys.m*(opt.T),     sys.n*(opt.T + 1), 'full');
    % As the system initial condition is known, we start by
    % separating the block-columns associated with the known initial
    % state and with the unknown disturbance realizations
    Phi_0 = [Phi_x(:, 1:sys.n); Phi_u(:, 1:sys.n)]; 
    Phi_w = [Phi_x(:, (sys.n + 1):end); Phi_u(:, (sys.n + 1):end)];
    
    Y = sdpvar(size(sls.Hw, 1), size(sls.H, 1), 'full'); % Define the dual variables
    lambda = sdpvar(1, 1, 'full'); 
    gamma  = sdpvar(1, 1, 'full');
    objective = gamma;
    
    constraints = [];
    
    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I]; 
    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.T-1
        for j = i+1:opt.T % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.n):((i+1)*sys.n), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.n, sys.n)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.m, sys.n)];
        end
    end
    % Impose the polytopic safety constraints
    offset = sls.H*Phi_0*sys.currentState;
    for i = 1:size(sls.H, 1)
        constraints = [constraints, Y(:, i)'*sls.hw <= sls.h(i) - offset(i)];
        constraints = [constraints, Y(:, i) >= 0];
    end
    constraints = [constraints, sls.H*Phi_w == Y'*sls.Hw];
    
    % Compute the matrix that defines the quadratic form measuring the cost incurred by the benchmark controller
    J_benchmark = [Phi_benchmark.x; Phi_benchmark.u]'*opt.C*[Phi_benchmark.x; Phi_benchmark.u];    
    % Impose the constraints deriving from the Schur complement
    P = [sys.currentState'*J_benchmark(1:sys.n, 1:sys.n)*sys.currentState + gamma - lambda (J_benchmark(sys.n+1:end, 1:sys.n)*sys.currentState)' sys.currentState'*Phi_0'*sqrtm(opt.Cf_hinf);
        J_benchmark(sys.n+1:end, 1:sys.n)*sys.currentState lambda*eye(sys.n*opt.T) + J_benchmark(sys.n+1:end, sys.n+1:end) Phi_w'*sqrtm(opt.Cf_hinf);
        sqrtm(opt.Cf_hinf)*Phi_0*sys.currentState sqrtm(opt.Cf_hinf)*Phi_w eye(sys.n*(opt.T + 1) + sys.m*opt.T)]; 
    constraints = [constraints, P >= 0];
    constraints = [constraints, lambda >= 0];

    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a regret-optimal
    % constrained causal linear control policy with respect to the given benchmark
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    objective = value(objective);
    
end