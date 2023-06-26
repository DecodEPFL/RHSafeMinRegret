function [Phi_x, Phi_u, objective] = hinf_constrained(sys, sls, opt)
%hinf_constrained computes a constrained causal linear control policy
%that is optimal in the h-infinity sense

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
    
    % Impose the constraints deriving from the Schur
    P = [gamma - lambda zeros(1, sys.n*opt.T) sys.currentState'*Phi_0'*sqrtm(opt.Cf_hinf);
        zeros(sys.n*opt.T, 1) lambda*eye(sys.n*opt.T) Phi_w'*sqrtm(opt.Cf_hinf);
        sqrtm(opt.Cf_hinf)*Phi_0*sys.currentState sqrtm(opt.Cf_hinf)*Phi_w eye(sys.n*(opt.T + 1) + sys.m*opt.T)];

    constraints = [constraints, P >= 0];
    constraints = [constraints, lambda >= 0];   
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a constrained causal 
    % linear controller that is optimal in the h-infinity sense
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    objective = value(objective);
    
end