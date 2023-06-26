function [Phi_x, Phi_u, objective] = h2_constrained(sys, sls, opt)
%h2_constrained computes a constrained causal linear control policy
%that is optimal in the h2 sense

    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.n*(opt.T + 1), sys.n*(opt.T + 1), 'full');
    Phi_u = sdpvar(sys.m*(opt.T),     sys.n*(opt.T + 1), 'full');
    % As the system initial condition is known, we start by
    % separating the block-columns associated with the known initial
    % state and with the unknown disturbance realizations
    Phi_0 = [Phi_x(:, 1:sys.n); Phi_u(:, 1:sys.n)]; 
    Phi_w = [Phi_x(:, (sys.n + 1):end); Phi_u(:, (sys.n + 1):end)];
    
    Y = sdpvar(size(sls.Hw, 1), size(sls.H, 1), 'full'); % Define the dual variables
    
    constraints = [];
    
    % We want to compute the squared Frobenius norm of z
    z = sqrtm(opt.Cf_h2)*Phi_w; 
    % Recall that (||z||_F)^2 = trace(z*z') = sum(diag(z*z')) = sum(z_ij^2)
    t = sdpvar(1, 1, 'full');
    objective = norm(sqrtm(opt.Cf_h2)*Phi_0*sys.currentState, 2)^2 + t^2;
    constraints = [constraints, cone(z(:), t)];        

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
    
    objective = value(objective); % Extract the optimal average control cost incurred by a constrained causal linear controller
end