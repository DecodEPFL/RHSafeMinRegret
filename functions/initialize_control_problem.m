%% Definition of the underlying discrete-time LTI system
sys.rho = 0.7; % Spectral radius
sys.A = sys.rho*[0.7 0.2 0; 0.3 0.7 -0.1; 0 -0.2 0.8];
sys.B = [1 0.2; 2 0.3; 1.5 0.5];

sys.n = size(sys.A, 1);   % Order of the system: state dimension
sys.m = size(sys.B, 2);   % Number of control input channels
sys.x0 = 5*(2*rand(sys.n, 1) - 1); % Initial condition randomly chosen in [-5, 5]
sys.currentState = sys.x0;

sys.Hu = [eye(sys.m); -eye(sys.m)]; % Polytopic constraints: Hu * u <= hu
sys.hu = 2*ones(size(sys.Hu, 1), 1);
sys.U = Polyhedron(sys.Hu, sys.hu);

sys.Hx = [eye(sys.n); -eye(sys.n)]; % Polytopic constraints: Hx * x <= hx
sys.hx = 3.5*ones(size(sys.Hx, 1), 1);
sys.X = Polyhedron(sys.Hx, sys.hx);

sys.Hw = [eye(sys.n); -eye(sys.n)]; % Polytopic disturbance set: Hw * w <= hw 
sys.hw = 1*ones(size(sys.Hw, 1), 1);
sys.W = Polyhedron(sys.Hw, sys.hw);
%% Definition of the parameters of the optimization problem
opt.Qt = eye(sys.n); % Stage cost: state weight matrix
opt.Rt = eye(sys.m); % Stage cost: input weight matrix

opt.T = 20; % Control horizon
%% Computation of the terminal ingredients and auxiliary control law
sys.epsilon = 0.5; % Minimal robust positively invariant set approximation level
if strcmp(flag, 'Hinf') || strcmp(flag, 'regret')
    sys.tol = 1e-03;   % Tolerance on the performance level mismatch from optimal value
    sys.gamma_f = infinite_horizon_gamma_bisection_search(sys, opt, sys.tol);
    % Solve the sign indefinite discrete-time algebraic Riccati equation
    [opt.P_hinf, ~, ~, ~] = idare(sys.A, [sys.B eye(sys.n)], opt.Qt, blkdiag(opt.Rt, -sys.gamma_f^2*eye(sys.n)), [], []);
    % Compute the corresponding unconstrained H-infinity state-feedback controller: u_t = Kf * x_t
    sys.Kf_hinf = -inv(opt.Rt + sys.B'*(opt.P_hinf + opt.P_hinf*inv(sys.gamma_f^2*eye(sys.n) - opt.P_hinf)*opt.P_hinf)*sys.B)*sys.B'*opt.P_hinf*sys.A;
    sys.Xf_hinf = approximate_mRPI_set(sys, sys.epsilon, 'Hinf'); 
    % Check that the terminal set is also constraint admissible
    if ~(sys.X.contains(sys.Xf_hinf) && sys.U.contains(sys.Kf_hinf*sys.Xf_hinf))
        error('Assumption 3 is not satisfied...')
    end
    sys.Xf = sys.Xf_hinf;
elseif strcmp(flag, 'H2')
    % For comparison, solve the sign definite discrete-time algebraic Riccati equation
    [opt.P_h2, ~, ~, ~] = idare(sys.A, sys.B, opt.Qt, opt.Rt, [], []);
    % For comparison, compute the unconstrained H2 state-feedback controller: u_t = Kf_h2 * x_t
    sys.Kf_h2 = -inv(opt.Rt  + sys.B'*opt.P_h2*sys.B)*sys.B'*opt.P_h2*sys.A;
    sys.Xf_h2 = approximate_mRPI_set(sys, sys.epsilon, 'H2'); 
    % Check that the terminal set is also constraint admissible
    if ~(sys.X.contains(sys.Xf_h2) && sys.U.contains(sys.Kf_h2*sys.Xf_h2))
         error('Assumption 3 is not satisfied...')
    end
    sys.Xf = sys.Xf_h2;
else
    error('Incorrect control problem initialization...')
end
%% Definition of the stacked system dynamics over the control horizon
sls.A = kron(eye(opt.T + 1), sys.A);
sls.B = [kron(eye(opt.T), sys.B); zeros(sys.n, opt.T*sys.m)];
sls.I = eye(sys.n*(opt.T + 1)); 
sls.Z = [zeros(sys.n, sys.n*opt.T) zeros(sys.n, sys.n); eye(sys.n*opt.T) zeros(sys.n*opt.T, sys.n)]; % Block-downshift operator

opt.Q = blkdiag(kron(eye(opt.T), opt.Qt), zeros(sys.n, sys.n));  % State cost matrix: without terminal penalty
opt.R = kron(eye(opt.T), opt.Rt);                                % Input cost matrix
opt.C = blkdiag(opt.Q, opt.R);             % Cost matrix: without   terminal penalty

if strcmp(flag, 'Hinf') || strcmp(flag, 'regret')
    opt.Qf_hinf = blkdiag(kron(eye(opt.T), opt.Qt), opt.P_hinf); % State cost matrix: with Hinf terminal penalty
    opt.Cf_hinf = blkdiag(opt.Qf_hinf, opt.R);                   % Total cost matrix: with Hinf terminal penalty
else
    opt.Qf_h2   = blkdiag(kron(eye(opt.T), opt.Qt), opt.P_h2);  % State cost matrix: with H2 terminal penalty
    opt.Cf_h2   = blkdiag(opt.Qf_h2,   opt.R);                  % Total cost matrix: with H2 terminal penalty
end

% Polytopic disturbance description and safety constraints
sls.Hx = blkdiag(kron(eye(opt.T), sys.Hx), sys.Xf.A);
sls.hx = [kron(ones(opt.T, 1), sys.hx); sys.Xf.b];

sls.Hu = kron(eye(opt.T), sys.Hu);
sls.hu = kron(ones(opt.T, 1), sys.hu);

sls.H = blkdiag(sls.Hx, sls.Hu);
sls.h = [sls.hx; sls.hu];

sls.Hw = kron(eye(opt.T), sys.Hw);
sls.hw = kron(ones(opt.T, 1), sys.hw); % We will assume that the initial condition is known