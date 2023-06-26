function [gamma] = infinite_horizon_gamma_bisection_search(sys, opt, tol)
%gamma_bisection_search computes the minimum achievable H-infinity 
%performance level for an infinite-horizon objective
    
    lb_gamma = 0; % Lower bound on the performance level
    ub_gamma = 2; % Candidate upper bound to start the bisection algorithm
    
    % Adjust the upper bound on the H-infinity performance level if necessary
    done = 0; 
    while ~done
        [~, ~, ~, info] = idare(sys.A, [sys.B eye(sys.n)], opt.Qt, blkdiag(opt.Rt, -ub_gamma^2*eye(sys.n)), [], []);
        if info.Report == 0
            done = 1;
        else % Candidate upper bound not high enough
            lb_gamma = ub_gamma;
            ub_gamma = 2*ub_gamma;
        end
    end
    % Find the optimal H-infinity performance level using a bisection algorithm
    while (ub_gamma - lb_gamma > tol)
        gamma = (ub_gamma + lb_gamma)/2; % Take the mean between upper and lower bound
        [~, ~, ~, info] = idare(sys.A, [sys.B eye(sys.n)], opt.Qt, blkdiag(opt.Rt, -gamma^2*eye(sys.n)), [], []);
        if info.Report == 0 % Problem feasible: a lower upper bound is found
            ub_gamma = gamma;
        else % Problem unfeasible: a higher lower bound is found
            lb_gamma = gamma; 
        end
    end
    gamma = ub_gamma;
end