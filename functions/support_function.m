function [h_w] = support_function(W, a)
%support_function solves a linear program to evaluate the support function 
% of a polytopic set
    
    % Define the decision variable of the optimization problem
    w = sdpvar(size(W.A, 2), 1, 'full');
    % Define the objective function
    objective = -a'*w;
    % Impose the polytopic constraints
    constraints = [];
    constraints = [constraints, W.A*w <= W.b];
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'linprog');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    h_w = -value(objective);
end

