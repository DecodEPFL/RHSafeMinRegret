function [F_alpha_s] = approximate_mRPI_set(sys, e, flag)
%approximate_mRPI_set computes a robust positively invariant, outer
% e-approximation of the minimal robust positively invariant set
    
    % Implementation of Algorithm 1 in "Raković et al. - 2005 - Invariant 
    % approximations of the minimal robust positively invariant set"
    
    % Choose any s in the set of natural numbers (ideally set s close to 0)
    s = 0;
    
    Mp = zeros(sys.n, 1); % Support function evaluated with positive sign argument (cf. Equation (13))
    Mn = zeros(sys.n, 1); % Support function evaluated with negative sign argument (cf. Equation (13))
    I = eye(sys.n); % Contains standard basis vectors
    
    if strcmp(flag, 'H2')
        A_cl = sys.A + sys.B*sys.Kf_h2;
    elseif strcmp(flag, 'Hinf')
        A_cl = sys.A + sys.B*sys.Kf_hinf;
    else
        error('Something went wrong...');
    end
    
    done = false; % Flag for loop termination
    while ~done
        s = s + 1; % Increment s
        fprintf('Iterating with s = %d...\n', s)
        % Compute alpha_o(s) as in Equation (11)
        alpha = zeros(size(sys.Hw, 1), 1);
        for i = 1:size(sys.Hw, 1)
            alpha(i) = support_function(sys.W, (A_cl^s)'*sys.Hw(i, :)') / sys.hw(i);
        end
        alpha = max(alpha);
        % Compute M(s) as in Equation (13)
        for j = 1:sys.n
            support_function(sys.W,  (A_cl^(s - 1))'*I(:, j));
            Mp(j) = Mp(j) + support_function(sys.W,  (A_cl^(s - 1))'*I(:, j));
            Mn(j) = Mn(j) + support_function(sys.W, -(A_cl^(s - 1))'*I(:, j));
        end
        M = max([Mp; Mn]);
        % Check terminal condition to exit the loop
        if alpha <= e/(e + M)
            done = true;
            fprintf('Computing the robust positively invariant set...');
        end
    end
    % Compute F_s as the Minkowski sum in Equation (2) and scale it as in 
    % Equation (5) to compute F_alpha_s
    F_s = Polyhedron();
    for i = 0:(s - 1)
        W_i = (sys.A^i)*sys.W;
        F_s = F_s + W_i.minHRep;
        F_s = F_s.minHRep;
    end
    F_alpha_s = (1 - alpha)^(-1)*F_s;
    F_alpha_s = F_alpha_s.minHRep;
    fprintf('done!\n\n');

end

