function [w] = get_realization(sys, ctrl, profile)
    freq = 2*pi/ctrl.T;
    switch profile
        case "Gaussian: N(0,1)"
            w = randn(sys.n, ctrl.T);
            w = w/max(abs(w(:))); % Make sure the disturbance lies within the bounds [-1, 1] to ensure recursive feasibility
        case "Uniform: U(0.5, 1)"
            w = 0.5 + 0.5*rand(sys.n, ctrl.T);
        case "Constant"
            w = ones(sys.n, ctrl.T);
        case "Ramp" 
            w = 0.2 + 0.8*(1:sys.n*ctrl.T)/(sys.n*ctrl.T);
            w = reshape(w, sys.n, ctrl.T); % Reshape for practical convenience
        case "Sinusoidal wave"
            w = sin(freq*(1:sys.n*ctrl.T));
            w = reshape(w, sys.n, ctrl.T); % Reshape for practical convenience
        case "Moving average sinousoidal wave"
            w = 0.5*sin(freq*(1:sys.n*ctrl.T)) + [zeros(1, sys.n*(ctrl.T - floor(ctrl.T/2))) 0.5*ones(1, sys.n*floor(ctrl.T/2))];
            w = reshape(w, sys.n, ctrl.T); % Reshape for practical convenience
        case "Sawtooth wave"
            w = sawtooth(freq*(1:sys.n*ctrl.T), 1);
            w = reshape(w, sys.n, ctrl.T); % Reshape for practical convenience
        case "Stairs function"
            w = [-ones(sys.n, ctrl.T - 2*floor(ctrl.T/3)) zeros(sys.n, floor(ctrl.T/3)) ones(sys.n, floor(ctrl.T/3))];
        otherwise
            error('Incorrect disturbance profile...')
    end
end

