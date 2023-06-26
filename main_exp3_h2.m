%% Code to reproduce the numerical examples of the paper "On the stability of receding horizon regret optimal control"
clc; close all; clear;
addpath('./functions') % Add path to the folder with auxiliary functions
rng(1234);             % Set random seed for reproducibility
% If available, load the plant model and the optimization specifications
source_file = 'init_rho0p7_hu2_hx3p5_hw1_T20_tol0p001_eps0p5_H2.mat';
if isfile(source_file)
    load(source_file);
else
    flag = 'H2';
    initialize_control_problem(); % Create sys, opt, and sls structures
    save init_rho0p7_hu2_hx3p5_hw1_T20_tol0p001_eps0p5_H2
end
clear source_file flag
clear functions
% Simulate the closed-loop system behavior under different receding horizon
% feedback policies with different disturbance profiles. In particolar,
% we first study the effect of optimizing more or less frequently when the
% planning horizon exactly matches the control horizon
ctrl.T = 60; % Control horizon for the simulation
freq = 2*pi/ctrl.T;
freq = linspace(0.5, 2, 10)*freq;
for i = 1:10
    % Sample a disturbance realization
    w.realization = sin(freq(i)*(1:sys.n*ctrl.T));
    w.realization = reshape(w.realization, sys.n, ctrl.T); % Reshape for practical convenience
    fprintf('Disturbance energy: %.2f\n\n', norm(w.realization(:)));
    % Number of times the optimization is repeated and number of
    % control actions that are applied before optimizing again
    ctrl.N = [3 4 5 6 10 12 15 20 30 60];
    ctrl.s = [20 15 12 10 6 5 4 3 2 1];
    for k = 1:size(ctrl.N, 2)
        fprintf('Preparing to solve instance %d %d...', i, k);
        h2.cum_costs(i, k) = evaluate_receding_horizon_policy(sys, opt, sls, ctrl, k, 'H2', [], w.realization);
        h2.cum_costs(i, k)
        fprintf('solved!\n\n');
    end
end
clear i k

save data_exp3_h2