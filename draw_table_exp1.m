clc; close all; clear;
% Load the data
load('data_exp1_h2.mat');
load('data_exp1_hinf.mat', 'hinf');
load('data_exp1_reg.mat', 'reg');
load('exp1_disturbance_energies.mat');

energy(energy == 0) = 1;
% Normalize the control cost by the energy of the disturbance sequence
for i = 1:size(energy, 2)
    h2.cum_costs(:, :, i) = h2.cum_costs(:, :, i)./energy;
    hinf.cum_costs(:, :, i) = hinf.cum_costs(:, :, i)./energy;
    reg.cum_costs(:, :, i) = reg.cum_costs(:, :, i)./energy;
end
% Initialize the tables
display = zeros(size(energy, 1), size(energy, 2), 3);
% For each control policy, compute the average normalized control cost
for i = 1:size(energy, 1)
    for j = 1:size(energy, 2)
        if w.is_stoch(i) == 1
            display(i, j, 1) = mean(h2.cum_costs(i, :, j));
            display(i, j, 2) = mean(hinf.cum_costs(i, :, j));
            display(i, j, 3) = mean(reg.cum_costs(i, :, j));
        else
            display(i, j, 1) = h2.cum_costs(i, 1, j);
            display(i, j, 2) = hinf.cum_costs(i, 1, j);
            display(i, j, 3) = reg.cum_costs(i, 1, j);
        end
    end
end
% Flip the tables to match the results presented in the paper
for i = 1:3
    display(:, :, i) = flip(display(:, :, i), 2);
end
fprintf('Average normalized control incurred by the H2 policy:\n')
avg_cost_h2 = display(:, :, 1)
fprintf('----------------------------------------------------------------------------------------------------\n')
fprintf('\nAverage normalized control incurred by the Hinfinity policy:\n')
avg_cost_hinf = display(:, :, 2)
fprintf('----------------------------------------------------------------------------------------------------\n')
fprintf('\nAverage normalized control incurred by the regret-optimal policy:\n')
avg_cost_reg = display(:, :, 3)
fprintf('----------------------------------------------------------------------------------------------------\n')