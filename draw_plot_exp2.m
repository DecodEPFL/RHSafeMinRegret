clc; close all; clear;
% Load the data
load('data_exp2_h2.mat');
load('data_exp2_hinf.mat', 'hinf');
load('data_exp2_reg.mat', 'reg');
load('exp2_disturbance_energies.mat');

% Normalize the control cost by the energy of the disturbance sequence
for i = 1:size(energy, 2)
    h2.cum_costs(i, :) = h2.cum_costs(i, :)/energy(i);
    hinf.cum_costs(i, :) = hinf.cum_costs(i, :)./energy(i);
    reg.cum_costs(i, :) = reg.cum_costs(i, :)./energy(i);
end

fig = figure; % Performance comparison with sinusoidal disturbances with different phases
plot(mean(h2.cum_costs), 'LineWidth', 1, 'Color', [0 0.4470 0.7410])
hold on
plot(mean(reg.cum_costs), 'LineWidth', 1, 'Color', [0.8500 0.3250 0.0980])
plot(mean(hinf.cum_costs), 'LineWidth', 1, 'Color', [0.9290 0.6940 0.1250])

h2_std = std(h2.cum_costs);
h2_mean_std = [mean(h2.cum_costs) + h2_std, fliplr(mean(h2.cum_costs) - h2_std)];
fill([1:10, fliplr(1:10)], h2_mean_std, [0 0.4470 0.7410], 'FaceAlpha', 0.1, 'LineStyle', 'none')

reg_std = std(reg.cum_costs);
reg_mean_std = [mean(reg.cum_costs) + reg_std, fliplr(mean(reg.cum_costs) - reg_std)];
fill([1:10, fliplr(1:10)], reg_mean_std, [0.8500 0.3250 0.0980], 'FaceAlpha', 0.1, 'LineStyle', 'none')

hinf_std = std(hinf.cum_costs);
hinf_mean_std = [mean(hinf.cum_costs) + hinf_std, fliplr(mean(hinf.cum_costs) - hinf_std)];
fill([1:10, fliplr(1:10)], hinf_mean_std, [0.9290 0.6940 0.1250], 'FaceAlpha', 0.1, 'LineStyle', 'none')

grid on; grid minor;
set(gca,'TickLabelInterpreter','latex')
xlabel('$s$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$\bar{J}$', 'Interpreter', 'latex', 'FontSize', 12)

xticks([1 2 3 4 5 6 7 8 9 10])
xticklabels({'20', '15', '12', '10', '6', '5', '4', '3', '2', '1'})

leg = legend('$\mathcal{H}_2$', '$\mathcal{R}$', '$\mathcal{H}_\infty$');
set(leg, 'Interpreter', 'latex', 'FontSize', 12);