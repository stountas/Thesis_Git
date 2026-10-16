% =========================================================================
% FAST Weight Optimization for Plasma Path Sorting (Multi-Objective)
% Single-File Version: Reads one file -> Sequences entire dataset -> Exports one file
% =========================================================================
clear; clc; close all;

fprintf('\n======================================================\n');
fprintf('STARTING SINGLE-FILE MINIMAX PATH SEQUENCER\n');
fprintf('======================================================\n');

% =========================================================================
% 1. SPECIFY INPUT & OUTPUT FILES
% =========================================================================
input_filename  = 'v10_Ar_eth_dataset_full.txt'; 
output_filename = 'v10_Ar_eth_dataset_sequenced.txt';

if ~exist(input_filename, 'file')
    error('Input file "%s" not found! Please verify the path/filename.', input_filename);
end

fprintf('Loading dataset from: %s\n', input_filename);

% Check if the original file contains a text header
fid_in = fopen(input_filename, 'r');
first_line = fgetl(fid_in);
fclose(fid_in);
has_header = any(isletter(first_line));

% Load numeric matrix (readmatrix automatically skips header text)
raw_data = readmatrix(input_filename);
num_points = size(raw_data, 1);
num_cols   = size(raw_data, 2);

fprintf('Successfully loaded %d points with %d columns.\n', num_points, num_cols);

% =========================================================================
% 2. FEATURE EXTRACTION & NORMALIZATION
% =========================================================================
% Column Mapping: 1: Power (W), 2: Pressure (Torr), 3: Feed (sccm), 4: Vbias (V)
power    = raw_data(:, 1);
pressure = raw_data(:, 2);
feed     = raw_data(:, 3);

% Calculate E/N Proxy (Power / Pressure)
EN_proxy = power ./ pressure; 
augmented_data = [raw_data(:, 1:4), EN_proxy];
norm_matrix    = zeros(size(augmented_data));

min_vals = min(augmented_data(:, 1:4), [], 1);
max_vals = max(augmented_data(:, 1:4), [], 1);
ranges   = max_vals - min_vals;
ranges(ranges == 0) = 1; % Prevent division by zero

for col = 1:5
    c_min = min(augmented_data(:, col));
    c_max = max(augmented_data(:, col));
    if c_max ~= c_min
        norm_matrix(:, col) = (augmented_data(:, col) - c_min) / (c_max - c_min);
    end
end

% =========================================================================
% 3. IDENTIFY STARTING SEED RECIPE (MOTHER POINT)
% =========================================================================
% Target initial recipe: [Power (W), Pressure (Torr), Feed (sccm), Vbias (V)]
target_seed = [1000, 0.018, 30, 100]; 

norm_target = (target_seed - min_vals(1:4)) ./ ranges(1:4);
dists_to_seed = sum((norm_matrix(:, 1:4) - norm_target).^2, 2);
[~, start_idx] = min(dists_to_seed);

fprintf('\nTarget Seed Recipe: [Power = %.0f W, Press = %.4f Torr, Feed = %.0f sccm, Vbias = %.0f V]\n', ...
    target_seed(1), target_seed(2), target_seed(3), target_seed(4));
fprintf('Nearest starting point located at Row: %d\n', start_idx);
fprintf('Actual coordinates: Power = %.2f W, Press = %.4f Torr, Feed = %.2f sccm, Vbias = %.2f V\n', ...
    raw_data(start_idx, 1), raw_data(start_idx, 2), raw_data(start_idx, 3), raw_data(start_idx, 4));

% =========================================================================
% 4. MULTI-OBJECTIVE WEIGHT OPTIMIZATION (PARALLEL SEARCH)
% =========================================================================
iterations       = 30000; % Search iterations
fatal_EN_jump    = 14000; % Max jump in Power/Pressure
fatal_flow_jump  = 15;    % Max jump in Feed (sccm)
fatal_press_jump = 0.005; % Max jump in Pressure (Torr)

fprintf('\nCommencing Monte Carlo search across weight space (%d iterations)...\n', iterations);

fitness_scores   = zeros(iterations, 1);
max_EN_jumps     = zeros(iterations, 1);
max_flow_jumps   = zeros(iterations, 1);
max_press_jumps  = zeros(iterations, 1); 
weights_history  = zeros(iterations, 5);

tic;
parfor iter = 1:iterations
    % Random weight candidate (0 to 10 scale)
    w_P  = rand() * 10;
    w_p  = rand() * 10; 
    w_Q  = rand() * 10; 
    w_V  = 0;          % Bias voltage has low solver divergence risk
    w_EN = rand() * 10;
    
    current_weights = [w_P, w_p, w_Q, w_V, w_EN];
    weighted_norm_matrix = norm_matrix .* current_weights;
    
    % Vectorized Nearest-Neighbor Tour
    visited = false(num_points, 1);
    sorted_indices = zeros(num_points, 1);
    current_idx = start_idx;
    sorted_indices(1) = current_idx;
    visited(current_idx) = true;
    
    for step = 2:num_points
        current_coords = weighted_norm_matrix(current_idx, :);
        unvisited_mask = ~visited;
        unvisited_indices = find(unvisited_mask);
        
        diffs = weighted_norm_matrix(unvisited_mask, :) - current_coords;
        dists = sum(diffs.^2, 2); 
        
        [~, min_rel_idx] = min(dists);
        current_idx = unvisited_indices(min_rel_idx);
        sorted_indices(step) = current_idx;
        visited(current_idx) = true;
    end
    
    % Jump Metrics
    sorted_EN    = EN_proxy(sorted_indices);
    max_EN_jump  = max(abs(diff(sorted_EN)));
    
    sorted_feed  = feed(sorted_indices);
    max_flow_jump = max(abs(diff(sorted_feed)));
    
    sorted_press = pressure(sorted_indices);
    max_press_jump = max(abs(diff(sorted_press)));
    
    % Normalized Bottleneck Fitness
    fitness = (max_EN_jump / fatal_EN_jump) + ...
              (max_flow_jump / fatal_flow_jump) + ...
              (max_press_jump / fatal_press_jump);
    
    % Penalty if any constraint is exceeded
    if max_EN_jump > fatal_EN_jump || ...
       max_flow_jump > fatal_flow_jump || ...
       max_press_jump > fatal_press_jump
        fitness = fitness + 1000;
    end
    
    fitness_scores(iter)  = fitness;
    max_EN_jumps(iter)    = max_EN_jump;
    max_flow_jumps(iter)  = max_flow_jump;
    max_press_jumps(iter) = max_press_jump;
    weights_history(iter, :) = current_weights;
end
search_time = toc;

% Extract Champion Weights
[~, best_idx] = min(fitness_scores);
best_max_EN   = max_EN_jumps(best_idx);
best_max_flow = max_flow_jumps(best_idx);
best_max_press= max_press_jumps(best_idx);
best_weights  = weights_history(best_idx, :);

fprintf('\n=== SEARCH COMPLETE (%.2f seconds) ===\n', search_time);
fprintf('Target Limits:            E/N < %d | Flow < %d sccm | Pressure Jump < %.4f Torr\n', ...
    fatal_EN_jump, fatal_flow_jump, fatal_press_jump);
fprintf('Best Path Max E/N Jump:   %.0f\n', best_max_EN);
fprintf('Best Path Max Flow Jump:  %.2f sccm\n', best_max_flow);
fprintf('Best Path Max Press Jump: %.5f Torr\n\n', best_max_press);

if best_max_EN < fatal_EN_jump && best_max_flow < fatal_flow_jump && best_max_press < fatal_press_jump
    fprintf('SUCCESS: Sequence path stays strictly below all crash thresholds!\n');
else
    warning('One or more jumps exceeded desired thresholds due to point sparsity.');
end

fprintf('\nOPTIMIZED FEATURE WEIGHTS:\n');
fprintf('  Power Weight:     %.3f\n', best_weights(1));
fprintf('  Pressure Weight:  %.3f\n', best_weights(2));
fprintf('  Feed Weight:      %.3f\n', best_weights(3));
fprintf('  Vbias Weight:     %.3f\n', best_weights(4));
fprintf('  E/N Proxy Weight: %.3f\n', best_weights(5));

% =========================================================================
% 5. CONSTRUCT FINAL SORTED PATH & EXPORT
% =========================================================================
weighted_norm_matrix = norm_matrix .* best_weights;
visited = false(num_points, 1);
final_sorted_indices = zeros(num_points, 1);
current_idx = start_idx;
final_sorted_indices(1) = current_idx; 
visited(current_idx) = true;

for step = 2:num_points
    unvisited_mask = ~visited;
    unvisited_indices = find(unvisited_mask);
    diffs = weighted_norm_matrix(unvisited_mask, :) - weighted_norm_matrix(current_idx, :);
    dists = sum(diffs.^2, 2); 
    [~, min_rel_idx] = min(dists);
    current_idx = unvisited_indices(min_rel_idx);
    final_sorted_indices(step) = current_idx; 
    visited(current_idx) = true;
end

sorted_data = raw_data(final_sorted_indices, :);

% =========================================================================
% 6. DIAGNOSTIC PLOTS
% =========================================================================
figure('Name', 'Trajectory Diagnostics', 'Position', [100 100 1200 400]);

% Plot 1: E/N Jumps
subplot(1, 3, 1);
plot(diff(EN_proxy), 'Color', [0.8 0.3 0.3], 'DisplayName', 'Original'); hold on;
plot(diff(EN_proxy(final_sorted_indices)), 'g', 'LineWidth', 1.2, 'DisplayName', 'Optimized');
yline(fatal_EN_jump, '--r', 'Threshold'); yline(-fatal_EN_jump, '--r');
xlabel('Step'); ylabel('\Delta E/N (W/Torr)'); title('E/N Proxy Jumps'); grid on; legend('Location', 'best');

% Plot 2: Flow Jumps
subplot(1, 3, 2);
plot(diff(feed), 'Color', [0.8 0.3 0.3], 'DisplayName', 'Original'); hold on;
plot(diff(feed(final_sorted_indices)), 'b', 'LineWidth', 1.2, 'DisplayName', 'Optimized');
yline(fatal_flow_jump, '--r', 'Threshold'); yline(-fatal_flow_jump, '--r');
xlabel('Step'); ylabel('\Delta Feed (sccm)'); title('Feed Flow Jumps'); grid on; legend('Location', 'best');

% Plot 3: Pressure Jumps
subplot(1, 3, 3);
plot(diff(pressure), 'Color', [0.8 0.3 0.3], 'DisplayName', 'Original'); hold on;
plot(diff(pressure(final_sorted_indices)), 'm', 'LineWidth', 1.2, 'DisplayName', 'Optimized');
yline(fatal_press_jump, '--r', 'Threshold'); yline(-fatal_press_jump, '--r');
xlabel('Step'); ylabel('\Delta Pressure (Torr)'); title('Pressure Jumps'); grid on; legend('Location', 'best');

% =========================================================================
% 7. WRITE TO SINGLE OUTPUT FILE
% =========================================================================
fprintf('\nWriting sorted data to: %s ...\n', output_filename);
fid_out = fopen(output_filename, 'w');

% Write header if original file had one
if has_header
    fprintf(fid_out, '%s\n', first_line);
end

% Build tab-separated format string for all data columns
fmt_str = [repmat('%.6f\t', 1, num_cols - 1), '%.6f\n'];
for r = 1:num_points
    fprintf(fid_out, fmt_str, sorted_data(r, :));
end
fclose(fid_out);

fprintf('SUCCESS! All %d points sequenced and exported to %s.\n', num_points, output_filename);
