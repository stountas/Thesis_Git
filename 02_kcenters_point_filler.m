% =========================================================================
% COMSOL DESIGN SPACE GENERATOR (ITERATIVE LOW-DENSITY FILLER)
% =========================================================================
clear; clc; close all;

% 1. Load Existing Data
data_file = 'all_clusters_COMBINED_MASTER_NoDuplicates.txt';
try
    existing_data_table = readtable(data_file, 'Delimiter', '\t');
    existing_data = table2array(existing_data_table(:, 1:4));
catch
    existing_data = load(data_file);
    existing_data = existing_data(:, 1:4);
end

% 2. Define Reactor Parameter Boundaries
bounds = [
    100,    1500;   % Power (W)
    0.01,   0.060;  % Pressure (Torr)
    20,     120;    % Feed flow (sccm)
    50,     550     % Vbias (V)
];

% Normalize existing data to [0, 1] range for fair distance calculation
norm_existing_data = zeros(size(existing_data));
for col = 1:4
    min_val = bounds(col, 1);
    max_val = bounds(col, 2);
    norm_existing_data(:, col) = (existing_data(:, col) - min_val) / (max_val - min_val);
end

valid_idx = all(norm_existing_data >= 0 & norm_existing_data <= 1, 2);
norm_existing_data = norm_existing_data(valid_idx, :);

% 3. User Setup
num_points = input('Enter total number of NEW points to generate (e.g., 1000): ');
num_clusters = input('Enter number of parallel clusters to split into (e.g., 4 or 88): ');
disp('Finding low-density regions to generate fill points...');

% 4. Generate Candidates & GREEDY SELECTION (The Fix)
num_candidates = num_points * 20; 
raw_candidates = rand(num_candidates, 4);

disp('Calculating initial nearest neighbor distances...');
% Get the distance from every candidate to the nearest existing original point
[~, min_dists] = knnsearch(norm_existing_data, raw_candidates);

best_candidates = zeros(num_points, 4);

disp('Iteratively selecting points to avoid clumping...');
for i = 1:num_points
    % Find the candidate that is currently the absolute furthest from anything
    [~, max_idx] = max(min_dists);
    
    % Save this candidate
    best_candidates(i, :) = raw_candidates(max_idx, :);
    
    % Update the distances of all remaining candidates. 
    % Their new distance is the minimum of their old distance OR their distance to this brand new point.
    dist_to_new = sqrt(sum((raw_candidates - best_candidates(i, :)).^2, 2));
    min_dists = min(min_dists, dist_to_new);
    
    % Prevent this specific candidate from being picked again
    min_dists(max_idx) = -1; 
end

% 5. Scale Selected Samples back to boundaries
scaled_data = zeros(num_points, 4);
for col = 1:4
    min_val = bounds(col, 1);
    max_val = bounds(col, 2);
    scaled_data(:, col) = min_val + (max_val - min_val) * best_candidates(:, col);
end

% 6. Sort Data
scaled_data = sortrows(scaled_data, [2, 1, 3, 4]); 

% 7. Split and Save
points_per_cluster = floor(num_points / num_clusters);
disp('Writing cluster text files...');
for c = 1:num_clusters
    start_idx = (c-1) * points_per_cluster + 1;
    if c == num_clusters
        end_idx = num_points;
    else
        end_idx = c * points_per_cluster;
    end
    
    cluster_chunk = scaled_data(start_idx:end_idx, :);
    filename = sprintf('cluster_dpOnt_%d_final_fill.txt', c);
    
    fid = fopen(filename, 'w');
    for r = 1:size(cluster_chunk, 1)
        fprintf(fid, '%.6f\t%.6f\t%.6f\t%.6f\n', ...
            cluster_chunk(r, 1), cluster_chunk(r, 2), ...
            cluster_chunk(r, 3), cluster_chunk(r, 4));
    end
    fclose(fid);
end
disp('Done!');

% =========================================================================
% 8. PLOTTING
% =========================================================================
figure('Name', 'Design Space: Original vs Filled (Iterative)', 'Position', [100, 100, 950, 750]);

color_original = [0.2, 0.5, 0.8]; 
color_new = [0.9, 0.2, 0.2];      
alpha_orig = 0.4;                 

subplot(2,2,1); hold on; grid on; box on;
scatter(existing_data(:,1), existing_data(:,2), 15, color_original, 'filled', 'MarkerFaceAlpha', alpha_orig);
scatter(scaled_data(:,1), scaled_data(:,2), 15, color_new, 'filled');
xlabel('Power (W)', 'FontWeight', 'bold'); ylabel('Pressure (Torr)', 'FontWeight', 'bold');
title('Power vs Pressure'); legend('Original Data', 'New Fill Points', 'Location', 'best');

subplot(2,2,2); hold on; grid on; box on;
scatter(existing_data(:,3), existing_data(:,4), 15, color_original, 'filled', 'MarkerFaceAlpha', alpha_orig);
scatter(scaled_data(:,3), scaled_data(:,4), 15, color_new, 'filled');
xlabel('Feed (sccm)', 'FontWeight', 'bold'); ylabel('Vbias (V)', 'FontWeight', 'bold');
title('Feed vs Vbias');

subplot(2,2,3); hold on; grid on; box on;
scatter(existing_data(:,1), existing_data(:,3), 15, color_original, 'filled', 'MarkerFaceAlpha', alpha_orig);
scatter(scaled_data(:,1), scaled_data(:,3), 15, color_new, 'filled');
xlabel('Power (W)', 'FontWeight', 'bold'); ylabel('Feed (sccm)', 'FontWeight', 'bold');
title('Power vs Feed');

subplot(2,2,4); hold on; grid on; box on;
scatter(existing_data(:,2), existing_data(:,4), 15, color_original, 'filled', 'MarkerFaceAlpha', alpha_orig);
scatter(scaled_data(:,2), scaled_data(:,4), 15, color_new, 'filled');
xlabel('Pressure (Torr)', 'FontWeight', 'bold'); ylabel('Vbias (V)', 'FontWeight', 'bold');
title('Pressure vs Vbias');

sgtitle('Design Space Point Distribution (Iterative Greedy Fill)', 'FontSize', 14, 'FontWeight', 'bold');
% =========================================================================
% 9. GENERATE 3D PLOT (Power vs Pressure vs Feed)
% =========================================================================
disp('Generating 3D visualization...');

figure('Name', '3D Design Space (Power vs Pressure vs Feed)', 'Position', [150, 150, 800, 700]);
hold on; grid on; box on;

% Add 3D scatter for original data
scatter3(existing_data(:,1), existing_data(:,2), existing_data(:,3), ...
    20, color_original, 'filled', 'MarkerFaceAlpha', alpha_orig);

% Add 3D scatter for the newly generated fill points
scatter3(scaled_data(:,1), scaled_data(:,2), scaled_data(:,3), ...
    20, color_new, 'filled');

% Set viewing angle (Azimuth, Elevation) for a good initial isometric view
view(45, 30);

% Labels and Title
xlabel('Power (W)', 'FontWeight', 'bold');
ylabel('Pressure (Torr)', 'FontWeight', 'bold');
zlabel('Feed (sccm)', 'FontWeight', 'bold');
title('3D Parameter Distribution: Power vs Pressure vs Feed', 'FontSize', 14, 'FontWeight', 'bold');

legend('Original Data', 'New Fill Points', 'Location', 'best');