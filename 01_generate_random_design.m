% =========================================================================
% COMSOL DESIGN SPACE GENERATOR (PURE RANDOM - EXACT FORMAT MATCH)
% =========================================================================
clear; clc;

% 1. Define Reactor Parameter Boundaries
% Power (W): 100 to 1500
% Pressure (Torr): 8 mTorr to 60 mTorr -> Converted to Torr (0.008 to 0.060)
% Feed (sccm): 20 to 120
% Vbias (V): 50 to 550
bounds = [
    200,    1500;   % Power (W)
    0.01,  0.060;  % Pressure (Torr) <- Converted to match your production script
    20,     120;    % Feed flow (sccm)
    50,     550     % Vbias (V)
];

% 2. User Setup Configuration
num_points = input('Enter total number of points to generate (e.g., 1000): ');
num_clusters = input('Enter number of parallel clusters to split into (e.g., 4 or 88): ');

disp('Generating pure uniform random points...');

% 3. Generate Pure Uniform Random Samples
raw_samples = rand(num_points, 4);

% 4. Scale Samples to Physical Reactor Boundaries
scaled_data = zeros(num_points, 4);
for col = 1:4
    min_val = bounds(col, 1);
    max_val = bounds(col, 2);
    scaled_data(:, col) = min_val + (max_val - min_val) * raw_samples(:, col);
end

% 5. Sort Data strictly to keep Warm Starts alive
scaled_data = sortrows(scaled_data, [2, 1, 3, 4]); % Sort by Pressure, then Power
disp('-> Sorted the random points sequentially to protect the warm-start chain.');

% 6. Split and Save into Cluster Files with EXACT 6-Decimal Tab Format
points_per_cluster = floor(num_points / num_clusters);

disp('Writing cluster text files...');
for c = 1:num_clusters
    start_idx = (c-1) * points_per_cluster + 1;
    
    % Ensure the last cluster grabs any leftover rounding points
    if c == num_clusters
        end_idx = num_points;
    else
        end_idx = c * points_per_cluster;
    end
    
    cluster_chunk = scaled_data(start_idx:end_idx, :);
    filename = sprintf('cluster_dpOnt_%d_sorted_fill.txt', c);
    
    % Open file for low-level precision writing
    fid = fopen(filename, 'w');
    
    % Print row-by-row using exactly 6 decimal places separated by tabs
    for r = 1:size(cluster_chunk, 1)
        fprintf(fid, '%.6f\t%.6f\t%.6f\t%.6f\n', ...
            cluster_chunk(r, 1), ...
            cluster_chunk(r, 2), ...
            cluster_chunk(r, 3), ...
            cluster_chunk(r, 4));
    end
    
    fclose(fid);
end

fprintf('\nSuccess! Generated %d points split across %d files with your exact format layout.\n', num_points, num_clusters);