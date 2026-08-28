import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np
import pandas as pd

# Automatically find the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_precomputed_splits

def extract_spatial_physics_parameters(y_train_tensor, percentile=99.0):
    """Calculates tau_i, W_i, and W_mono_i from the scaled training targets."""
    y_train = y_train_tensor.numpy()
    
    # 1st and 2nd derivatives across the 10 radial points
    first_diffs = y_train[:, 1:] - y_train[:, :-1]
    second_diffs = first_diffs[:, 1:] - first_diffs[:, :-1]
    
    epsilon = 1e-6
    
    # 1. Jaggedness Weights (Length 8)
    # Extract tau_i (max natural curvature per hinge)
    tau_i = np.percentile(np.abs(second_diffs), percentile, axis=0)
    # Calculate W_i (inverse of tau_i, normalized)
    W_i = 1.0 / (tau_i + epsilon)
    W_i = W_i / np.mean(W_i)
    
    # 2. Monotonicity Weights (Length 9)
    # Weight inversely proportional to the standard deviation of the slopes
    std_first_diffs = np.std(first_diffs, axis=0)
    W_mono_i = 1.0 / (std_first_diffs + epsilon)
    W_mono_i = W_mono_i / np.mean(W_mono_i)
    
    return tau_i, W_i, W_mono_i

def main():
    print("=========================================================")
    print("🚀 PIPELINE STEP 00b: Extracting Spatial Physics Weights")
    print("=========================================================")
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    WEIGHTS_DIR = os.path.join(DATA_DIR, "spatial_weights")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    FRACTIONS = [1.0, 0.75, 0.50, 0.25, 0.125, 0.10, 0.075, 0.05]
    SEEDS = [42, 123, 456, 789, 1024, 2024, 3049, 4096, 5012, 6120]
    METHODS = ["kcenters", "random"]
    
    total_files = 0
    
    for frac in FRACTIONS:
        frac_str = "100" if frac == 1.0 else f"{frac * 100}"
        
        for method in METHODS:
            for seed in SEEDS:
                prefix = f"{method}_{frac_str}pct_seed{seed}"
                
                # Check if train split exists
                if not os.path.exists(os.path.join(DATA_DIR, f"{prefix}_train.csv")):
                    continue
                    
                # Load the data splits (to get the scaled y_tr)
                data = load_precomputed_splits(prefix, DATA_DIR)
                
                # Extract parameters
                tau_i, W_i, W_mono_i = extract_spatial_physics_parameters(data["y_tr"])
                
                # Format into a DataFrame (9 intervals, 8 hinges)
                records = []
                for i in range(9):
                    records.append({
                        "Interval_Index": i + 1,
                        "W_mono_i": round(float(W_mono_i[i]), 6),
                        "Tau_Limit": round(float(tau_i[i]), 6) if i < 8 else np.nan,
                        "Spatial_Weight": round(float(W_i[i]), 6) if i < 8 else np.nan
                    })
                
                df_weights = pd.DataFrame(records)
                
                # Save specifically for this method, budget, and seed
                csv_filename = f"weights_{prefix}.csv"
                df_weights.to_csv(os.path.join(WEIGHTS_DIR, csv_filename), index=False)
                total_files += 1
                
        print(f"    [+] Processed Budget: {frac_str}%")

    print(f"\n✅ Success! Exported {total_files} spatial weight files to '{WEIGHTS_DIR}'.")
    print("Each seed now has its own mathematically isolated physics parameters!")

if __name__ == "__main__":
    main()