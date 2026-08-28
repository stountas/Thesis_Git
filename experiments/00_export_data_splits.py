import os
import sys
import pandas as pd

# Automatically find the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_plasma_data

def main():
    print("=========================================================")
    print("🚀 INITIALIZATION: Exporting Two-Tiered Evaluation Splits")
    print("=========================================================")
    
    DATA_PATH = os.path.join(BASE_DIR, "data", "v10_Ar_eth_dataset_full.txt")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: Dataset not found at {DATA_PATH}")
        return
        
    FRACTIONS = [1.0, 0.75, 0.50, 0.25, 0.125, 0.10, 0.075, 0.05]
    SEEDS = [42, 123, 456, 789, 1024, 2024, 3049, 4096, 5012, 6120]
    METHODS = ["random", "kcenters"]
    
    total_files = 0
    
    for frac in FRACTIONS:
        frac_pct = int(frac * 100) if frac == 1.0 else frac * 100
        print(f"\n[+] Exporting DoE Budget: {frac_pct}%...")
        
        for method in METHODS:
            for seed in SEEDS:
                # Load the two-tiered split
                data = load_plasma_data(DATA_PATH, fraction=frac, method=method, seed=seed)
                
                # Because Train dictates what is leftover for Val, Test, and Unseen, 
                # all subsets are now cleanly separated by the METHOD.
                prefix = f"{method}_{frac_pct}pct_seed{seed}"
                
                train_fname = f"{prefix}_train.csv"
                val_fname = f"{prefix}_val.csv"
                test_fname = f"{prefix}_test.csv"
                # Update this specific line inside the loop:
                unseen_fname = f"{prefix}_unseen.csv"
                
                data["df_train"].to_csv(os.path.join(OUTPUT_DIR, train_fname), index=False)
                data["df_val"].to_csv(os.path.join(OUTPUT_DIR, val_fname), index=False)
                data["df_test"].to_csv(os.path.join(OUTPUT_DIR, test_fname), index=False)
                
                # For 100% budget, there is no global unseen data left over
                if not data["df_global_unseen"].empty:
                    data["df_global_unseen"].to_csv(os.path.join(OUTPUT_DIR, unseen_fname), index=False)
                    total_files += 4
                else:
                    total_files += 3

    print(f"\n✅ Success! Exported {total_files} structural files to '{OUTPUT_DIR}'.")
    print("Your data is now mathematically sealed for Two-Tiered Evaluation (Local DoE + Global Universe)!")

if __name__ == "__main__":
    main()