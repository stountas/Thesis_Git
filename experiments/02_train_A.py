import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Automatically find the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_precomputed_splits
from src.models import SurrogateMLP
from src.training import train_lbfgs, train_adam

def train_worker(args):
    """Independent worker function for parallel execution."""
    frac_pct, method, seed, optim_name, data_dir, models_root, loss_type, arch = args
    
    # Critical: Prevent PyTorch from using all cores inside this single process
    torch.set_num_threads(1)
    
    prefix = f"{method}_{frac_pct}pct_seed{seed}"
    if not os.path.exists(os.path.join(data_dir, f"{prefix}_train.csv")):
        return f"⚠️ Skipped {prefix} (data not found)"
        
    opt_str = optim_name.lower().replace("-", "") 
    save_dir = os.path.join(models_root, f"{frac_pct}pct", opt_str, loss_type)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"model_{method}_{frac_pct}pct_{opt_str}_{loss_type}_dseed_{seed}_arch_{arch}.pt"
    history_filename = f"history_{filename.replace('.pt', '.csv')}"
    
    save_path = os.path.join(save_dir, filename)
    history_path = os.path.join(save_dir, history_filename)
    
    if os.path.exists(save_path) and os.path.exists(history_path):
        return f"⏭️ Already exists: {filename}"
        
    # Load Data
    data = load_precomputed_splits(prefix, data_dir)
    
    # Init Model & Loss
    torch.manual_seed(seed)
    model = SurrogateMLP()
    criterion = nn.MSELoss()
    
    # Train
    if optim_name == "L-BFGS":
        model, history = train_lbfgs(model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], criterion)
    else:
        model, history = train_adam(model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], criterion)
    
    # Save
    torch.save(model.state_dict(), save_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    
    return f"✅ Trained: {filename}"

def main():
    print("=========================================================")
    print("🚀 EXPERIMENT 1A: PARALLEL Training Adam & L-BFGS (Standard Loss)")
    print("=========================================================")
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_ROOT = os.path.join(BASE_DIR, "models")
    
    FRACTIONS = [1.0, 0.75, 0.50, 0.25, 0.125, 0.10, 0.075, 0.05]
    SEEDS = [42, 123, 456, 789, 1024, 2024, 3049, 4096, 5012, 6120]
    METHODS = ["kcenters", "random"]
    OPTIMIZERS = ["L-BFGS", "Adam"]
    LOSS_TYPE = "std_loss"
    ARCH = "20_20_20"
    
    # Build a list of all tasks to execute
    tasks = []
    for frac in FRACTIONS:
        frac_pct = int(frac * 100) if frac == 1.0 else frac * 100
        for method in METHODS:
            for seed in SEEDS:
                for optim_name in OPTIMIZERS:
                    tasks.append((frac_pct, method, seed, optim_name, DATA_DIR, MODELS_ROOT, LOSS_TYPE, ARCH))
                    
    print(f"[+] Total tasks queued: {len(tasks)}")
    print("[+] Launching pool with 6 concurrent workers...")

    # Execute tasks in parallel
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(train_worker, task) for task in tasks]
        
        for future in as_completed(futures):
            print(future.result())

    print("\n✅ Script 01_train_A complete! All standard models saved.")

if __name__ == "__main__":
    main()