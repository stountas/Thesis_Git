import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import glob
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed

# Automatically find the root directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from src.data_loader import load_precomputed_splits
from src.models import SurrogateMLP
from src.training import train_lbfgs
from src.physics_loss import PhysicsGuidedLoss, SpatiallyAwarePhysicsLoss

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_spatial_parameters(prefix, data_dir):
    weights_path = os.path.join(data_dir, "spatial_weights", f"weights_{prefix}.csv")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Spatial weights missing for {prefix}! Run 00b first.")
        
    df = pd.read_csv(weights_path).sort_values('Interval_Index')
    W_mono_i = torch.tensor(df['W_mono_i'].values, dtype=torch.float32)
    df_hinges = df.dropna(subset=['Tau_Limit', 'Spatial_Weight'])
    tau_i = torch.tensor(df_hinges['Tau_Limit'].values, dtype=torch.float32)
    W_i = torch.tensor(df_hinges['Spatial_Weight'].values, dtype=torch.float32)
    
    return tau_i, W_i, W_mono_i

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Violations (%)": np.nan, "Jaggedness": np.nan}
        
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    diffs = np.diff(y_pred, axis=1)
    violations_pct = (np.sum(diffs > 0) / diffs.size) * 100
    
    second_diffs = np.diff(y_pred, n=2, axis=1)
    jaggedness = np.mean(second_diffs**2)
    
    return {"MAE": mae, "RMSE": rmse, "Violations (%)": violations_pct, "Jaggedness": jaggedness}

# ---------------------------------------------------------
# PARALLEL WORKER FOR TRAINING & TIMING
# ---------------------------------------------------------
def train_worker(args):
    fw_name, ens_id, data_budget, data_seed, save_folder, base_dir, lm, ls, lp = args
    torch.set_num_threads(1) # Prevent CPU thread locking
    
    # HARDCODED REPRODUCIBILITY: Seed is strictly locked to the ensemble ID.
    ensemble_seed = 4200 + ens_id
    
    model_filename = f"model_kcenters_{data_budget}_lbfgs_dseed_{data_seed}_ens_{ens_id}.pt"
    model_path = os.path.join(save_folder, model_filename)
    
    # Because seeds are explicitly locked, existing models are guaranteed 
    # to be identical. Safe to skip! We return NaN for time since it wasn't trained now.
    if os.path.exists(model_path):
        return {
            "status": "skipped", 
            "msg": f"    ⏭️ Skipped (Exists - Seed {ensemble_seed}): [{data_budget}] {fw_name} - Ens {ens_id}",
            "Data Budget": data_budget,
            "Model Framework": fw_name,
            "Ensemble ID": ens_id,
            "Train Time (s)": np.nan
        }
        
    data_dir = os.path.join(base_dir, "data", "processed")
    split_name = f"kcenters_{data_budget}_seed{data_seed}"
    data = load_precomputed_splits(split_name, data_dir)
    
    # Apply the hardcoded deterministic seed
    torch.manual_seed(ensemble_seed)
    np.random.seed(ensemble_seed)
    
    model = SurrogateMLP()
    
    if fw_name == "Pure MSE":
        criterion = nn.MSELoss()
    elif fw_name == "Baseline Mono":
        criterion = PhysicsGuidedLoss(lambda_mono=1.0)
    elif fw_name == "Optuna Optimal":
        tau_i, W_i, W_mono_i = get_spatial_parameters(split_name, data_dir)
        criterion = SpatiallyAwarePhysicsLoss(
            tau_i=tau_i, W_i=W_i, W_mono_i=W_mono_i,
            p_mean=data["p_mean"], p_std=data["p_std"], 
            lambda_mono=lm, lambda_smooth=ls, lambda_pos=lp
        )
        
    # Measure exactly how long the L-BFGS optimizer takes for this specific model
    start_time = time.perf_counter()
    model, _ = train_lbfgs(
        model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], criterion
    )
    end_time = time.perf_counter()
    train_duration = end_time - start_time
    
    torch.save(model.state_dict(), model_path)
    
    return {
        "status": "success", 
        "msg": f"    ✅ Trained (Seed {ensemble_seed}): [{data_budget}] {fw_name} - Ens {ens_id} | Time: {train_duration:.2f}s",
        "Data Budget": data_budget,
        "Model Framework": fw_name,
        "Ensemble ID": ens_id,
        "Train Time (s)": train_duration
    }

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    DATA_SEED = 42
    TOTAL_MODELS = 200                     # Bumped to 200 (first 100 will be skipped automatically)
    ENSEMBLE_SIZES = [10, 30, 50, 100, 200] 
    BOOTSTRAP_ITERS = 50
    NUM_CORES = 6  
    
    TARGET_BUDGETS = ["5.0pct", "10.0pct", "12.5pct", "25.0pct", "50.0pct", "100.0pct"]
    
    # PERFECTLY EXTRACTED LAMBDAS
    BUDGET_CONFIGS = {
        "100.0pct": {"trial": "trial_35", "lm": 2.66, "ls": 1.15, "lp": 2.33},
        "50.0pct":  {"trial": "trial_19", "lm": 0.23, "ls": 0.35, "lp": 2.79},
        "25.0pct":  {"trial": "trial_9",  "lm": 0.97, "ls": 3.57, "lp": 7.32},
        "12.5pct":  {"trial": "trial_5",  "lm": 5.80, "ls": 0.60, "lp": 0.19},
        "10.0pct":  {"trial": "trial_3",  "lm": 3.47, "ls": 8.91, "lp": 0.15},
        "5.0pct":   {"trial": "trial_13", "lm": 11.75,"ls": 1.54, "lp": 2.03}
    }
    
    print("=========================================================")
    print(f"🚀 MASSIVE ENSEMBLE SCALING: 10 vs 30 vs 50 vs 100 vs 200")
    print(f"   Budgets: {TARGET_BUDGETS} | Cores: {NUM_CORES}")
    print("=========================================================")
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    OUT_DIR = os.path.join(BASE_DIR, "results", "summary_metrics")
    PLOT_DIR = os.path.join(BASE_DIR, "results", "analysis")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # =========================================================
    # PHASE 1: PARALLEL TRAINING & INDIVIDUAL TIMING
    # =========================================================
    print("\n" + "="*50)
    print("⚙️ PHASE 1: LOAD/TRAIN PREEXISTING MODELS (Up to 200)")
    print("="*50)
    
    tasks = []
    for budget in TARGET_BUDGETS:
        config = BUDGET_CONFIGS[budget]
        ENS_DIR = os.path.join(BASE_DIR, "models", "ensembles", budget, "lbfgs")
        frameworks = {
            "Pure MSE": os.path.join(ENS_DIR, "std_loss"),
            "Baseline Mono": os.path.join(ENS_DIR, "mono_loss"),
            "Optuna Optimal": os.path.join(ENS_DIR, "spatial_loss_optuna", config["trial"])
        }
        
        for fw_name, save_path in frameworks.items():
            os.makedirs(save_path, exist_ok=True)
            for ens_id in range(TOTAL_MODELS):
                tasks.append((fw_name, ens_id, budget, DATA_SEED, save_path, BASE_DIR, config["lm"], config["ls"], config["lp"]))
            
    print(f"[*] Queued {len(tasks)} total model training tasks. Launching workers...")
    
    individual_train_times = []
    
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(train_worker, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            
            # Record timing for every model (will log as NaN if skipped)
            individual_train_times.append({
                "Data Budget": result["Data Budget"],
                "Model Framework": result["Model Framework"],
                "Ensemble ID": result["Ensemble ID"],
                "Train Time (s)": result["Train Time (s)"]
            })
            
            # Only print actual training events to avoid spamming the console with skips
            if result["status"] == "success":
                print(result["msg"])
                
    # Save individual training times to CSV
    df_train_times = pd.DataFrame(individual_train_times)
    out_csv_indiv = os.path.join(OUT_DIR, "26_Individual_Training_Times_CPU.csv")
    df_train_times.to_csv(out_csv_indiv, index=False)
    print(f"\n✅ Individual training times for newly trained models exported to {out_csv_indiv}")

    # =========================================================
    # PHASE 2: EVALUATION AND VISUALIZATION
    # =========================================================
    print("\n" + "="*50)
    print("📊 PHASE 2: BOOTSTRAPPING (Evaluating up to 200)")
    print("="*50)
    
    for budget in TARGET_BUDGETS:
        config = BUDGET_CONFIGS[budget]
        print(f"\n[{budget}] Starting Evaluation...")
        ENS_DIR = os.path.join(BASE_DIR, "models", "ensembles", budget, "lbfgs")
        frameworks = {
            "Pure MSE": os.path.join(ENS_DIR, "std_loss"),
            "Baseline Mono": os.path.join(ENS_DIR, "mono_loss"),
            "Optuna Optimal": os.path.join(ENS_DIR, "spatial_loss_optuna", config["trial"])
        }
        
        split_name = f"kcenters_{budget}_seed{DATA_SEED}"
        data = load_precomputed_splits(split_name, DATA_DIR)
        X_eval = data["X_unseen"]
        y_actual = data["y_unseen_raw"]
        
        results = []
        for fw_name, folder_path in frameworks.items():
            search_pattern = os.path.join(folder_path, f"*dseed_{DATA_SEED}*.pt")
            model_files = glob.glob(search_pattern)
            
            if len(model_files) == 0:
                print(f"    [!] Warning: No models found for {budget} - {fw_name}")
                continue
                
            all_preds = []
            for p in model_files:
                model = SurrogateMLP()
                model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
                model.eval()
                with torch.no_grad():
                    pred = (model(X_eval).numpy() * data["p_std"]) + data["p_mean"]
                    all_preds.append(pred)
            all_preds = np.array(all_preds)
            
            for size in ENSEMBLE_SIZES:
                print(f"    -> Bootstrapping Size: {size}")
                for b in range(BOOTSTRAP_ITERS):
                    idx = np.random.choice(len(all_preds), size=size, replace=True)
                    ens_pred = np.mean(all_preds[idx], axis=0)
                    metrics = calculate_metrics(y_actual, ens_pred)
                    results.append({
                        "Model Framework": fw_name,
                        "Ensemble Size": f"{size} Models",
                        "Bootstrap_ID": b,
                        **metrics
                    })

        if not results:
            continue

        # Save Metrics
        df_results = pd.DataFrame(results)
        out_csv = os.path.join(OUT_DIR, f"24_Ensemble_Scaling_{budget}_seed{DATA_SEED}.csv")
        df_results.to_csv(out_csv, index=False)

    # =========================================================
    # PHASE 3: TIMING ANALYSIS (TRAIN AND INFERENCE)
    # =========================================================
    print("\n" + "="*50)
    print("⏱️ PHASE 3: PRECISE TIMING ANALYSIS (TRAIN & INFERENCE)")
    print("="*50)

    timing_results = []
    TIMING_SIZES = [1, 10, 30, 50, 100, 200]

    for budget in TARGET_BUDGETS:
        config = BUDGET_CONFIGS[budget]
        print(f"\n[{budget}] Running Timing Analysis...")
        ENS_DIR = os.path.join(BASE_DIR, "models", "ensembles", budget, "lbfgs")
        frameworks = {
            "Pure MSE": os.path.join(ENS_DIR, "std_loss"),
            "Baseline Mono": os.path.join(ENS_DIR, "mono_loss"),
            "Optuna Optimal": os.path.join(ENS_DIR, "spatial_loss_optuna", config["trial"])
        }

        split_name = f"kcenters_{budget}_seed{DATA_SEED}"
        data = load_precomputed_splits(split_name, DATA_DIR)
        X_eval = data["X_unseen"]

        for fw_name, folder_path in frameworks.items():
            print(f"  -> Profiling {fw_name}...")

            # 1. Measure Single Model Training Time Exactly (Ensures we always get a reading)
            start_train = time.perf_counter()
            model_dummy = SurrogateMLP()
            
            if fw_name == "Pure MSE":
                criterion_dummy = nn.MSELoss()
            elif fw_name == "Baseline Mono":
                criterion_dummy = PhysicsGuidedLoss(lambda_mono=1.0)
            elif fw_name == "Optuna Optimal":
                tau_i, W_i, W_mono_i = get_spatial_parameters(split_name, DATA_DIR)
                criterion_dummy = SpatiallyAwarePhysicsLoss(
                    tau_i=tau_i, W_i=W_i, W_mono_i=W_mono_i,
                    p_mean=data["p_mean"], p_std=data["p_std"], 
                    lambda_mono=config["lm"], lambda_smooth=config["ls"], lambda_pos=config["lp"]
                )
                
            # Perform single training
            _ = train_lbfgs(model_dummy, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], criterion_dummy)
            t_train_single = time.perf_counter() - start_train

            # 2. Get a valid pre-trained model file to duplicate for inference
            search_pattern = os.path.join(folder_path, f"*dseed_{DATA_SEED}*.pt")
            model_files = glob.glob(search_pattern)
            if not model_files:
                continue
            base_model_path = model_files[0]

            # 3. Accurately Time Inference Over Ensemble Sizes
            for size in TIMING_SIZES:
                ensemble_models = []
                # Re-loading into memory sequentially mimics actual computation graph execution
                for _ in range(size):
                    m = SurrogateMLP()
                    m.load_state_dict(torch.load(base_model_path, map_location="cpu", weights_only=True))
                    m.eval()
                    ensemble_models.append(m)

                # Warmup iterations
                with torch.no_grad():
                    for _ in range(3):
                        _ = [m(X_eval) for m in ensemble_models]

                # Exact CPU Timing (Averaged over 10 full trials)
                trials = 10
                start_inf = time.perf_counter()
                with torch.no_grad():
                    for _ in range(trials):
                        preds = [m(X_eval) for m in ensemble_models]
                        _ = torch.mean(torch.stack(preds), dim=0)
                t_inf_avg = (time.perf_counter() - start_inf) / trials

                timing_results.append({
                    "Data Budget": budget,
                    "Model Framework": fw_name,
                    "Ensemble Size": size,
                    "Single Train Time (s)": t_train_single,
                    "Speculated Total Train Time (s)": t_train_single * size,
                    "Inference Time (s)": t_inf_avg
                })

    # Save to final CSV
    df_timing = pd.DataFrame(timing_results)
    out_csv_timing = os.path.join(OUT_DIR, "25_Ensemble_Timing_Stats.csv")
    df_timing.to_csv(out_csv_timing, index=False)
    print(f"\n✅ Timing statistics correctly exported to {out_csv_timing}")

if __name__ == "__main__":
    main()