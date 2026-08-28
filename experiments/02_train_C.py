import os
import sys
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Automatically find the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_precomputed_splits
from src.models import SurrogateMLP
from src.physics_loss import PhysicsGuidedLoss
from src.training import train_lbfgs

# ---------------------------------------------------------
# METRIC HELPERS
# ---------------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_violations_rate(y_pred):
    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    violations = np.sum(diffs > 1e-4) 
    return (violations / (y_pred.shape[0] * 9)) * 100

def evaluate_metrics(y_actual, y_pred):
    r2 = r2_score(y_actual.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_actual.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual.flatten(), y_pred.flatten())
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    v_rate = calculate_violations_rate(y_pred)
    return r2, rmse, mae, mape, v_rate

# ---------------------------------------------------------
# PARALLEL WORKER
# ---------------------------------------------------------
def train_and_eval_worker(args):
    # Notice we now accept frac_str instead of frac_pct
    frac_str, seed, lmbd, data_dir, models_root = args
    
    # Critical: Prevent PyTorch from oversubscribing CPU cores
    torch.set_num_threads(1)
    
    method = "kcenters"
    opt_str = "lbfgs"
    arch = "20_20_20"
    
    # This will now correctly evaluate to "kcenters_5.0pct_seed42"
    prefix = f"{method}_{frac_str}pct_seed{seed}"
    
    if not os.path.exists(os.path.join(data_dir, f"{prefix}_train.csv")):
        return {"status": "error", "msg": f"⚠️ Skipped {prefix} (data not found)"}
        
    # Setup directories (models/lamda_mono/XX.Xpct/)
    save_dir = os.path.join(models_root, "lamda_mono", f"{frac_str}pct")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"model_{method}_{frac_str}pct_{opt_str}_mono_loss_lmbd_{lmbd}_dseed_{seed}_arch_{arch}.pt"
    history_filename = f"history_{filename.replace('.pt', '.csv')}"
    
    save_path = os.path.join(save_dir, filename)
    history_path = os.path.join(save_dir, history_filename)
    
    # Load Data
    data = load_precomputed_splits(prefix, data_dir)
    
    # Init Model & Loss
    torch.manual_seed(seed)
    model = SurrogateMLP()
    criterion = PhysicsGuidedLoss(lambda_mono=lmbd)
    
    # Train only if model doesn't already exist
    if not (os.path.exists(save_path) and os.path.exists(history_path)):
        model, history = train_lbfgs(model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], criterion)
        torch.save(model.state_dict(), save_path)
        pd.DataFrame(history).to_csv(history_path, index=False)
    else:
        # If it exists, load it to perform evaluation
        model.load_state_dict(torch.load(save_path, weights_only=True))
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds_scaled_loc = model(data["X_te"]).numpy()
        preds_phys_loc = (preds_scaled_loc * data["p_std"]) + data["p_mean"]
        loc_r2, loc_rmse, loc_mae, loc_mape, loc_vrate = evaluate_metrics(data["y_te_raw"], preds_phys_loc)
        
        if data["X_unseen"] is not None:
            preds_scaled_glob = model(data["X_unseen"]).numpy()
            preds_phys_glob = (preds_scaled_glob * data["p_std"]) + data["p_mean"]
            glob_r2, glob_rmse, glob_mae, glob_mape, glob_vrate = evaluate_metrics(data["y_unseen_raw"], preds_phys_glob)
        else:
            glob_r2, glob_rmse, glob_mae, glob_mape, glob_vrate = (np.nan, np.nan, np.nan, np.nan, np.nan)

    # Return comprehensive metrics
    metrics = {
        "Data Budget (%)": f"{frac_str}%",
        "Sampling": method.capitalize(),
        "Lambda_Mono": lmbd,
        "Seed": seed,
        "Local_R2": loc_r2, "Global_R2": glob_r2,
        "Local_RMSE": loc_rmse, "Global_RMSE": glob_rmse,
        "Local_MAE": loc_mae, "Global_MAE": glob_mae,
        "Local_MAPE": loc_mape, "Global_MAPE": glob_mape,
        "Local_Violations": loc_vrate, "Global_Violations": glob_vrate
    }
    
    return {"status": "success", "metrics": metrics, "msg": f"✅ Processed: {filename}"}

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("=========================================================")
    print("🚀 EXPERIMENT 2C: PARALLEL Lambda Mono Sensitivity Study")
    print("=========================================================")
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_ROOT = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", "sensitivity_analysis")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Target values based on Thesis Table 1.11
    FRACTIONS = [0.25, 0.10, 0.05]
    LAMBDAS = [0.0, 0.1, 1.0, 10.0, 20.0]
    SEEDS = [42, 123, 456, 789, 1024, 2024, 3049, 4096, 5012, 6120]
    
    tasks = []
    for frac in FRACTIONS:
        # FIX: Generate exact strings "25.0", "10.0", "5.0" to match files
        frac_str = "100" if frac == 1.0 else f"{frac * 100}"
        for lmbd in LAMBDAS:
            for seed in SEEDS:
                tasks.append((frac_str, seed, lmbd, DATA_DIR, MODELS_ROOT))
                    
    print(f"[+] Total tasks queued: {len(tasks)}")
    print("[+] Launching pool with 6 concurrent workers...")

    summary_records = []
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(train_and_eval_worker, task) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            print(result["msg"])
            if result["status"] == "success":
                summary_records.append(result["metrics"])

    # ---------------------------------------------------------
    # EXPORT METRICS
    # ---------------------------------------------------------
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        raw_csv_path = os.path.join(RESULTS_DIR, "Table_1.11_Lambda_Mono_Study_Raw.csv")
        df_summary.to_csv(raw_csv_path, index=False)
        
        # Aggregation includes full suite of metrics
        agg_table = df_summary.groupby(["Data Budget (%)", "Sampling", "Lambda_Mono"]).agg({
            "Local_R2": ["mean", "std"], "Global_R2": ["mean", "std"],
            "Local_RMSE": ["mean", "std"], "Global_RMSE": ["mean", "std"],
            "Local_MAE": ["mean", "std"], "Global_MAE": ["mean", "std"],
            "Local_MAPE": ["mean", "std"], "Global_MAPE": ["mean", "std"],
            "Local_Violations": ["mean", "std"], "Global_Violations": ["mean", "std"]
        }).round(5)
        
        agg_csv_path = os.path.join(RESULTS_DIR, "Table_1.11_Lambda_Mono_Study_Aggregated.csv")
        agg_table.to_csv(agg_csv_path)
        
        print("\n✅ Script 02_train_C complete!")
        print(f"Check '{RESULTS_DIR}' for the aggregated Trade-Off metrics.")

if __name__ == "__main__":
    main()