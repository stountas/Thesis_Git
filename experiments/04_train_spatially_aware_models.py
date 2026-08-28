import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Suppress Optuna's verbose trial logging so it doesn't flood the console
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Automatically find the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_precomputed_splits
from src.models import SurrogateMLP
from src.physics_loss import SpatiallyAwarePhysicsLoss
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
    return (violations / (y_pred.shape[0] * (y_pred.shape[1]-1))) * 100

def calculate_jaggedness(y_pred):
    first_diffs = y_pred[:, 1:] - y_pred[:, :-1]
    second_diffs = first_diffs[:, 1:] - first_diffs[:, :-1]
    return np.mean(second_diffs ** 2)

def evaluate_metrics(y_actual, y_pred):
    r2 = r2_score(y_actual.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_actual.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual.flatten(), y_pred.flatten())
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    v_rate = calculate_violations_rate(y_pred)
    jag_index = calculate_jaggedness(y_pred)
    return r2, rmse, mae, mape, v_rate, jag_index

# ---------------------------------------------------------
# SPATIAL WEIGHTS LOADER
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
            
# ---------------------------------------------------------
# PARALLEL WORKER
# ---------------------------------------------------------
def train_and_eval_worker(args):
    # Unpack trial_id along with other arguments
    frac_str, method, seed, data_dir, models_root, l_mono, l_smooth, l_pos, trial_id = args
    
    torch.set_num_threads(1)
    
    opt_str = "lbfgs"
    loss_type = "spatial_loss_optuna" 
    arch = "20_20_20"
    prefix = f"{method}_{frac_str}pct_seed{seed}"
    
    if not os.path.exists(os.path.join(data_dir, f"{prefix}_train.csv")):
        return {"status": "error", "msg": f"    ⚠️ Skipped {prefix} (data not found)"}
        
    # Setup directories with trial subfolder: models/25.0pct/lbfgs/spatial_loss_optuna/trial_XXX/
    save_dir = os.path.join(models_root, f"{frac_str}pct", opt_str, loss_type, f"trial_{trial_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"model_{method}_{frac_str}pct_{opt_str}_{loss_type}_trial_{trial_id}_dseed_{seed}_arch_{arch}.pt"
    history_filename = f"history_{filename.replace('.pt', '.csv')}"
    
    save_path = os.path.join(save_dir, filename)
    history_path = os.path.join(save_dir, history_filename)
    
    data = load_precomputed_splits(prefix, data_dir)
    tau_i, W_i, W_mono_i = get_spatial_parameters(prefix, data_dir)
    
    torch.manual_seed(seed)
    model = SurrogateMLP()
    
    criterion = SpatiallyAwarePhysicsLoss(
        tau_i=tau_i, W_i=W_i, W_mono_i=W_mono_i,
        p_mean=data["p_mean"], p_std=data["p_std"], 
        lambda_mono=l_mono, lambda_smooth=l_smooth, lambda_pos=l_pos
    )
    
    if not (os.path.exists(save_path) and os.path.exists(history_path)):
        model, history = train_lbfgs(model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], criterion)
        torch.save(model.state_dict(), save_path)
        pd.DataFrame(history).to_csv(history_path, index=False)
    else:
        model.load_state_dict(torch.load(save_path, weights_only=True))
        
    model.eval()
    with torch.no_grad():
        preds_scaled_loc = model(data["X_te"]).numpy()
        preds_phys_loc = (preds_scaled_loc * data["p_std"]) + data["p_mean"]
        loc_r2, loc_rmse, loc_mae, loc_mape, loc_vrate, loc_jag = evaluate_metrics(data["y_te_raw"], preds_phys_loc)
        
        if data["X_unseen"] is not None:
            preds_scaled_glob = model(data["X_unseen"]).numpy()
            preds_phys_glob = (preds_scaled_glob * data["p_std"]) + data["p_mean"]
            glob_r2, glob_rmse, glob_mae, glob_mape, glob_vrate, glob_jag = evaluate_metrics(data["y_unseen_raw"], preds_phys_glob)
        else:
            glob_r2, glob_rmse, glob_mae, glob_mape, glob_vrate, glob_jag = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    metrics = {
        "Data Budget (%)": f"{frac_str}%",
        "Sampling": method.capitalize(),
        "Trial_ID": trial_id,
        "Lambda_Mono": l_mono,
        "Lambda_Smooth": l_smooth,
        "Lambda_Pos": l_pos,
        "Seed": seed,
        "Local_R2": loc_r2, "Global_R2": glob_r2,
        "Local_RMSE": loc_rmse, "Global_RMSE": glob_rmse,
        "Local_MAE": loc_mae, "Global_MAE": glob_mae,
        "Local_MAPE": loc_mape, "Global_MAPE": glob_mape,
        "Local_Violations": loc_vrate, "Global_Violations": glob_vrate,
        "Local_Jaggedness": loc_jag, "Global_Jaggedness": glob_jag
    }
    
    return {"status": "success", "metrics": metrics, "msg": f"    ✅ Processed Trial {trial_id} | {frac_str}% | Seed {seed}"}

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("=========================================================")
    print("🚀 EXPERIMENT 2D: PARALLEL Training L-BFGS (Top Optuna Lambdas)")
    print("=========================================================")
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_ROOT = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", "summary_metrics")
    OPTUNA_DB_DIR = os.path.join(BASE_DIR, "results", "optuna_lambda_search")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    FRACTIONS = [1.0, 0.75, 0.50, 0.25, 0.125, 0.10, 0.075, 0.05]
    SEEDS = [42, 123, 456, 789, 1024, 2024, 3049, 4096, 5012, 6120] 
    METHODS = ["kcenters"] 
    
    NUM_TOP_TRIALS = 5  # Train the top 5 optimal combinations found by Optuna
    tasks = []
    
    for frac in FRACTIONS:
        frac_str = "100" if frac == 1.0 else f"{frac * 100}"
        db_path = os.path.join(OPTUNA_DB_DIR, f"optuna_advanced_lambdas_{frac_str}pct.db")
        
        top_trials_to_train = []
        
        if os.path.exists(db_path):
            study = optuna.load_study(study_name=f"adv_lambda_search_{frac_str}", storage=f"sqlite:///{db_path}")
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
            
            # Prioritize trials with exactly 0.0 violations (index 2)
            zero_viol_trials = [t for t in completed_trials if t.values[2] == 0.0]
            
            if zero_viol_trials:
                # Sort by lowest RMSE (index 0)
                zero_viol_trials.sort(key=lambda t: t.values[0])
                top_trials_to_train = zero_viol_trials[:NUM_TOP_TRIALS]
            else:
                # Fallback: Sort Pareto front by lowest violations, then lowest RMSE
                best_trials = study.best_trials
                best_trials.sort(key=lambda t: (t.values[2], t.values[0]))
                top_trials_to_train = best_trials[:NUM_TOP_TRIALS]
                
            print(f"[*] {frac_str}%: Found {len(top_trials_to_train)} top Optuna Trials to train.")
        else:
            print(f"[*] {frac_str}%: ⚠️ No Optuna DB found. Defaulting to standard lambda = 1.0 (Trial 0).")
            # Create a dummy trial representation for the default case
            class DummyTrial:
                def __init__(self):
                    self.number = 0
                    self.params = {'lambda_mono': 1.0, 'lambda_smooth': 1.0, 'lambda_pos': 1.0}
            top_trials_to_train = [DummyTrial()]

        # Generate tasks for EACH of the top trials across ALL seeds
        for method in METHODS:
            for trial in top_trials_to_train:
                l_mono = trial.params['lambda_mono']
                l_smooth = trial.params['lambda_smooth']
                l_pos = trial.params['lambda_pos']
                
                for seed in SEEDS:
                    tasks.append((frac_str, method, seed, DATA_DIR, MODELS_ROOT, l_mono, l_smooth, l_pos, trial.number))
                    
    print(f"\n[+] Total tasks queued: {len(tasks)}")
    print("[+] Launching pool with 6 concurrent workers...\n")

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
        raw_csv_path = os.path.join(RESULTS_DIR, "02D_Spatial_Loss_MultiTrial_Raw.csv")
        df_summary.to_csv(raw_csv_path, index=False)
        
        # Aggregation grouped by Budget, Sampling, AND Trial_ID
        agg_table = df_summary.groupby(["Data Budget (%)", "Sampling", "Trial_ID"]).agg({
            "Lambda_Mono": "first", "Lambda_Smooth": "first", "Lambda_Pos": "first",
            "Local_R2": ["mean", "std"], "Global_R2": ["mean", "std"],
            "Local_RMSE": ["mean", "std"], "Global_RMSE": ["mean", "std"],
            "Local_MAE": ["mean", "std"], "Global_MAE": ["mean", "std"],
            "Local_MAPE": ["mean", "std"], "Global_MAPE": ["mean", "std"],
            "Local_Violations": ["mean", "std"], "Global_Violations": ["mean", "std"],
            "Local_Jaggedness": ["mean", "std"], "Global_Jaggedness": ["mean", "std"]
        }).round(5)
        
        agg_csv_path = os.path.join(RESULTS_DIR, "02D_Spatial_Loss_MultiTrial_Aggregated.csv")
        agg_table.to_csv(agg_csv_path)
        
        print("\n✅ Script 02_train_D complete!")
        print(f"Check '{RESULTS_DIR}' for the multi-trial aggregated metrics.")

if __name__ == "__main__":
    main()