import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

# Automatically find the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_precomputed_splits
from src.models import SurrogateMLP
from src.training import train_lbfgs

# =========================================================
# 1. EXTRACT SPATIAL PHYSICS PARAMETERS
# =========================================================
def extract_spatial_physics_parameters(y_train_tensor, percentile=99.0):
    y_train = y_train_tensor.numpy()
    first_diffs = y_train[:, 1:] - y_train[:, :-1]
    second_diffs = first_diffs[:, 1:] - first_diffs[:, :-1]
    
    epsilon = 1e-6
    
    # 1. Jaggedness Weights (Length 8)
    tau_i = np.percentile(np.abs(second_diffs), percentile, axis=0)
    W_i = 1.0 / (tau_i + epsilon)
    W_i = W_i / np.mean(W_i) 
    
    # 2. Monotonicity Weights (Length 9)
    std_first_diffs = np.std(first_diffs, axis=0)
    W_mono_i = 1.0 / (std_first_diffs + epsilon)
    W_mono_i = W_mono_i / np.mean(W_mono_i)
    
    return torch.tensor(tau_i, dtype=torch.float32), torch.tensor(W_i, dtype=torch.float32), torch.tensor(W_mono_i, dtype=torch.float32)

# =========================================================
# 2. LOSS FUNCTIONS
# =========================================================
class PureMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        return self.mse(pred, target)

class PhysicsGuidedLoss(nn.Module):
    def __init__(self, lambda_mono=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_mono = lambda_mono
        
    def forward(self, pred, target):
        loss_data = self.mse(pred, target)
        radial_diffs = pred[:, 1:] - pred[:, :-1]
        loss_mono = torch.mean(torch.relu(radial_diffs) ** 2)
        return loss_data + (self.lambda_mono * loss_mono)

class AdvancedPhysicsLoss(nn.Module):
    def __init__(self, tau_i, W_i, W_mono_i, p_mean, p_std, lambda_mono=1.0, lambda_smooth=1.0, lambda_pos=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        
        self.lambda_mono = lambda_mono
        self.lambda_smooth = lambda_smooth
        self.lambda_pos = lambda_pos
        
        self.register_buffer('tau_i', tau_i)
        self.register_buffer('W_i', W_i)
        self.register_buffer('W_mono_i', W_mono_i)
        self.register_buffer('p_mean', torch.tensor(p_mean, dtype=torch.float32))
        self.register_buffer('p_std', torch.tensor(p_std, dtype=torch.float32))
        
    def forward(self, pred, target):
        loss_data = self.mse(pred, target)
        
        # 1. Spatially-Weighted Monotonicity
        first_diffs = pred[:, 1:] - pred[:, :-1]
        excess_mono = torch.relu(first_diffs)
        weighted_mono = excess_mono * self.W_mono_i
        loss_mono = torch.mean(weighted_mono ** 2)
        
        # 2. Spatially-Aware Jaggedness
        second_diffs = first_diffs[:, 1:] - first_diffs[:, :-1]
        excess_jaggedness = torch.relu(torch.abs(second_diffs) - self.tau_i)
        weighted_jaggedness = excess_jaggedness * self.W_i
        loss_smooth = torch.mean(weighted_jaggedness ** 2)
        
        # 3. Positivity
        unscaled_pred = (pred * self.p_std) + self.p_mean
        loss_pos = torch.mean(torch.relu(-unscaled_pred) ** 2)
        
        return loss_data + (self.lambda_mono * loss_mono) + (self.lambda_smooth * loss_smooth) + (self.lambda_pos * loss_pos)

# =========================================================
# 3. HELPER METRICS
# =========================================================
def calculate_violations(y_pred):
    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    mono_violations = np.sum(diffs > 1e-4)
    pos_violations = np.sum(y_pred < 0.0)
    return mono_violations + pos_violations

def calculate_jaggedness(y_pred):
    """Calculates the Jaggedness Index (Mean Squared 2nd Derivative)"""
    first_diffs = y_pred[:, 1:] - y_pred[:, :-1]
    second_diffs = first_diffs[:, 1:] - first_diffs[:, :-1]
    return np.mean(second_diffs ** 2)

def evaluate_metrics(y_act, y_pred):
    mae = mean_absolute_error(y_act.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_act.flatten(), y_pred.flatten()))
    r2 = r2_score(y_act.flatten(), y_pred.flatten())
    v_rate = (calculate_violations(y_pred) / (y_pred.shape[0] * y_pred.shape[1])) * 100
    jag_index = calculate_jaggedness(y_pred)
    return r2, rmse, mae, v_rate, jag_index

# =========================================================
# 4. OPTUNA OBJECTIVE FUNCTION
# =========================================================
def create_objective(data, tau_i, W_i, W_mono_i, seed):
    def objective(trial):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        
        hidden_sizes = [20, 20, 20]
        
        lambda_mono = trial.suggest_float('lambda_mono', 0.1, 20.0, log=True)
        lambda_smooth = trial.suggest_float('lambda_smooth', 0.1, 20.0, log=True)
        lambda_pos = trial.suggest_float('lambda_pos', 0.1, 20.0, log=True)
        
        model = SurrogateMLP(hidden_sizes=hidden_sizes)
        criterion = AdvancedPhysicsLoss(
            tau_i=tau_i, W_i=W_i, W_mono_i=W_mono_i, p_mean=data["p_mean"], p_std=data["p_std"], 
            lambda_mono=lambda_mono, lambda_smooth=lambda_smooth, lambda_pos=lambda_pos
        )
        
        model, _ = train_lbfgs(model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], criterion)
        model.eval()
        
        with torch.no_grad():
            preds_va = (model(data["X_va"]).numpy() * data["p_std"]) + data["p_mean"]
            y_va_raw = (data["y_va"].numpy() * data["p_std"]) + data["p_mean"]
            
        _, val_rmse, val_mae, val_v_rate, val_jaggedness = evaluate_metrics(y_va_raw, preds_va)
        
        # Save Jaggedness to the Optuna Trial so we can use it for final selection later
        trial.set_user_attr("jaggedness", float(val_jaggedness))
        
        # 3-Objective Optimization: RMSE, MAE, Violations
        return val_rmse, val_mae, val_v_rate

    return objective

# =========================================================
# 5. PLOTTING & HISTORY FUNCTIONS
# =========================================================
def extract_train_val(history):
    train_loss, val_loss = [], []
    if isinstance(history, dict):
        train_loss = history.get('train_loss', history.get('loss', []))
        val_loss = history.get('val_loss', [])
    elif isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
        tk = 'train_loss' if 'train_loss' in history[0] else 'loss'
        vk = 'val_loss' if 'val_loss' in history[0] else None
        train_loss = [step.get(tk, np.nan) for step in history]
        if vk:
            val_loss = [step.get(vk, np.nan) for step in history]
            
    if not val_loss or len(val_loss) == 0:
        val_loss = [np.nan] * len(train_loss)
    return train_loss, val_loss

def plot_comprehensive_curves(hist_mse, hist_base, hist_champ, budget, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"L-BFGS Learning Curves (Train vs Validation) - Budget {budget}%", fontweight='bold', fontsize=16, y=1.05)
    
    models_data = [
        ("Pure MSE [20, 20, 20]", hist_mse),
        ("Baseline Mono [20, 20, 20]", hist_base),
        ("Optuna Champion (Advanced)", hist_champ)
    ]
    
    for ax, (title, hist) in zip(axes, models_data):
        t_loss, v_loss = extract_train_val(hist)
        epochs = range(len(t_loss))
        
        ax.plot(epochs, t_loss, label='Train Loss', color='#1f77b4', linewidth=2)
        if not pd.isna(v_loss).all():
            ax.plot(epochs, v_loss, label='Val Loss', color='#2ca02c', linestyle='--', linewidth=2)
            
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("L-BFGS Iterations", fontweight='bold')
        ax.set_ylabel("MSE Loss", fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✅ Saved Comprehensive Training Curves: {os.path.basename(save_path)}")

def trial_logging_callback(study, trial):
    if trial.values is None: return
    best_trials = study.best_trials
    
    # We are returning (RMSE, MAE, Violations), so Violations is at index 2
    zero_viol_trials = [t for t in best_trials if t.values[2] == 0.0]
    
    if zero_viol_trials:
        # Sort by RMSE (index 0) to get models without massive outliers
        zero_viol_trials.sort(key=lambda t: t.values[0])
        top_rmse = zero_viol_trials[:5]
        # Out of the top 5 most accurate, pick the one with the smoothest curve (Lowest Jaggedness)
        best_t = min(top_rmse, key=lambda t: t.user_attrs.get('jaggedness', float('inf')))
        jag_val = best_t.user_attrs.get('jaggedness', 0.0)
        best_str = f"👑 CHAMPION (Smoothest): RMSE {best_t.values[0]:.4f} | MAE {best_t.values[1]:.4f} | Jag: {jag_val:.4f}"
    elif best_trials:
        # If no zero violation trials, sort by lowest violations
        best_t = sorted(best_trials, key=lambda t: t.values[2])[0]
        best_str = f"👑 PARETO BEST: RMSE {best_t.values[0]:.4f} | Viol {best_t.values[2]:.2f}%"
    else:
        best_str = "Waiting..."

    lm = trial.params.get('lambda_mono', 0.0)
    ls = trial.params.get('lambda_smooth', 0.0)
    lp = trial.params.get('lambda_pos', 0.0)
    
    print(f"  -> Trial {trial.number:03d} | Lm: {lm:.1f}, Ls: {ls:.1f}, Lp: {lp:.1f} | RMSE: {trial.values[0]:.4f} | MAE: {trial.values[1]:.4f} | Viol: {trial.values[2]:.2f}% || {best_str}")

# =========================================================
# 6. MAIN EXPERIMENT
# =========================================================
def main():
    print("=========================================================")
    print("🚀 EXPERIMENT 09.5.2: 3-Objective Advanced Lambda Search")
    print("=========================================================")
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", "optuna_lambda_search")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    BUDGETS = ["5.0", "10.0", "12.5", "25.0", "50.0", "100"]
    SEED = 42
    NUM_TRIALS = 50  
    NUM_CORES = 6     
    
    # --- TOGGLE: FORCE OVERWRITE ---
    FORCE_FRESH_SEARCH = True 
    
    summary_records = []
    all_point_predictions = []
    all_training_histories = []
    
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    for budget in BUDGETS:
        print(f"\n[*] Launching Optuna Lambda Search for {budget}% Budget...")
        prefix = f"kcenters_{budget}pct_seed{SEED}"
        
        if not os.path.exists(os.path.join(DATA_DIR, f"{prefix}_train.csv")):
            continue
            
        data = load_precomputed_splits(prefix, DATA_DIR)
        tau_i, W_i, W_mono_i = extract_spatial_physics_parameters(data["y_tr"])
        
        # 1. OPTUNA SEARCH 
        db_path = os.path.join(RESULTS_DIR, f"optuna_advanced_lambdas_{budget}pct.db")
        
        if FORCE_FRESH_SEARCH and os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"    🗑️ Deleted old Optuna DB to force fresh search.")
            except Exception as e:
                print(f"    ⚠️ Could not delete {db_path} - it might be open in another program.")
# 1. CREATE A LOCKED SAMPLER USING YOUR DATA SEED
        # In modern Optuna, TPESampler automatically handles multi-objective tasks!
        locked_sampler = optuna.samplers.TPESampler(seed=SEED)

        # 3-Objective Directions: Minimize RMSE, Minimize MAE, Minimize Violations
        study = optuna.create_study(
            study_name=f"adv_lambda_search_{budget}", 
            storage=f"sqlite:///{db_path}", 
            directions=["minimize", "minimize", "minimize"], 
            sampler=locked_sampler,   # <--- THIS LOCKS OPTUNA TO SEED 42
            load_if_exists=True
        )
        
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        trials_to_run = max(0, NUM_TRIALS - completed_trials)
        
        if trials_to_run > 0:
            print(f"    -> Running {trials_to_run} trials to reach {NUM_TRIALS} total...")
            study.optimize(create_objective(data, tau_i, W_i, W_mono_i, SEED), n_trials=trials_to_run, n_jobs=NUM_CORES, callbacks=[trial_logging_callback], show_progress_bar=False)
        else:
            print(f"    -> Optuna already completed {NUM_TRIALS} trials! Skipping search phase.")
        
        # Select Champion using the NEW Jaggedness Logic
        completed_trials_list = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
        zero_viol_trials = [t for t in completed_trials_list if t.values[2] == 0.0]
        best_trial = None
        
        if zero_viol_trials:
            zero_viol_trials.sort(key=lambda t: t.values[0]) # Sort by RMSE
            top_rmse_trials = zero_viol_trials[:5]
            best_trial = min(top_rmse_trials, key=lambda t: t.user_attrs.get('jaggedness', float('inf')))
        else:
            best_trials = study.best_trials
            if best_trials:
                best_trial = sorted(best_trials, key=lambda t: t.values[2])[0] # Fallback to lowest violations
            else:
                best_trial = completed_trials_list[0]
            
        hidden_sizes = [20, 20, 20] 
        l_mono, l_smooth, l_pos = best_trial.params['lambda_mono'], best_trial.params['lambda_smooth'], best_trial.params['lambda_pos']
        champ_jag = best_trial.user_attrs.get('jaggedness', 0.0)
        
        # 2. TRAIN CHAMPION
        print(f"    🌟 Retraining CHAMPION MODEL with optimal Lambdas -> Lm: {l_mono:.2f}, Ls: {l_smooth:.2f}, Lp: {l_pos:.2f} (Jaggedness: {champ_jag:.4f})")
        champion_model = SurrogateMLP(hidden_sizes=hidden_sizes)
        champion_criterion = AdvancedPhysicsLoss(tau_i=tau_i, W_i=W_i, W_mono_i=W_mono_i, p_mean=data["p_mean"], p_std=data["p_std"], lambda_mono=l_mono, lambda_smooth=l_smooth, lambda_pos=l_pos)
        torch.manual_seed(SEED)
        champion_model, champ_history = train_lbfgs(champion_model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], champion_criterion)
        champion_model.eval()
        
        # 3. TRAIN BASELINE [20, 20, 20] (Mono \lambda=1.0)
        print(f"    📏 Retraining BASELINE MODEL [20, 20, 20] (Mono λ=1.0)...")
        base_model = SurrogateMLP(hidden_sizes=[20, 20, 20])
        base_criterion = PhysicsGuidedLoss(lambda_mono=1.0)
        torch.manual_seed(SEED)
        base_model, base_history = train_lbfgs(base_model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], base_criterion)
        base_model.eval()

        # 4. TRAIN PURE MSE [20, 20, 20] (No Physics)
        print(f"    📉 Retraining PURE MSE MODEL [20, 20, 20] (No Physics)...")
        mse_model = SurrogateMLP(hidden_sizes=[20, 20, 20])
        mse_criterion = PureMSELoss()
        torch.manual_seed(SEED)
        mse_model, mse_history = train_lbfgs(mse_model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], mse_criterion)
        mse_model.eval()
        
        # 5. RECORD HISTORIES & PLOT COMPREHENSIVE CURVES
        plot_comprehensive_curves(mse_history, base_history, champ_history, budget, os.path.join(RESULTS_DIR, f"09.5.2_Training_Curves_1x3_{budget}pct.png"))
        
        for m_name, hist_obj in [("Pure MSE", mse_history), ("Baseline Mono", base_history), ("Champion", champ_history)]:
            t_loss, v_loss = extract_train_val(hist_obj)
            for it, (t, v) in enumerate(zip(t_loss, v_loss)):
                all_training_histories.append({
                    "Budget": budget,
                    "Model": m_name,
                    "Iteration": it + 1,
                    "Train_Loss": t,
                    "Val_Loss": v
                })
        
        # 6. EVALUATE ON UNSEEN DATA
        if data["X_unseen"] is not None:
            y_act = data["y_unseen_raw"]
            
            with torch.no_grad():
                preds_champ = (champion_model(data["X_unseen"]).numpy() * data["p_std"]) + data["p_mean"]
                preds_base = (base_model(data["X_unseen"]).numpy() * data["p_std"]) + data["p_mean"]
                preds_mse = (mse_model(data["X_unseen"]).numpy() * data["p_std"]) + data["p_mean"]
                
            r2_c, rmse_c, mae_c, v_c, jag_c = evaluate_metrics(y_act, preds_champ)
            r2_b, rmse_b, mae_b, v_b, jag_b = evaluate_metrics(y_act, preds_base)
            r2_m, rmse_m, mae_m, v_m, jag_m = evaluate_metrics(y_act, preds_mse)
            
            summary_records.append({
                "Budget": f"{budget}%", "Model": "Pure MSE [20, 20, 20]", 
                "Lambdas": "None",
                "Global MAE": round(mae_m, 4), "Global RMSE": round(rmse_m, 4), "Violations (%)": round(v_m, 2), "Jaggedness": round(jag_m, 4)
            })
            summary_records.append({
                "Budget": f"{budget}%", "Model": "Baseline [20, 20, 20]", 
                "Lambdas": "λ_Mono=1.0",
                "Global MAE": round(mae_b, 4), "Global RMSE": round(rmse_b, 4), "Violations (%)": round(v_b, 2), "Jaggedness": round(jag_b, 4)
            })
            summary_records.append({
                "Budget": f"{budget}%", "Model": "Optuna Champion", 
                "Lambdas": f"Lm={l_mono:.2f}, Ls={l_smooth:.2f}, Lp={l_pos:.2f}",
                "Global MAE": round(mae_c, 4), "Global RMSE": round(rmse_c, 4), "Violations (%)": round(v_c, 2), "Jaggedness": round(jag_c, 4)
            })
            
            for idx in range(y_act.shape[0]):
                for p in range(y_act.shape[1]):
                    all_point_predictions.append({
                        "Budget": budget, "Recipe_Idx": idx, "Radial_Point": p + 1,
                        "Ground_Truth": round(y_act[idx, p], 4),
                        "Pred_PureMSE": round(preds_mse[idx, p], 4),
                        "Pred_Baseline": round(preds_base[idx, p], 4),
                        "Pred_Champion": round(preds_champ[idx, p], 4)
                    })

    if summary_records:
        df_metrics = pd.DataFrame(summary_records)
        df_metrics.to_csv(os.path.join(RESULTS_DIR, "09.5.2_Optuna_Advanced_Champion_Metrics.csv"), index=False)
        print("\n--- PURE MSE vs BASELINE vs CHAMPION GLOBAL PERFORMANCE ---")
        print(df_metrics.to_string(index=False))
        
        df_points = pd.DataFrame(all_point_predictions)
        df_points.to_csv(os.path.join(RESULTS_DIR, "09.5.2_All_Point_Predictions.csv"), index=False)
        
        df_history = pd.DataFrame(all_training_histories)
        df_history.to_csv(os.path.join(RESULTS_DIR, "09.5.2_Training_Histories.csv"), index=False)
        
        print("\n✅ All Point Predictions saved to 09.5.2_All_Point_Predictions.csv")
        print("\n✅ All Training Histories saved to 09.5.2_Training_Histories.csv")

if __name__ == "__main__":
    main()