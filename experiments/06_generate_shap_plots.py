import os
# Prevent MKL memory crashes during multiprocessing
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import torch
import numpy as np
import matplotlib
# Force Matplotlib to be thread-safe in background processes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import shap
from concurrent.futures import ProcessPoolExecutor, as_completed

# Automatically find the root directory (assuming script is in experiments/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_precomputed_splits
from src.models import SurrogateMLP

# =========================================================
# WRAPPER TO CONVERT MULTI-OUTPUT TO SINGLE-OUTPUT FOR SHAP
# =========================================================
class MeanEtchRateModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return self.base_model(x).mean(dim=1, keepdim=True)

# =========================================================
# PARALLEL WORKER FUNCTION
# =========================================================
def generate_shap_plot_worker(args):
    budget, trial_id, seed = args
    
    # Restrict PyTorch threads inside the worker
    torch.set_num_threads(1)
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_ROOT = os.path.join(BASE_DIR, "models", budget, "lbfgs")
    OUT_DIR = os.path.join(BASE_DIR, "results", "analysis")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load Data
    try:
        data = load_precomputed_splits(f"kcenters_{budget}_seed{seed}", DATA_DIR)
    except Exception as e:
        return {"status": "error", "msg": f"  [!] ERROR: Could not load data for {budget}: {e}"}
        
    X_train = data["X_tr"]
    
    # Fallback to the test set if the global unseen set doesn't exist
    X_eval = data["X_unseen"] if data["X_unseen"] is not None else data["X_te"]
    
    if X_eval is None or len(X_eval) == 0:
        return {"status": "error", "msg": f"  [!] ERROR: No evaluation data available for {budget} SHAP."}
    
    # 2. Load Model
    spa_path = os.path.join(MODELS_ROOT, "spatial_loss_optuna", trial_id, f"model_kcenters_{budget}_lbfgs_spatial_loss_optuna_{trial_id}_dseed_{seed}_arch_20_20_20.pt")
    
    if not os.path.exists(spa_path):
        return {"status": "error", "msg": f"  [!] ERROR: Model not found for {budget} at {spa_path}"}
        
    base_model = SurrogateMLP()
    base_model.load_state_dict(torch.load(spa_path, map_location="cpu", weights_only=True))
    base_model.eval()
    
    model = MeanEtchRateModel(base_model)
    model.eval()
    
    # 3. Setup SHAP GradientExplainer
    background = X_train[:100]
    explainer = shap.GradientExplainer(model, background)
    
    test_samples = X_eval 
    shap_values = explainer.shap_values(test_samples)
    
    # 4. Reshape Fix
    if isinstance(shap_values, list) or isinstance(shap_values, tuple):
        shap_values = shap_values[0]
    if torch.is_tensor(shap_values):
        shap_values = shap_values.numpy()
        
    shap_values = np.reshape(shap_values, test_samples.shape)

    # 5. Feature Names
    num_features = X_train.shape[1]
    known_names = ["Power (W)", "Pressure (mTorr)", "Feed (sccm)", "V-Bias (V)"]
    if len(known_names) == num_features:
        feature_names = known_names
    else:
        feature_names = [f"Param {i+1}" for i in range(num_features)]

    # 6. Generate Plot
    plt.figure(figsize=(10, 6))
    
    shap.summary_plot(
        shap_values, 
        test_samples.numpy(), 
        feature_names=feature_names, 
        show=False,
        plot_type="dot" 
    )
    
    # Lock the X-axis scale across all budgets
    plt.xlim(-2.5, 2.5)
    plt.xlabel("SHAP value (impact on mean standardized etch rate)", fontsize=12)
    
    plt.title(f"SHAP Feature Importance (Optuna Optimal, {budget})", fontsize=16, weight='bold', y=1.05)
    
    out_path = os.path.join(OUT_DIR, f"shap_beeswarm_{budget}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {"status": "success", "msg": f"  [+] Saved Uniform SHAP Plot for {budget} ({len(test_samples)} points)"}

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("=========================================================")
    print("[*] GENERATING UNIFORM SHAP PLOTS IN PARALLEL (6 WORKERS)")
    print("=========================================================")
    
    OPTIMAL_TRIALS = {
        "100pct": "trial_35",  
        "50.0pct": "trial_19",  
        "25.0pct": "trial_9",   
        "12.5pct": "trial_5",   
        "10.0pct": "trial_3",   
        "5.0pct":  "trial_13"   
    }
    
    SEED = 42
    tasks = [(budget, trial_id, SEED) for budget, trial_id in OPTIMAL_TRIALS.items()]
    
    print(f"[*] Launching ProcessPoolExecutor with 6 workers...\n")
    
    # Launch exactly 6 concurrent processes
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(generate_shap_plot_worker, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result["msg"])
            except Exception as e:
                print(f"  [!] Process failed: {e}")
                
    print("\n[*] All SHAP plots generated successfully!")

if __name__ == "__main__":
    main()
