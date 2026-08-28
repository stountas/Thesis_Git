import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def calculate_violations(pred):
    """Calculates non-physical positive gradients across the wafer."""
    diffs = pred[1:] - pred[:-1]
    return np.sum(diffs > 1e-4) # Threshold to ignore floating point artifacts

def export_detailed_predictions(y_actual, y_pred, X_actual, experiment_name, output_dir="results/csv_predictions"):
    """Saves a detailed CSV with row-by-row prediction arrays and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    
    for i in range(len(y_actual)):
        real = y_actual[i]
        pred = y_pred[i]
        inputs = X_actual[i]
        
        mae = mean_absolute_error(real, pred)
        mse = mean_squared_error(real, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(real, pred) if np.var(real) > 1e-5 else np.nan
        violations = calculate_violations(pred)
        
        records.append({
            "Recipe_ID": i,
            "Inputs_(P_p_Q_Vb)": f"[{inputs[0]:.1f}, {inputs[1]:.3f}, {inputs[2]:.1f}, {inputs[3]:.1f}]",
            "Real_EtchRate": "[" + ", ".join([f"{v:.2f}" for v in real]) + "]",
            "Predicted_EtchRate": "[" + ", ".join([f"{v:.2f}" for v in pred]) + "]",
            "R2_Score": round(r2, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Physics_Violations": violations
        })
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, f"{experiment_name}_detailed.csv")
    df.to_csv(csv_path, index=False)
    
    # Calculate Global Metrics
    global_r2 = r2_score(y_actual.flatten(), y_pred.flatten())
    global_mse = mean_squared_error(y_actual.flatten(), y_pred.flatten())
    global_rmse = np.sqrt(global_mse)
    global_mae = mean_absolute_error(y_actual.flatten(), y_pred.flatten())
    global_violations = np.sum(df["Physics_Violations"]) / (len(y_actual) * 9) * 100 # Violation Rate %
    
    return global_r2, global_mse, global_rmse, global_mae, global_violations