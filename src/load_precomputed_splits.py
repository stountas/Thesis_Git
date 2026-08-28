import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def k_center_greedy(X, n_samples, seed=42):
    """Spatial Core-Set Selection (K-Centers) for data ablation."""
    rng = np.random.RandomState(seed)
    start_idx = rng.choice(X.shape[0])
    selected = [start_idx]
    
    if n_samples == 1:
        return np.array(selected)
        
    min_distances = np.linalg.norm(X - X[start_idx], axis=1)
    
    for _ in range(1, n_samples):
        farthest = np.argmax(min_distances)
        selected.append(farthest)
        min_distances = np.minimum(min_distances, np.linalg.norm(X - X[farthest], axis=1))
    return np.array(selected)

def load_plasma_data(filepath, fraction=1.0, method="kcenters", seed=42):
    df = pd.read_csv(filepath, sep="\t").drop_duplicates().reset_index(drop=True)
    
    input_cols = ["Power", "Pressure", "Feed", "Vbias"]
    output_cols = [f"Point{i}_EtchRate" for i in range(1, 11)]
    
    # 1. Total DoE Budget based on fraction of the whole universe (1513 points)
    total_budget = max(1, int(len(df) * fraction))
    
    # 2. 70/15/15 Split INSIDE the Budget
    n_train = max(1, int(0.70 * total_budget))
    n_val = max(1, int(0.15 * total_budget))
    n_test = max(1, total_budget - n_train - n_val)
    
    # 3. Sample Training Points from the ENTIRE parameter space
    if method == "kcenters":
        scaler_temp = StandardScaler()
        X_scaled = scaler_temp.fit_transform(df[input_cols].values)
        idx_train = k_center_greedy(X_scaled, n_train, seed=seed)
    else:
        rng_tr = np.random.RandomState(seed)
        idx_train = rng_tr.choice(len(df), size=n_train, replace=False)
        
    df_train = df.iloc[idx_train].copy().reset_index(drop=True)
    
    # 4. Remove selected train points from the universe
    df_rest = df.drop(index=idx_train).reset_index(drop=True)
    
    # 5. Sample Validation Points from remainder
    rng_va = np.random.RandomState(seed + 1)
    idx_val = rng_va.choice(len(df_rest), size=n_val, replace=False)
    df_val = df_rest.iloc[idx_val].copy().reset_index(drop=True)
    df_rest = df_rest.drop(index=idx_val).reset_index(drop=True)
    
    # 6. Sample Test Points from remainder
    rng_te = np.random.RandomState(seed + 2)
    idx_test = rng_te.choice(len(df_rest), size=n_test, replace=False)
    df_test = df_rest.iloc[idx_test].copy().reset_index(drop=True)
    
    # 7. The remaining points form the Global Unseen Universe
    df_global_unseen = df_rest.drop(index=idx_test).reset_index(drop=True)
    
    # 8. Strict Scaling based ONLY on Training Data to prevent leakage
    scaler_X = StandardScaler()
    X_tr = torch.tensor(scaler_X.fit_transform(df_train[input_cols].values), dtype=torch.float32)
    X_va = torch.tensor(scaler_X.transform(df_val[input_cols].values), dtype=torch.float32)
    X_te = torch.tensor(scaler_X.transform(df_test[input_cols].values), dtype=torch.float32)
    
    y_tr_raw = df_train[output_cols].values
    p_mean, p_std = np.mean(y_tr_raw, axis=0), np.std(y_tr_raw, axis=0) + 1e-8
    
    y_tr = torch.tensor((y_tr_raw - p_mean) / p_std, dtype=torch.float32)
    y_va = torch.tensor((df_val[output_cols].values - p_mean) / p_std, dtype=torch.float32)
    y_te = torch.tensor((df_test[output_cols].values - p_mean) / p_std, dtype=torch.float32)
    
    # Safely transform unseen data if it exists
    if not df_global_unseen.empty:
        X_unseen = torch.tensor(scaler_X.transform(df_global_unseen[input_cols].values), dtype=torch.float32)
        y_unseen = torch.tensor((df_global_unseen[output_cols].values - p_mean) / p_std, dtype=torch.float32)
    else:
        X_unseen, y_unseen = None, None
    
    return {
        "X_tr": X_tr, "y_tr": y_tr, 
        "X_va": X_va, "y_va": y_va, 
        "X_te": X_te, "y_te": y_te,
        "X_unseen": X_unseen, "y_unseen": y_unseen,
        "y_te_raw": df_test[output_cols].values,
        "X_te_raw": df_test[input_cols].values,
        "y_unseen_raw": df_global_unseen[output_cols].values if not df_global_unseen.empty else None,
        "X_unseen_raw": df_global_unseen[input_cols].values if not df_global_unseen.empty else None,
        "p_mean": p_mean, "p_std": p_std,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "df_global_unseen": df_global_unseen
    }