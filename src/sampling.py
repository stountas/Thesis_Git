import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc

def k_center_greedy(X, n_samples, seed=42):
    rng = np.random.RandomState(seed)
    start_idx = rng.choice(X.shape[0])
    selected = [start_idx]
    min_distances = np.linalg.norm(X - X[start_idx], axis=1)
    for _ in range(1, n_samples):
        farthest = np.argmax(min_distances)
        selected.append(farthest)
        min_distances = np.minimum(min_distances, np.linalg.norm(X - X[farthest], axis=1))
    return np.array(selected)

def random_sampling(X, n_samples, seed=42):
    rng = np.random.RandomState(seed)
    return rng.choice(len(X), size=n_samples, replace=False)

def latin_hypercube_sampling(X, n_samples, seed=42):
    sampler = qmc.LatinHypercube(d=X.shape[1], seed=seed)
    sample_unit = sampler.random(n=n_samples)
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    lhs_pts = qmc.scale(sample_unit, xmin, xmax)
    
    selected_indices = []
    for pt in lhs_pts:
        idx = np.argmin(np.linalg.norm(X - pt, axis=1))
        selected_indices.append(idx)
    return np.unique(selected_indices)[:n_samples] # Fallback if duplicates exist

def prepare_plasma_data(filepath, fraction=0.10, method="kcenters", seed=42):
    """Performs 70/15/15 split, applies sampling strictly to training, and scales."""
    df = pd.read_csv(filepath, sep="\t").drop_duplicates().reset_index(drop=True)
    input_cols = ["Power", "Pressure", "Feed", "Vbias"]
    output_cols = [f"Point{i}_EtchRate" for i in range(1, 11)]
    
    # 1. Base 70/15/15 Split
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train, n_val = int(0.70 * len(df_shuffled)), int(0.15 * len(df_shuffled))
    
    base_train = df_shuffled.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df_shuffled.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    df_test = df_shuffled.iloc[n_train + n_val:].copy().reset_index(drop=True)
    
    # 2. Apply Data Budget (Ablation)
    budget = min(len(base_train), max(1, int(len(df_shuffled) * fraction)))
    scaler_temp = StandardScaler()
    X_train_raw = scaler_temp.fit_transform(base_train[input_cols].values)
    
    if method == "kcenters":
        idx = k_center_greedy(X_train_raw, budget, seed=seed)
    elif method == "lhs":
        idx = latin_hypercube_sampling(X_train_raw, budget, seed=seed)
    else:
        idx = random_sampling(X_train_raw, budget, seed=seed)
        
    df_train = base_train.iloc[idx].copy().reset_index(drop=True)
        
    # 3. Fit Scalers strictly on sampled Training Data
    scaler_X = StandardScaler()
    X_tr = torch.tensor(scaler_X.fit_transform(df_train[input_cols].values), dtype=torch.float32)
    X_va = torch.tensor(scaler_X.transform(df_val[input_cols].values), dtype=torch.float32)
    X_te = torch.tensor(scaler_X.transform(df_test[input_cols].values), dtype=torch.float32)
    
    y_tr_raw = df_train[output_cols].values
    p_mean, p_std = np.mean(y_tr_raw, axis=0), np.std(y_tr_raw, axis=0) + 1e-8
    
    y_tr = torch.tensor((y_tr_raw - p_mean) / p_std, dtype=torch.float32)
    y_va = torch.tensor((df_val[output_cols].values - p_mean) / p_std, dtype=torch.float32)
    
    return {
        "X_tr": X_tr, "y_tr": y_tr, 
        "X_va": X_va, "y_va": y_va, 
        "X_te": X_te, 
        "y_te_raw": df_test[output_cols].values,
        "X_te_raw": df_test[input_cols].values,
        "p_mean": p_mean, "p_std": p_std
    }