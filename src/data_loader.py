import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def greedy_k_centers(X, n_samples, seed):
    """Greedy K-Centers algorithm for diverse geometric sampling."""
    np.random.seed(seed)
    selected_indices = [np.random.randint(0, X.shape[0])]
    min_distances = np.linalg.norm(X - X[selected_indices[0]], axis=1)

    for _ in range(1, n_samples):
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
        new_dist = np.linalg.norm(X - X[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_dist)

    return selected_indices

def load_plasma_data(filepath, fraction=1.0, method="random", seed=42):
    """
    Loads raw dataset and splits it based on budget, method, and seed.
    Returns Train, Val, Test, and Global Unseen datasets.
    """
    df = pd.read_csv(filepath, sep="\t").drop_duplicates().reset_index(drop=True)
    input_cols = ["Power", "Pressure", "Feed", "Vbias"]

    n_total = len(df)
    n_budget = max(1, int(n_total * fraction))

    # 1. Apply Sampling Strategy
    if method == "kcenters":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df[input_cols].values)
        selected_idx = greedy_k_centers(X_scaled, n_budget, seed)
    else: # Uniform Random
        np.random.seed(seed)
        selected_idx = np.random.permutation(n_total)[:n_budget].tolist()

    # Determine unused points for the global unseen universe
    unselected_idx = list(set(range(n_total)) - set(selected_idx))
    
    df_budget = df.iloc[selected_idx].reset_index(drop=True)
    df_global_unseen = df.iloc[unselected_idx].reset_index(drop=True)

    # 2. Split the selected budget into Train (70%), Val (15%), Test (15%)
    if len(df_budget) >= 3:
        df_train, df_temp = train_test_split(df_budget, test_size=0.30, random_state=seed)
        df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=seed)
    else:
        df_train, df_val, df_test = df_budget, df_budget, df_budget # Fallback for tiny budgets

    return {
        "df_train": df_train.reset_index(drop=True),
        "df_val": df_val.reset_index(drop=True),
        "df_test": df_test.reset_index(drop=True),
        "df_global_unseen": df_global_unseen
    }

def load_precomputed_splits(prefix, data_dir):
    """
    Loads pre-computed Train, Val, Test, and Global Unseen CSVs.
    Fits scalers strictly on the Train set to prevent data leakage.
    """
    train_path = os.path.join(data_dir, f"{prefix}_train.csv")
    val_path = os.path.join(data_dir, f"{prefix}_val.csv")
    test_path = os.path.join(data_dir, f"{prefix}_test.csv")
    unseen_path = os.path.join(data_dir, f"{prefix}_unseen.csv") 
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    if os.path.exists(unseen_path):
        df_unseen = pd.read_csv(unseen_path)
    else:
        df_unseen = None
    
    input_cols = ["Power", "Pressure", "Feed", "Vbias"]
    output_cols = [f"Point{i}_EtchRate" for i in range(1, 11)]
    
    scaler_X = StandardScaler()
    X_tr = torch.tensor(scaler_X.fit_transform(df_train[input_cols].values), dtype=torch.float32)
    X_va = torch.tensor(scaler_X.transform(df_val[input_cols].values), dtype=torch.float32)
    X_te = torch.tensor(scaler_X.transform(df_test[input_cols].values), dtype=torch.float32)
    
    if df_unseen is not None:
        X_unseen = torch.tensor(scaler_X.transform(df_unseen[input_cols].values), dtype=torch.float32)
        X_unseen_raw = df_unseen[input_cols].values
        y_unseen_raw = df_unseen[output_cols].values
    else:
        X_unseen, X_unseen_raw, y_unseen_raw = None, None, None
    
    y_tr_raw = df_train[output_cols].values
    p_mean = np.mean(y_tr_raw, axis=0)
    p_std = np.std(y_tr_raw, axis=0) + 1e-8
    
    y_tr = torch.tensor((y_tr_raw - p_mean) / p_std, dtype=torch.float32)
    y_va = torch.tensor((df_val[output_cols].values - p_mean) / p_std, dtype=torch.float32)
    y_te = torch.tensor((df_test[output_cols].values - p_mean) / p_std, dtype=torch.float32)
    X_mean = scaler_X.mean_
    X_std = scaler_X.scale_
    return {
        "X_tr": X_tr, "y_tr": y_tr, "X_tr_raw": df_train[input_cols].values,
        "X_va": X_va, "y_va": y_va, 
        "X_te": X_te, "y_te": y_te,
        "y_te_raw": df_test[output_cols].values,
        "X_te_raw": df_test[input_cols].values,
        "X_unseen": X_unseen, "X_unseen_raw": X_unseen_raw, "y_unseen_raw": y_unseen_raw,
        "p_mean": p_mean, "p_std": p_std,
        "X_mean": X_mean, "X_std": X_std  # <--- Make sure they are added to the dictionary here!
    }