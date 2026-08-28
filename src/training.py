import time
import copy
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

def train_lbfgs(model, X_train, y_train, X_val, y_val, criterion, epochs=150, lr=0.1, patience=30):
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    history = {"epoch": [], "train_loss": [], "val_loss": [], "time_sec": []}
    
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            return loss
        optimizer.step(closure)
        
        # Calculate train loss after the L-BFGS step
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train).item()
            val_loss = criterion(model(X_val), y_val).item()
            
        epoch_time = time.time() - start_time
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["time_sec"].append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: 
                break
                
    model.load_state_dict(best_weights)
    return model, history

def train_adam(model, X_train, y_train, X_val, y_val, criterion, epochs=400, lr=0.003, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1.7e-06)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    history = {"epoch": [], "train_loss": [], "val_loss": [], "time_sec": []}
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        running_train_loss = 0.0
        for b_X, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * b_X.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
            
        epoch_time = time.time() - start_time
        
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["time_sec"].append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: 
                break
                
    model.load_state_dict(best_weights)
    return model, history