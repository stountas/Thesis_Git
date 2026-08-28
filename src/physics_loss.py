import torch
import torch.nn as nn

class PhysicsGuidedLoss(nn.Module):
    """Mean Squared Error + Uniform Spatial Monotonicity Penalty"""
    def __init__(self, lambda_mono=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_mono = lambda_mono
        
    def forward(self, pred, target):
        loss_data = self.mse(pred, target)
        # Penalize positive radial gradients (EtchRate should decrease outward)
        radial_diffs = pred[:, 1:] - pred[:, :-1]
        loss_mono = torch.mean(torch.relu(radial_diffs) ** 2)
        total_loss = loss_data + (self.lambda_mono * loss_mono)
        return total_loss

class SpatiallyAwarePhysicsLoss(nn.Module):
    """Advanced MSE + Spatially-Weighted Monotonicity + Jaggedness Penalty + Positivity"""
    def __init__(self, tau_i, W_i, W_mono_i, p_mean, p_std, lambda_mono=1.0, lambda_smooth=1.0, lambda_pos=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        
        self.lambda_mono = lambda_mono
        self.lambda_smooth = lambda_smooth
        self.lambda_pos = lambda_pos
        
        self.register_buffer('tau_i', tau_i)
        self.register_buffer('W_i', W_i)
        self.register_buffer('W_mono_i', W_mono_i)
        
        # Register normalization parameters as buffers so they stay on the correct device
        self.register_buffer('p_mean', torch.tensor(p_mean, dtype=torch.float32))
        self.register_buffer('p_std', torch.tensor(p_std, dtype=torch.float32))

    def forward(self, pred, target):
        loss_data = self.mse(pred, target)
        
        # 1. Spatially-Weighted Monotonicity Penalty
        first_diffs = pred[:, 1:] - pred[:, :-1]
        excess_mono = torch.relu(first_diffs)
        weighted_mono = excess_mono * self.W_mono_i
        loss_mono = torch.mean(weighted_mono ** 2)
        
        # 2. Spatially-Aware Smoothness
        second_diffs = first_diffs[:, 1:] - first_diffs[:, :-1]
        excess_jaggedness = torch.relu(torch.abs(second_diffs) - self.tau_i)
        weighted_jaggedness = excess_jaggedness * self.W_i
        loss_smooth = torch.mean(weighted_jaggedness ** 2)
        
        # 3. Positivity Penalty (Prevents impossible negative etch rates)
        # Unscale predictions to the original physical domain before penalizing
        unscaled_pred = (pred * self.p_std) + self.p_mean
        loss_pos = torch.mean(torch.relu(-unscaled_pred) ** 2)
        
        total_loss = loss_data + (self.lambda_mono * loss_mono) + (self.lambda_smooth * loss_smooth) + (self.lambda_pos * loss_pos)
        return total_loss