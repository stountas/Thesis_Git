import torch
import torch.nn as nn

class SurrogateMLP(nn.Module):
    """The 20-20-20 MLP Architecture."""
    def __init__(self, hidden_sizes=[20, 20, 20], input_size=4, output_size=10):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.SiLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.net(x)