import torch
import torch.nn as nn
import torch.nn.functional as F
from physicsnemo.models.fno import FNO


class PhysicsNeMoFNOModel(nn.Module):
    def __init__(self, 
                 in_channels=4,  # 3 for dA/dt + 1 for conductivity
                 out_channels=3,  # 3 for E-field
                 modes=16,  # Fourier modes 
                 width=32,  # Hidden dimension
                 n_layers=4):
        super(PhysicsNeMoFNOModel, self).__init__()
        
        # FNO for 3D physics
        self.fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            dimension=3,  # 3D FNO
            latent_channels=width,
            decoder_layer_size=width,
            num_fno_layers=n_layers,
            num_fno_modes=modes,
            activation_fn="gelu",
            coord_features=True
        )
        
    def forward(self, dadt, cond):
        # Concatenate inputs along channel dimension
        # dadt: (B, 3, D, H, W), cond: (B, 1, D, H, W) -> (B, 4, D, H, W)
        x = torch.cat([dadt, cond], dim=1)
        
        # FNO expects input in format (B, C, D, H, W)
        out = self.fno(x)
        
        return out


# Simplified approach - focus on FNO for now
# DeepONet can be added later if needed


# Main model class (using FNO)
class PhysicsNeMoEFieldModel(nn.Module):
    def __init__(self, **kwargs):
        super(PhysicsNeMoEFieldModel, self).__init__()
        self.model = PhysicsNeMoFNOModel(**kwargs)
            
    def forward(self, dadt, cond):
        return self.model(dadt, cond)


# Legacy compatibility
DualBranchResNet3D = PhysicsNeMoEFieldModel
