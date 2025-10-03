import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    def __init__(self, data_weight=1.0, physics_weight=0.01, dx=1.0):
        super(PhysicsInformedLoss, self).__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.dx = dx  # spatial resolution
        
    def gradient_3d(self, field, dim):
        """Compute gradient using finite differences"""
        if dim == 0:  # x direction
            grad = torch.zeros_like(field)
            grad[:, :, 1:, :, :] = field[:, :, 1:, :, :] - field[:, :, :-1, :, :]
            grad[:, :, 0, :, :] = grad[:, :, 1, :, :]  # boundary
        elif dim == 1:  # y direction  
            grad = torch.zeros_like(field)
            grad[:, :, :, 1:, :] = field[:, :, :, 1:, :] - field[:, :, :, :-1, :]
            grad[:, :, :, 0, :] = grad[:, :, :, 1, :]  # boundary
        elif dim == 2:  # z direction
            grad = torch.zeros_like(field)
            grad[:, :, :, :, 1:] = field[:, :, :, :, 1:] - field[:, :, :, :, :-1]
            grad[:, :, :, :, 0] = grad[:, :, :, :, 1]  # boundary
        return grad / self.dx
    
    def divergence_loss(self, e_field):
        """Enforce ∇·E = 0 in source-free regions"""
        ex, ey, ez = e_field[:, 0:1], e_field[:, 1:2], e_field[:, 2:3]
        
        div_ex = self.gradient_3d(ex, 0)
        div_ey = self.gradient_3d(ey, 1)  
        div_ez = self.gradient_3d(ez, 2)
        
        divergence = div_ex + div_ey + div_ez
        return torch.mean(divergence**2)
    
    def curl_consistency_loss(self, e_field, dadt):
        """Enforce ∇xE = -∂A/∂t (Faraday's law)"""
        ex, ey, ez = e_field[:, 0:1], e_field[:, 1:2], e_field[:, 2:3]

        # Compute curl of E-field
        curl_x = self.gradient_3d(ez, 1) - self.gradient_3d(ey, 2)
        curl_y = self.gradient_3d(ex, 2) - self.gradient_3d(ez, 0)
        curl_z = self.gradient_3d(ey, 0) - self.gradient_3d(ex, 1)
        
        curl_e = torch.cat([curl_x, curl_y, curl_z], dim=1)
        
        # Should equal -dA/dt
        faraday_residual = curl_e + dadt
        return torch.mean(faraday_residual**2)
    
    def conductivity_consistency_loss(self, e_field, conductivity):
        """Higher conductivity regions should have lower E-field magnitude"""
        e_magnitude = torch.sqrt(torch.sum(e_field**2, dim=1, keepdim=True) + 1e-8)
        # Inverse relationship: high conductivity → low E-field
        consistency = e_magnitude * conductivity
        return torch.mean(consistency)
    
    def forward(self, prediction, target, dadt=None, conductivity=None):
        # Data fidelity loss
        data_loss = F.mse_loss(prediction, target)
        
        total_loss = self.data_weight * data_loss
        
        if self.physics_weight > 0:
            physics_losses = []
            
            # Divergence loss (∇·E = 0)
            div_loss = self.divergence_loss(prediction)
            physics_losses.append(div_loss)
            
            # Faraday's law (∇×E = -∂A/∂t)
            if dadt is not None:
                curl_loss = self.curl_consistency_loss(prediction, dadt)
                physics_losses.append(curl_loss)
            
            # Conductivity consistency  
            if conductivity is not None:
                cond_loss = self.conductivity_consistency_loss(prediction, conductivity)
                physics_losses.append(0.1 * cond_loss)  # Lower weight
            
            physics_loss = sum(physics_losses)
            total_loss += self.physics_weight * physics_loss
        
        return total_loss