import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from newModel import PhysicsNeMoEFieldModel
from newDataset import make_dataloaders

class WeightedMSELoss(nn.Module):
  def __init__(self, weight=None):
    super(WeightedMSELoss, self).__init__()
    self.register_buffer('weight', weight)

  def forward(self, input, target):
    w = target**2
    w = .2*w.sum(dim=1)+1
    w = w.unsqueeze(1)
    w = torch.broadcast_to(w, target.size())
    return F.mse_loss(w*input, w*target, reduction="mean")


# Uncomment to use physics-informed loss:
# from PhysicsLoss import PhysicsInformedLoss

# PhysicsNeMo FNO handles variable input sizes naturally - no padding needed

def train_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for (dadt, cond), target in tqdm(loader, desc="Training", leave=False):
        dadt, cond, target = dadt.to(device), cond.to(device), target.to(device)

        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            output = model(dadt, cond)
            # Pass additional physics inputs if using PhysicsInformedLoss
            if hasattr(loss_fn, 'physics_weight'):
                loss = loss_fn(output, target, dadt=dadt, conductivity=cond)
            else:
                loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * dadt.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (dadt, cond), target in tqdm(loader, desc="Validation", leave=False):
            dadt, cond, target = dadt.to(device), cond.to(device), target.to(device)

            with autocast(device_type='cuda'):
                output = model(dadt, cond)
                # Pass additional physics inputs if using PhysicsInformedLoss
                if hasattr(loss_fn, 'physics_weight'):
                    loss = loss_fn(output, target, dadt=dadt, conductivity=cond)
                else:
                    loss = loss_fn(output, target)

            running_loss += loss.item() * dadt.size(0)

    return running_loss / len(loader.dataset)


def train_model(
    dadt_dir, cond_dir, efield_dir,
    epochs=20, batch_size=4, lr=1e-4, weight_decay=1e-5,
    checkpoint_dir="checkpoints", device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, val_loader = make_dataloaders(
        dadt_dir=dadt_dir, cond_dir=cond_dir, efield_dir=efield_dir,
        batch_size=batch_size, num_workers=2
    )

    # Initialize PhysicsNeMo FNO model
    model = PhysicsNeMoEFieldModel(
        in_channels=4,  # 3 dA/dt + 1 conductivity
        out_channels=3,  # 3 E-field components
        modes=16,  # Fourier modes
        width=64,  # Increased width for better performance
        n_layers=4
    ).to(device)
    # Choose loss function:
    loss_fn = WeightedMSELoss().to(device)
    
    # To use physics-informed loss instead, uncomment these lines:
    # loss_fn = PhysicsInformedLoss(
    #     data_weight=1.0,      # Weight for data fidelity
    #     physics_weight=0.01,  # Weight for physics constraints (start small)
    #     dx=1.0               # Spatial resolution (adjust based on voxel size)
    # ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save checkpoint if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print("✅ Saved best model")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth"))

if __name__ == "__main__":
    train_model(
        dadt_dir="/content/drive/MyDrive/Colab Notebooks/data/dadt",
        cond_dir="/content/drive/MyDrive/Colab Notebooks/data/conductivity",
        efield_dir="/content/drive/MyDrive/Colab Notebooks/data/efield",
        epochs=100,
        batch_size=8  # Reduced batch size due to FNO memory requirements
    )
