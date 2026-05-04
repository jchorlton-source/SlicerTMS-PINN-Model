import os
import numpy as np
import torch
import nibabel as nib

from newModel import DualBranchResNet3D
from newDataset import make_dataloaders

# --- Config ---
SCRATCH = os.environ['MYSCRATCH']
SIM_DIR = os.path.join(SCRATCH, 'simOutput')
CKPT    = os.path.join(SCRATCH, 'benchmark/checkpoints/best_model_ema.pth')
OUT_DIR = os.path.join(SCRATCH, 'benchmark/sanity_check')
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda:0')

# --- Build model and load EMA weights ---
model = DualBranchResNet3D(base_channels=16).to(device)
state = torch.load(CKPT, map_location=device)
# best_model_ema.pth is just the EMA shadow dict, no wrapper
model.load_state_dict(state, strict=True)
model.eval()
print(f"Loaded checkpoint: {CKPT}")

# --- Get one validation sample (no DDP, single process) ---
_, val_loader, _ = make_dataloaders(
    sim_output_dir=SIM_DIR,
    batch_size=1,
    num_workers=0,
    distributed=False,
)

(dadt, cond), target = next(iter(val_loader))
dadt   = dadt.to(device)
cond   = cond.to(device)
target = target.to(device)

print(f"dadt   shape: {tuple(dadt.shape)}   range: [{dadt.min():.3e}, {dadt.max():.3e}]")
print(f"cond   shape: {tuple(cond.shape)}   range: [{cond.min():.3e}, {cond.max():.3e}]")
print(f"target shape: {tuple(target.shape)} range: [{target.min():.3e}, {target.max():.3e}]")

# --- Predict ---
with torch.no_grad():
    pred = model(dadt, cond)
print(f"pred   shape: {tuple(pred.shape)}   range: [{pred.min():.3e}, {pred.max():.3e}]")

# --- Per-sample relative L2 (sanity-check it matches your training metric) ---
mag_target = target.abs().sum(dim=1, keepdim=True)
mask       = (mag_target > 1e-6).to(pred.dtype)
num = ((pred - target) * mask).pow(2).sum().item()
den = (target * mask).pow(2).sum().item()
rel_l2 = (num / den) ** 0.5
print(f"\nThis sample's rel L2: {rel_l2:.4f}")
print(f"Mask coverage: {mask.float().mean().item():.3f} of voxels")

# --- Compute magnitudes (|E|) for easy viewing ---
target_mag = target.norm(dim=1).squeeze(0).cpu().numpy()   # (D, H, W)
pred_mag   = pred.norm(dim=1).squeeze(0).cpu().numpy()
diff_mag   = (pred - target).norm(dim=1).squeeze(0).cpu().numpy()
dadt_mag   = dadt.norm(dim=1).squeeze(0).cpu().numpy()
cond_vol   = cond.squeeze().cpu().numpy()

# --- Save as NIfTIs for viewing in your favourite viewer ---
affine = np.eye(4)  # identity affine — fine for relative viewing
def save(arr, name):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine),
             os.path.join(OUT_DIR, name))

save(target_mag, "target_magnitude.nii.gz")
save(pred_mag,   "pred_magnitude.nii.gz")
save(diff_mag,   "abs_diff_magnitude.nii.gz")
save(dadt_mag,   "input_dadt_magnitude.nii.gz")
save(cond_vol,   "input_conductivity.nii.gz")

# --- Also save individual E-field components ---
target_np = target.squeeze(0).cpu().numpy()  # (3, D, H, W)
pred_np   = pred.squeeze(0).cpu().numpy()
for i, axis in enumerate(['x', 'y', 'z']):
    save(target_np[i], f"target_E{axis}.nii.gz")
    save(pred_np[i],   f"pred_E{axis}.nii.gz")

print(f"\nSaved 11 NIfTIs to {OUT_DIR}")
print("View them with fsleyes, ITK-SNAP, or 3D Slicer.")

# --- Slice comparison PNG ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
slices = [40, 64, 88]
vmax = max(target_mag.max(), pred_mag.max())

for col, sl in enumerate(slices):
    axes[0, col].imshow(target_mag[sl], cmap='hot', vmin=0, vmax=vmax)
    axes[0, col].set_title(f"Target |E|, z={sl}")
    axes[1, col].imshow(pred_mag[sl],   cmap='hot', vmin=0, vmax=vmax)
    axes[1, col].set_title(f"Pred |E|, z={sl}")
    axes[2, col].imshow(diff_mag[sl],   cmap='hot', vmin=0, vmax=vmax * 0.5)
    axes[2, col].set_title(f"|Diff|, z={sl}")
    for r in range(3):
        axes[r, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "slices_comparison.png"), dpi=120)
print(f"Saved slice comparison to {OUT_DIR}/slices_comparison.png")

