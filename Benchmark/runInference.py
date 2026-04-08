import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
from newModel import DualBranchResNet3D  # import your model
from model import Modified3DUNet

# --- Config ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# For affine & shape
cond_path = "/content/drive/MyDrive/Colab Notebooks/data/conductivity/AF8_C5_100307_TMS_1-0001_Magstim_70mm_Fig8_scalar_conductivity.nii.gz"
dadt_path = "/content/drive/MyDrive/Colab Notebooks/data/dadt/AF8_C5_100307_TMS_1-0001_Magstim_70mm_Fig8_scalar_D.nii.gz"
ref_path = "/content/drive/MyDrive/Colab Notebooks/data/efield/AF8_C5_100307_TMS_1-0001_Magstim_70mm_Fig8_scalar_E.nii.gz"
model_path = "/content/drive/MyDrive/Colab Notebooks/checkpoints/model_epoch_30.pth"
# model_path = "/content/drive/MyDrive/Colab Notebooks/MultiCoil_MSE_epoch_900.pth.tar"
output_path = "/content/drive/MyDrive/Colab Notebooks/outputs/45epochTest.nii.gz"

# --- Load and preprocess input ---
def load_nifti_as_tensor(path, channels_last=False):
    data = nib.load(path).get_fdata()
    tensor = torch.tensor(np.nan_to_num(data), dtype=torch.float32)
    if not channels_last and tensor.ndim == 4:
        tensor = tensor.permute(3, 0, 1, 2)  # (C, D, H, W)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # (1, D, H, W)
    return tensor.unsqueeze(0)  # (1, C, D, H, W)

dadt = load_nifti_as_tensor(dadt_path).to(device)  # (1, 3, D, H, W)
cond = load_nifti_as_tensor(cond_path).to(device)  # (1, 1, D, H, W)
original_shape = dadt.shape[2:]

def pad_to_multiple(tensor, multiple=8):
    _, _, D, H, W = tensor.shape
    pad_d = (multiple - D % multiple) % multiple
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    padding = [
        pad_w // 2, pad_w - pad_w // 2,  # W
        pad_h // 2, pad_h - pad_h // 2,  # H
        pad_d // 2, pad_d - pad_d // 2   # D
    ]
    padded_tensor = F.pad(tensor, padding, mode='constant', value=0)
    return padded_tensor, padding

def unpad_tensor(tensor, padding):
    _, _, D, H, W = tensor.shape
    pw1, pw2 = padding[0], padding[1]
    ph1, ph2 = padding[2], padding[3]
    pd1, pd2 = padding[4], padding[5]
    return tensor[:, :, pd1:D - pd2, ph1:H - ph2, pw1:W - pw2]

dadt_pad, pad = pad_to_multiple(dadt)
cond_pad, _ = pad_to_multiple(cond)

# --- Load model ---
model = DualBranchResNet3D().to(device)
# model = Modified3DUNet(4, 3, 16).cuda(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
torch.cuda.empty_cache()

# --- Run inference ---
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
      pred_pad = model(dadt_pad, cond_pad)

pred = unpad_tensor(pred_pad, pad)
mask = (cond > 0).float()
pred = pred * mask

# --- Compute magnitude and save ---
def save_magnitude_map(tensor, ref_path, save_path):
    mag = torch.norm(tensor[0].cpu(), dim=0)  # (D, H, W)
    mag = mag.numpy().astype(np.float32)

    ref_img = nib.load(ref_path)
    mag_nifti = nib.Nifti1Image(mag, affine=ref_img.affine)
    nib.save(mag_nifti, save_path)
    print(f"Saved magnitude map to {save_path}")

    efield = nib.load("/content/drive/MyDrive/Colab Notebooks/data/efield/AF8_C5_100307_TMS_1-0001_Magstim_70mm_Fig8_scalar_E.nii.gz").get_fdata()
    print(f"min: {efield.min():.4e}, max: {efield.max():.4e}, mean: {efield.mean():.4e}")
    
    print("Prediction stats:")
    print(f"min: {pred.min().item():.4f}, max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}")


save_magnitude_map(pred, ref_path, output_path)
