import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

def center_crop_3d(volume, crop_size):
    """Crop a 3D tensor (C, D, H, W) to the given crop_size (cd, ch, cw)."""
    _, D, H, W = volume.shape
    cd, ch, cw = crop_size

    d1 = max((D - cd) // 2, 0)
    h1 = max((H - ch) // 2, 0)
    w1 = max((W - cw) // 2, 0)

    return volume[:, d1:d1+cd, h1:h1+ch, w1:w1+cw]

class EFieldDataset(Dataset):
    def __init__(self, dadt_files, cond_files, efield_files, transform=None):
        self.dadt_files = sorted(dadt_files)
        self.cond_files = sorted(cond_files)
        self.efield_files = sorted(efield_files)
        self.transform = transform

    def __len__(self):
        return len(self.dadt_files)

    def load_nifti(self, path):
        data = nib.load(path).get_fdata()
        data = np.nan_to_num(data)
        return data.astype(np.float32)

    def __getitem__(self, idx):
        dadt = self.load_nifti(self.dadt_files[idx])
        cond = self.load_nifti(self.cond_files[idx])
        efield = self.load_nifti(self.efield_files[idx])

        # Rearrange axis: from DHWC to CDHW
        dadt = torch.from_numpy(dadt).permute(3, 0, 1, 2)  # (3, D, H, W)
        cond = torch.from_numpy(cond).unsqueeze(0)         # (1, D, H, W)
        efield = torch.from_numpy(efield).permute(3, 0, 1, 2)  # (3, D, H, W)

        crop_size = (96, 96, 96)
        dadt = center_crop_3d(dadt, crop_size)
        cond = center_crop_3d(cond, crop_size)
        efield = center_crop_3d(efield, crop_size)
        
        # Normalize inputs for better neural operator performance
        dadt = (dadt - dadt.mean()) / (dadt.std() + 1e-8)
        cond = (cond - cond.mean()) / (cond.std() + 1e-8)

        if self.transform:
            dadt, cond, efield = self.transform((dadt, cond, efield))

        return (dadt, cond), efield


def make_dataloaders(dadt_dir, cond_dir, efield_dir, batch_size=4, num_workers=4, split_ratio=0.8):
    from glob import glob
    from torch.utils.data import DataLoader, random_split

    dadt_files = glob(os.path.join(dadt_dir, '*_D.nii.gz'))
    cond_files = glob(os.path.join(cond_dir, '*_conductivity.nii.gz'))
    efield_files = glob(os.path.join(efield_dir, '*_E.nii.gz'))

    dataset = EFieldDataset(dadt_files, cond_files, efield_files)

    train_len = int(len(dataset) * split_ratio)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader


