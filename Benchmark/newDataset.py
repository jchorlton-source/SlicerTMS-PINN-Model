import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# Fixed filenames inside every subject_volumes/ folder
DADT_FILENAME   = 'ernie_TMS_1-0001_Magstim_70mm_Fig8_scalar_D.nii.gz'
COND_FILENAME   = 'ernie_TMS_1-0001_Magstim_70mm_Fig8_scalar_conductivity.nii.gz'
EFIELD_FILENAME = 'ernie_TMS_1-0001_Magstim_70mm_Fig8_scalar_E.nii.gz'


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
        self.dadt_files   = dadt_files
        self.cond_files   = cond_files
        self.efield_files = efield_files
        self.transform    = transform

        assert len(self.dadt_files) == len(self.cond_files) == len(self.efield_files), \
            f"File list length mismatch: dadt={len(self.dadt_files)}, " \
            f"cond={len(self.cond_files)}, efield={len(self.efield_files)}"

    def __len__(self):
        return len(self.dadt_files)

    def load_nifti(self, path):
        data = nib.load(path).get_fdata()
        data = np.nan_to_num(data)
        return data.astype(np.float32)

    def __getitem__(self, idx):
        dadt   = self.load_nifti(self.dadt_files[idx])
        cond   = self.load_nifti(self.cond_files[idx])
        efield = self.load_nifti(self.efield_files[idx])

        # Rearrange axes: from (D, H, W, C) to (C, D, H, W)
        dadt   = torch.tensor(dadt).permute(3, 0, 1, 2)         # (3, D, H, W)
        cond   = torch.tensor(np.squeeze(cond)).unsqueeze(0)     # (1, D, H, W)
        efield = torch.tensor(efield).permute(3, 0, 1, 2)       # (3, D, H, W)

        # Center crop to fixed 128^3
        crop_size = (128, 128, 128)
        dadt   = center_crop_3d(dadt,   crop_size)
        cond   = center_crop_3d(cond,   crop_size)
        efield = center_crop_3d(efield, crop_size)

        if self.transform:
            dadt, cond, efield = self.transform((dadt, cond, efield))

        return (dadt, cond), efield


def make_dataloaders(sim_output_dir, batch_size=4, num_workers=8, split_ratio=0.8):
    """
    Walks simOutput/<idx>_<pos>_<dir>/subject_volumes/ for each simulation
    and collects the fixed-name dadt, cond, and efield NIfTI files.

    Expected structure:
        simOutput/
            0_Fp1_Fz/
                subject_volumes/
                    ernie_TMS_1-0001_Magstim_70mm_Fig8_scalar_D.nii.gz
                    ernie_TMS_1-0001_Magstim_70mm_Fig8_scalar_conductivity.nii.gz
                    ernie_TMS_1-0001_Magstim_70mm_Fig8_scalar_E.nii.gz
            1_Fp1_AF3/
                subject_volumes/
                    ...
    """
    if not os.path.isdir(sim_output_dir):
        raise FileNotFoundError(f"simOutput directory not found: {sim_output_dir}")

    matched_dadt, matched_cond, matched_efield = [], [], []
    missing = []

    subdirs = sorted(os.listdir(sim_output_dir))

    for subdir in subdirs:
        vol_dir = os.path.join(sim_output_dir, subdir, 'subject_volumes')

        dadt_path   = os.path.join(vol_dir, DADT_FILENAME)
        cond_path   = os.path.join(vol_dir, COND_FILENAME)
        efield_path = os.path.join(vol_dir, EFIELD_FILENAME)

        if (os.path.exists(dadt_path) and
                os.path.exists(cond_path) and
                os.path.exists(efield_path)):
            matched_dadt.append(dadt_path)
            matched_cond.append(cond_path)
            matched_efield.append(efield_path)
        else:
            missing.append(subdir)

    if missing:
        print(f"[WARNING] {len(missing)} simulations missing files. "
              f"First few: {missing[:5]}")

    print(f"[INFO] Matched {len(matched_dadt)} complete samples "
          f"({len(missing)} skipped)")

    if len(matched_dadt) == 0:
        raise RuntimeError(
            f"No complete samples found in {sim_output_dir}. "
            f"Check that subject_volumes/ folders exist and filenames match."
        )

    dataset   = EFieldDataset(matched_dadt, matched_cond, matched_efield)
    train_len = int(len(dataset) * split_ratio)
    val_len   = len(dataset) - train_len

    # Fixed seed ensures same split every run
    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[INFO] Train samples: {train_len} | Val samples: {val_len}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader
