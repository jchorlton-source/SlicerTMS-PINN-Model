import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from contextlib import nullcontext
from tqdm import tqdm

from newModel import DualBranchResNet3D
from newDataset import make_dataloaders


# -----------------------------
# Composite loss with explicit mask support
# -----------------------------
class CompositeLoss(nn.Module):
    """
    L = w_mae * |(pred*mask) - (tgt*mask)|
      + ramp_t * (w_grad * |∇(pred*mask) - ∇(tgt*mask)| + w_tv * TV(pred*mask))
      + w_bg * |pred * (1 - mask)|
      + w_mse * MSE((pred*mask), (tgt*mask))

    If no mask is provided, an auto-mask is built from target magnitude.
    """
    def __init__(self, w_mae=1.0, w_grad=0.10, w_tv=0.002, w_bg=0.20, w_mse=0.20,
                 warmup_epochs=10, auto_mask_threshold=1e-6, use_auto_mask=True):
        super().__init__()
        self.w_mae               = w_mae
        self.w_grad_max          = w_grad
        self.w_tv_max            = w_tv
        self.w_bg_max            = w_bg
        self.w_mse_max           = w_mse
        self.warmup_epochs       = warmup_epochs
        self.auto_mask_threshold = auto_mask_threshold
        self.use_auto_mask       = use_auto_mask
        self._epoch              = 0

    def set_epoch(self, e: int):
        self._epoch = int(e)

    @staticmethod
    def _grad3(x):
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz

    def _weights(self):
        e = self._epoch
        A, B = self.warmup_epochs, 2 * self.warmup_epochs
        if e <= A:
            return (self.w_mae, 0.0, 0.0, self.w_bg_max, 0.0)
        elif e <= B:
            t = (e - A) / max(1, A)
            return (self.w_mae,
                    t * self.w_grad_max,
                    t * self.w_tv_max,
                    (1.0 - 0.5 * t) * self.w_bg_max,
                    t * self.w_mse_max)
        else:
            return (self.w_mae,
                    0.5 * self.w_grad_max,
                    0.5 * self.w_tv_max,
                    0.5 * self.w_bg_max,
                    self.w_mse_max)

    def forward(self, pred, target, mask: torch.Tensor | None = None):
        if mask is None:
            if self.use_auto_mask:
                mag  = target.abs().sum(dim=1, keepdim=True)
                mask = (mag > self.auto_mask_threshold).to(pred.dtype)
            else:
                mask = torch.ones_like(pred[:, :1])

        mask = mask.to(dtype=pred.dtype)
        w_mae, w_grad, w_tv, w_bg, w_mse = self._weights()

        pred_in = pred * mask
        tgt_in  = target * mask

        l_mae = F.l1_loss(pred_in, tgt_in)

        if w_grad or w_tv:
            px, py, pz = self._grad3(pred_in)
            tx, ty, tz = self._grad3(tgt_in)
            l_grad = (px-tx).abs().mean() + (py-ty).abs().mean() + (pz-tz).abs().mean()
            dx, dy, dz = self._grad3(pred_in)
            l_tv   = dx.abs().mean() + dy.abs().mean() + dz.abs().mean()
        else:
            l_grad = l_tv = pred.new_tensor(0.0)

        inv   = (1.0 - mask)
        l_bg  = (pred * inv).abs().mean() if w_bg  else pred.new_tensor(0.0)
        l_mse = F.mse_loss(pred_in, tgt_in) if w_mse else pred.new_tensor(0.0)

        return w_mae*l_mae + w_grad*l_grad + w_tv*l_tv + w_bg*l_bg + w_mse*l_mse


# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


# -----------------------------
# Batch unpack
# -----------------------------
def unpack_batch(batch):
    (inputs, target) = batch
    if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
        dadt, cond, mask = inputs
    else:
        dadt, cond = inputs
        mask = None
    return dadt, cond, target, mask


# -----------------------------
# Train one epoch
# -----------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    ema=None, max_grad_norm=1.0):
    model.train()
    running = 0.0
    amp_ctx = autocast(device_type='cuda', dtype=torch.float16) \
              if device.startswith('cuda') else nullcontext()

    for batch in tqdm(loader, desc="Training", leave=False):
        dadt, cond, target, mask = unpack_batch(batch)
        dadt   = dadt.to(device)
        cond   = cond.to(device)
        target = target.to(device)
        mask   = None if mask is None else mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            out  = model(dadt, cond)
            loss = loss_fn(out, target, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        running += loss.item() * dadt.size(0)

    return running / len(loader.dataset)


# -----------------------------
# Validate
# -----------------------------
@torch.no_grad()
def validate(model, loader, loss_fn, device, ema=None):
    """
    Validate using EMA weights if available.
    Live model weights are saved and restored so training is unaffected.
    """
    if ema is not None:
        # Save live weights before overwriting with EMA
        live_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    model.eval()
    running = 0.0
    amp_ctx = autocast(device_type='cuda', dtype=torch.float16) \
              if device.startswith('cuda') else nullcontext()

    for batch in tqdm(loader, desc="Validation", leave=False):
        dadt, cond, target, mask = unpack_batch(batch)
        dadt   = dadt.to(device)
        cond   = cond.to(device)
        target = target.to(device)
        mask   = None if mask is None else mask.to(device)

        with amp_ctx:
            out  = model(dadt, cond)
            loss = loss_fn(out, target, mask)

        running += loss.item() * dadt.size(0)

    # Restore live weights so training continues correctly
    if ema is not None:
        model.load_state_dict(live_state)

    return running / len(loader.dataset)


# -----------------------------
# Orchestrator
# -----------------------------
def train_model(
    sim_output_dir,
    epochs=50,
    batch_size=8,
    base_lr=2e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    checkpoint_dir="checkpoints",
    resume_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader = make_dataloaders(
        sim_output_dir=sim_output_dir,
        batch_size=batch_size,
        num_workers=8
    )

    model = DualBranchResNet3D().to(device)

    loss_fn = CompositeLoss(
        w_mae=1.0,
        w_grad=0.10,
        w_tv=0.002,
        w_bg=0.20,
        w_mse=0.20,
        warmup_epochs=10,
        use_auto_mask=True,
        auto_mask_threshold=1e-6
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        ],
        milestones=[warmup_epochs]
    )

    scaler      = GradScaler(enabled=(device.startswith('cuda')))
    ema         = EMA(model, decay=0.999)
    best_val    = float("inf")
    start_epoch = 1

    # Resume from checkpoint if provided and exists
    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        ema.shadow  = ckpt['ema_shadow']
        best_val    = ckpt.get('best_val', float("inf"))
        start_epoch = ckpt['epoch'] + 1
        print(f"[INFO] Resumed from epoch {ckpt['epoch']} | "
              f"Best val so far: {best_val:.6f}")
    else:
        print("[INFO] Starting training from scratch")

    for epoch in range(start_epoch, epochs + 1):
        loss_fn.set_epoch(epoch)
        print(f"\nEpoch {epoch}/{epochs}  "
              f"(lr: {optimizer.param_groups[0]['lr']:.2e})")

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer,
            scaler, device, ema=ema, max_grad_norm=1.0
        )
        val_loss = validate(model, val_loader, loss_fn, device, ema=ema)

        print(f"  Train Loss: {train_loss:.6f} | Val Loss (EMA): {val_loss:.6f}")

        # Save best EMA weights
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                ema.shadow,
                os.path.join(checkpoint_dir, "best_model_ema.pth")
            )
            print("  Saved best EMA weights")

        # Save resume checkpoint (overwrites each epoch — only latest needed)
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'ema_shadow':           ema.shadow,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val':             best_val,
        }, os.path.join(checkpoint_dir, "resume_checkpoint.pth"))

        # Save per-epoch weights for later analysis
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        )

        scheduler.step()

    print(f"\nDone. Best Val (EMA): {best_val:.6f}")


if __name__ == "__main__":
    SCRATCH = os.environ.get('MYSCRATCH')
    base    = os.path.join(SCRATCH, 'Thesis/SlicerTMS-PINN-Model/benchmark/BenchmarkModel')

    train_model(
        sim_output_dir = os.path.join(SCRATCH, 'Thesis/SlurmScripts/simOutput'),
        checkpoint_dir = os.path.join(base, 'checkpoints'),
        resume_path    = os.path.join(base, 'checkpoints/resume_checkpoint.pth'),
        epochs         = 50,
        batch_size     = 8,
        warmup_epochs  = 5,
    )
