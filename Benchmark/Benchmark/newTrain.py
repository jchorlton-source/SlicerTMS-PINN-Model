import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR,
)
from tqdm import tqdm

# H100-friendly precision settings — must be set before any CUDA work
torch.backends.cudnn.benchmark         = True
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True
torch.set_float32_matmul_precision('high')

from newModel import DualBranchResNet3D
from newDataset import make_dataloaders


history = {
    "epoch": [], "train_loss": [], "val_loss": [],
    "rel_l2": [], "epoch_time": [], "lr": [],
}


# -----------------------------
# Relative L2 (DDP-reduceable)
# -----------------------------
def relative_l2_components(pred, target, mask=None):
    if mask is not None:
        pred   = pred * mask
        target = target * mask
    num = torch.sum((pred - target) ** 2)
    den = torch.sum(target ** 2) + 1e-12
    return num, den


# -----------------------------
# Composite loss with explicit mask support
#
# Set warmup_epochs=0 to disable the per-epoch weight schedule and use the
# given weights as constants from epoch 1. That's the simpler "benchmark"
# configuration and is what we use for the base_channels=32 run.
# -----------------------------
class CompositeLoss(nn.Module):
    """
    L = w_mae * masked_mean(|pred - tgt|)
      + w_grad * masked_mean(|grad(pred) - grad(tgt)|)
      + w_tv   * masked_mean(|grad(pred)|)
      + w_bg   * mean(|pred * (1 - mask)|)
      + w_mse  * masked_mean((pred - tgt)^2)

    Masked means divide only by foreground voxel count (per channel),
    so the loss magnitude is independent of how much of the volume is brain.
    """

    def __init__(self, w_mae=1.0, w_grad=0.10, w_tv=0.002, w_bg=0.15, w_mse=1.0,
                 warmup_epochs=0, auto_mask_threshold=1e-6, use_auto_mask=True):
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

    @staticmethod
    def _masked_mean(x, mask, channels):
        # x:    (B, C, D, H, W)
        # mask: (B, 1, D, H, W) — broadcast over channels
        denom = mask.sum() * channels + 1e-8
        return (x * mask).sum() / denom

    def _weights(self):
        # warmup_epochs <= 0 means "no schedule" — return constant weights.
        if self.warmup_epochs <= 0:
            return (self.w_mae,
                    self.w_grad_max,
                    self.w_tv_max,
                    self.w_bg_max,
                    self.w_mse_max)

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
        C    = pred.shape[1]
        w_mae, w_grad, w_tv, w_bg, w_mse = self._weights()

        pred_in = pred * mask
        tgt_in  = target * mask

        l_mae = self._masked_mean((pred_in - tgt_in).abs(), mask, C)

        if w_grad or w_tv:
            px, py, pz = self._grad3(pred_in)
            tx, ty, tz = self._grad3(tgt_in)
            # Build a slightly-shrunk mask for each grad direction so we
            # only count foreground voxels in the masked mean. We use the
            # min of adjacent mask values, which is just an AND for {0,1}
            # masks — keeps the math correct for soft masks too.
            mx = torch.minimum(mask[:, :, 1:, :, :], mask[:, :, :-1, :, :])
            my = torch.minimum(mask[:, :, :, 1:, :], mask[:, :, :, :-1, :])
            mz = torch.minimum(mask[:, :, :, :, 1:], mask[:, :, :, :, :-1])

            l_grad = (
                self._masked_mean((px - tx).abs(), mx, C)
                + self._masked_mean((py - ty).abs(), my, C)
                + self._masked_mean((pz - tz).abs(), mz, C)
            )
            # Reuse px/py/pz — pred_in didn't change, so a second _grad3 call
            # would compute the exact same thing.
            l_tv = (
                self._masked_mean(px.abs(), mx, C)
                + self._masked_mean(py.abs(), my, C)
                + self._masked_mean(pz.abs(), mz, C)
            )
        else:
            l_grad = pred.new_tensor(0.0)
            l_tv   = pred.new_tensor(0.0)

        if w_bg:
            inv  = 1.0 - mask
            # Background loss is naturally over (1 - mask), so use a regular
            # mean weighted by how much of the volume is background.
            bg_denom = inv.sum() * C + 1e-8
            l_bg = (pred.abs() * inv).sum() / bg_denom
        else:
            l_bg = pred.new_tensor(0.0)

        if w_mse:
            l_mse = self._masked_mean((pred_in - tgt_in) ** 2, mask, C)
        else:
            l_mse = pred.new_tensor(0.0)

        return w_mae * l_mae + w_grad * l_grad + w_tv * l_tv + w_bg * l_bg + w_mse * l_mse


# -----------------------------
# EMA — uses a real model copy instead of swapping state dicts at validate
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        m = model.module if hasattr(model, 'module') else model
        # Maintain shadow weights as a flat dict; keep them on the same device.
        self.shadow = {k: v.detach().clone() for k, v in m.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        m = model.module if hasattr(model, 'module') else model
        for k, v in m.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # buffers like int counters — just copy the latest value
                self.shadow[k].copy_(v.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        m = model.module if hasattr(model, 'module') else model
        m.load_state_dict(self.shadow, strict=False)


# -----------------------------
# Batch unpack
# -----------------------------
def unpack_batch(batch):
    inputs, target = batch
    if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
        dadt, cond, mask = inputs
    else:
        dadt, cond = inputs
        mask = None
    return dadt, cond, target, mask


def _to_device_cl(t, device):
    """Move tensor to device with channels_last_3d layout, async."""
    if t is None:
        return None
    return t.to(device, memory_format=torch.channels_last_3d, non_blocking=True)


# -----------------------------
# Train one epoch
# -----------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    ema=None, max_grad_norm=1.0, is_master=True):
    model.train()

    # Use bf16 on H100 — same speed as fp16, full fp32 exponent range,
    # no need for GradScaler.
    amp_ctx = (autocast(device_type='cuda', dtype=torch.bfloat16)
               if 'cuda' in str(device) else nullcontext())

    iterator = tqdm(loader, desc="Training", leave=False) if is_master else loader

    # Local accumulators for a globally-correct mean train loss.
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)

    m_unwrapped = model.module if hasattr(model, 'module') else model

    for batch in iterator:
        dadt, cond, target, mask = unpack_batch(batch)
        dadt   = _to_device_cl(dadt,   device)
        cond   = _to_device_cl(cond,   device)
        target = _to_device_cl(target, device)
        mask   = _to_device_cl(mask,   device) if mask is not None else None

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            out  = model(dadt, cond)
            loss = loss_fn(out, target, mask)

        # bf16: no scaler needed
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_unwrapped.parameters(), max_grad_norm)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = dadt.size(0)
        loss_sum     += loss.detach().double() * bs
        sample_count += bs

    # Global mean across all ranks
    dist.all_reduce(loss_sum,     op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    return (loss_sum / sample_count).item()


# -----------------------------
# Validate
# -----------------------------
@torch.no_grad()
def validate(model, loader, loss_fn, device, ema=None, is_master=True):
    # Swap in EMA weights for evaluation, then restore.
    if ema is not None:
        m = model.module if hasattr(model, 'module') else model
        live_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        ema.copy_to(model)

    model.eval()

    amp_ctx = (autocast(device_type='cuda', dtype=torch.bfloat16)
               if 'cuda' in str(device) else nullcontext())

    iterator = tqdm(loader, desc="Validation", leave=False) if is_master else loader

    loss_sum     = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    rel_num      = torch.zeros((), device=device, dtype=torch.float64)
    rel_den      = torch.zeros((), device=device, dtype=torch.float64)

    for batch in iterator:
        dadt, cond, target, mask = unpack_batch(batch)
        dadt   = _to_device_cl(dadt,   device)
        cond   = _to_device_cl(cond,   device)
        target = _to_device_cl(target, device)
        mask   = _to_device_cl(mask,   device) if mask is not None else None

        with amp_ctx:
            out  = model(dadt, cond)
            loss = loss_fn(out, target, mask)

        bs = dadt.size(0)
        loss_sum     += loss.detach().double() * bs
        sample_count += bs

        num, den = relative_l2_components(out.float(), target.float(), mask)
        rel_num += num.double()
        rel_den += den.double()

    # Restore live weights
    if ema is not None:
        m = model.module if hasattr(model, 'module') else model
        m.load_state_dict(live_state)

    # DDP reduction — sum loss * count, sum count, then divide.
    # This is correct regardless of how samples are distributed across ranks.
    dist.all_reduce(loss_sum,     op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(rel_num,      op=dist.ReduceOp.SUM)
    dist.all_reduce(rel_den,      op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / sample_count).item()
    rel_l2   = torch.sqrt(rel_num / rel_den).item()
    return val_loss, rel_l2


# -----------------------------
# Orchestrator
# -----------------------------
def train_model(
    sim_output_dir,
    epochs=100,
    batch_size=4,
    base_lr=2e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    plateau_epochs=45,         # flat-LR phase between warmup and cosine decay
    base_channels=32,          # width of the encoders/decoder
    checkpoint_dir="checkpoints",
    resume_path=None,
    use_compile=True,
    keep_last_n_epoch_checkpoints=3,
    ema_decay=0.999,
):
    # --- DDP setup ---
    # Bridge SLURM env vars to torchrun-style names when launched via
    # `srun python newTrain.py` (no torchrun wrapper). If LOCAL_RANK is
    # already set we were launched by torchrun and leave things alone.
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["RANK"]       = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device     = torch.device(f'cuda:{local_rank}')
    is_master  = (local_rank == 0)
    torch.cuda.set_device(local_rank)

    # sqrt scaling is the conservative AdamW choice; switch to linear
    # (base_lr * world_size) if you want more aggressive scaling.
    effective_lr = base_lr * (world_size ** 0.5)

    cosine_epochs = max(1, epochs - warmup_epochs - plateau_epochs)

    if is_master:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[INFO] Using {world_size} GPUs")
        print(f"[INFO] base_channels: {base_channels}")
        print(f"[INFO] Effective batch size: {batch_size * world_size}")
        print(f"[INFO] Effective LR: {effective_lr:.2e}")
        print(f"[INFO] LR schedule: warmup {warmup_epochs} → "
              f"plateau {plateau_epochs} → cosine {cosine_epochs}  "
              f"(total {epochs})")

    train_loader, val_loader, train_sampler = make_dataloaders(
        sim_output_dir=sim_output_dir,
        batch_size=batch_size,
        num_workers=8,
        prefetch_factor=4,
        distributed=True,
        rank=local_rank,
        world_size=world_size,
    )

    # Build model in channels_last_3d layout for faster H100 cuDNN kernels.
    model = (DualBranchResNet3D(base_channels=base_channels)
             .to(device)
             .to(memory_format=torch.channels_last_3d))
    model = DDP(model, device_ids=[local_rank])

    if use_compile:
        # Static input shapes (128^3) make this safe and effective.
        # NOTE: if you hit `CUDA error: misaligned address` during the first
        # batch (Inductor kernel alignment bug at uncommon channel counts),
        # set use_compile=False in __main__ and resubmit.
        model = torch.compile(model, mode='max-autotune')

    # Constant-weight, MSE-dominated loss. warmup_epochs=0 disables the
    # internal weight schedule — see CompositeLoss._weights().
    loss_fn = CompositeLoss(
        w_mae=1.0,
        w_grad=0.10,
        w_tv=0.002,
        w_bg=0.15,
        w_mse=1.0,           # MSE dominates so rel L2 (an L2 metric) is the right thing to drive
        warmup_epochs=0,     # no per-epoch ramp; constant weights from epoch 1
        use_auto_mask=True,
        auto_mask_threshold=1e-6,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        fused=True,    # small free speedup on CUDA (PyTorch 2.3+)
    )

    # 3-phase LR schedule:
    #   epochs 1..warmup_epochs                             — linear warmup from 0.1 * peak
    #   epochs warmup_epochs+1..warmup_epochs+plateau       — flat at peak LR
    #   remaining epochs                                    — cosine decay to eta_min
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            ConstantLR(optimizer, factor=1.0, total_iters=plateau_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs, warmup_epochs + plateau_epochs],
    )

    ema         = EMA(model, decay=ema_decay)
    best_val    = float("inf")
    start_epoch = 1

    # -----------------------------
    # Resume
    # -----------------------------
    if resume_path and os.path.exists(resume_path):
        if is_master:
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        # When the model is wrapped by torch.compile + DDP, the underlying
        # module is at model.module._orig_mod. Resolve it carefully.
        m = model
        while hasattr(m, 'module'):
            m = m.module
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        m.load_state_dict(ckpt['model_state_dict'])

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        ema.shadow  = {k: v.to(device) for k, v in ckpt['ema_shadow'].items()}
        best_val    = ckpt.get('best_val', float("inf"))
        start_epoch = ckpt['epoch'] + 1

        if is_master:
            print(f"[INFO] Resumed from epoch {ckpt['epoch']} | "
                  f"Best val so far: {best_val:.6f}")
    else:
        if is_master:
            print("[INFO] Starting training from scratch")

    # Helper to get the underlying nn.Module for state-dict saving,
    # past both DDP and torch.compile wrappers.
    def _raw_module():
        m = model
        while hasattr(m, 'module'):
            m = m.module
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        return m

    # -----------------------------
    # Train loop
    # -----------------------------
    for epoch in range(start_epoch, epochs + 1):
        loss_fn.set_epoch(epoch)
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        if is_master:
            print(f"\nEpoch {epoch}/{epochs}  "
                  f"(lr: {optimizer.param_groups[0]['lr']:.2e})")

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer,
            device, ema=ema, max_grad_norm=1.0, is_master=is_master,
        )
        val_loss, rel_l2 = validate(
            model, val_loader, loss_fn, device,
            ema=ema, is_master=is_master,
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        if is_master:
            print(f"  Train Loss: {train_loss:.6f} | "
                  f"Val Loss (EMA): {val_loss:.6f} | "
                  f"Rel L2: {rel_l2:.6e} | "
                  f"Time: {epoch_time:.2f}s")

            # Save best EMA weights
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    ema.shadow,
                    os.path.join(checkpoint_dir, "best_model_ema.pth"),
                )
                print("  Saved best EMA weights")

            raw = _raw_module()

            # Save resume checkpoint (overwrite each epoch)
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     raw.state_dict(),
                'ema_shadow':           ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val':             best_val,
                'base_channels':        base_channels,  # for sanity-checking on resume
            }, os.path.join(checkpoint_dir, "resume_checkpoint.pth"))

            # Save per-epoch weights, then prune to last N
            per_epoch_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(raw.state_dict(), per_epoch_path)

            if keep_last_n_epoch_checkpoints is not None:
                old = epoch - keep_last_n_epoch_checkpoints
                if old >= 1:
                    stale = os.path.join(checkpoint_dir, f"model_epoch_{old}.pth")
                    if os.path.exists(stale):
                        os.remove(stale)

            # Append history and rewrite CSV (master only)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["rel_l2"].append(rel_l2)
            history["epoch_time"].append(epoch_time)
            history["lr"].append(current_lr)

            csv_path = os.path.join(checkpoint_dir, "training_log.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss",
                                 "rel_l2", "epoch_time", "lr"])
                for i in range(len(history["epoch"])):
                    writer.writerow([
                        history["epoch"][i],
                        history["train_loss"][i],
                        history["val_loss"][i],
                        history["rel_l2"][i],
                        history["epoch_time"][i],
                        history["lr"][i],
                    ])

        scheduler.step()

    if is_master:
        print(f"\nDone. Best Val (EMA): {best_val:.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    SCRATCH = os.environ.get('MYSCRATCH')
    base    = os.path.join(SCRATCH, 'benchmark')

    # Fresh run at base_channels=32 — write to a NEW checkpoint directory
    # so the previous (base=16) checkpoints aren't accidentally loaded
    # and aren't overwritten.
    ckpt_dir = os.path.join(base, 'checkpoints_base32')

    train_model(
        sim_output_dir = os.path.join(SCRATCH, 'simOutput'),
        checkpoint_dir = ckpt_dir,
        # resume_path points into the new dir — won't exist on first launch
        # (so trains from scratch), but lets the job auto-resume itself if
        # it's killed by walltime and resubmitted.
        resume_path    = os.path.join(ckpt_dir, 'resume_checkpoint.pth'),
        epochs         = 100,
        batch_size     = 4,        # per-GPU. Drop to 2 if OOM at base_channels=32.
        base_lr        = 2e-4,
        weight_decay   = 1e-4,
        warmup_epochs  = 5,        # epochs 1-5
        plateau_epochs = 45,       # epochs 6-50 at peak LR
        base_channels  = 32,
        ema_decay      = 0.999,
        use_compile    = True,     # disable if "misaligned address" returns at base=32
    )
