"""
PhysicsLoss.py
==============

Physics-informed loss for TMS-induced E-field prediction in the
quasi-static regime.

Physics
-------
At kHz frequencies the induced E-field decomposes into a primary part
driven by the coil current and a secondary part from charge accumulation
at conductivity boundaries:

    E = -∂A/∂t - ∇φ                                              (1)

Two exact consequences of (1) are used as PDE residuals:

  (A) Current continuity (charge conservation in steady state):

          ∇·(σE) = 0                                             (2)

  (B) The secondary field is curl-free:

          ∇×(E + ∂A/∂t) = -∇×∇φ ≡ 0                              (3)

The loss combines (2) and (3) with:
  - a data-fidelity term (MSE or L1) over the foreground (brain),
  - a background-suppression term that drives predictions to zero
    outside the brain mask.

The background term is necessary. Without it, the loss provides no signal
in voxels outside the foreground mask — the model is free to output any
values there, and in practice predictions drift, inflating any unmasked
validation metric. The benchmark CompositeLoss had this term; v2 of this
file did not, which produced runs where train/val loss dropped while
rel-L2 climbed because background predictions blew up.

Usage
-----
The ``forward`` signature is:

    loss = loss_fn(pred, target, dadt=dadt, conductivity=cond, mask=mask)

Pass ``mask=None`` to use the auto-mask built from the target.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =============================================================================
# Finite-difference primitives
# =============================================================================
def _grad_central_3d(x: torch.Tensor, dim: int, dx: float) -> torch.Tensor:
    """
    Same-shape central difference along spatial axis ``dim`` (2, 3, or 4
    on a (B, C, D, H, W) tensor).

    Edge slabs use one-sided diffs so the derivative is defined
    everywhere — at the cost of slightly biased edges. With masked
    losses the bias is mostly outside the foreground.
    """
    g = torch.empty_like(x)
    if dim == 2:
        g[:, :, 1:-1] = (x[:, :, 2:] - x[:, :, :-2]) * 0.5
        g[:, :,  0]   =  x[:, :,  1] - x[:, :,  0]
        g[:, :, -1]   =  x[:, :, -1] - x[:, :, -2]
    elif dim == 3:
        g[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) * 0.5
        g[:, :, :,  0]   =  x[:, :, :,  1] - x[:, :, :,  0]
        g[:, :, :, -1]   =  x[:, :, :, -1] - x[:, :, :, -2]
    elif dim == 4:
        g[:, :, :, :, 1:-1] = (x[:, :, :, :, 2:] - x[:, :, :, :, :-2]) * 0.5
        g[:, :, :, :,  0]   =  x[:, :, :, :,  1] - x[:, :, :, :,  0]
        g[:, :, :, :, -1]   =  x[:, :, :, :, -1] - x[:, :, :, :, -2]
    else:
        raise ValueError(f"dim must be 2, 3, or 4 (got {dim})")
    return g / dx


# =============================================================================
# Loss
# =============================================================================
class PhysicsInformedLoss(nn.Module):
    """
    Quasi-static TMS PINN loss with optional data anchor and background
    suppression.

    Parameters
    ----------
    w_data, w_continuity, w_curl, w_bg : float
        Term weights. Defaults match the benchmark's effective balance
        (w_bg = 0.15 from CompositeLoss). Log ``last_components`` after
        epoch 1 and rebalance so weighted contributions are within ~1
        order of magnitude before drawing conclusions.
    dx : float
        Voxel spacing in physical units (mm for SimNIBS ``ernie``).
    data_loss : {"mse", "l1"}
        L2 vs L1 data fidelity.
    dadt_sign : {+1.0, -1.0}
        Sign convention for the input ``dadt`` channel.
    use_auto_mask : bool
        Build a mask from target magnitude when none is provided.
    auto_mask_threshold : float
        Magnitude threshold above which a target voxel counts as
        foreground.
    """

    _VALID_DATA_LOSS = ("mse", "l1")

    def __init__(
        self,
        w_data: float = 1.0,
        w_continuity: float = 1.0,
        w_curl: float = 1.0,
        w_bg: float = 0.15,
        dx: float = 1.0,
        data_loss: str = "mse",
        dadt_sign: float = +1.0,
        use_auto_mask: bool = True,
        auto_mask_threshold: float = 1e-6,
    ):
        super().__init__()

        if data_loss not in self._VALID_DATA_LOSS:
            raise ValueError(
                f"data_loss must be one of {self._VALID_DATA_LOSS}, got {data_loss!r}"
            )
        if dadt_sign not in (+1.0, -1.0):
            raise ValueError(f"dadt_sign must be +1.0 or -1.0, got {dadt_sign}")
        if min(w_data, w_continuity, w_curl, w_bg) < 0:
            raise ValueError("All loss weights must be non-negative")

        self.w_data              = float(w_data)
        self.w_continuity        = float(w_continuity)
        self.w_curl              = float(w_curl)
        self.w_bg                = float(w_bg)
        self.dx                  = float(dx)
        self.data_loss_kind      = data_loss
        self.dadt_sign           = float(dadt_sign)
        self.use_auto_mask       = bool(use_auto_mask)
        self.auto_mask_threshold = float(auto_mask_threshold)

        # Last-forward components, exposed for CSV logging.
        self.last_components: dict[str, torch.Tensor] = {}

    # API compatibility with CompositeLoss.set_epoch — no-op here.
    def set_epoch(self, e: int) -> None:
        return

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _masked_mean(
        x: torch.Tensor, mask: torch.Tensor, channels: int
    ) -> torch.Tensor:
        denom = mask.sum() * channels + 1e-8
        return (x * mask).sum() / denom

    def _build_mask(
        self, target: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        if self.use_auto_mask:
            mag = target.abs().sum(dim=1, keepdim=True)
            return (mag > self.auto_mask_threshold).to(pred.dtype)
        return torch.ones_like(pred[:, :1])

    # ---- PDE residuals -----------------------------------------------------
    def continuity_residual(
        self, e_field: torch.Tensor, conductivity: torch.Tensor
    ) -> torch.Tensor:
        sx = conductivity * e_field[:, 0:1]
        sy = conductivity * e_field[:, 1:2]
        sz = conductivity * e_field[:, 2:3]
        d_sx = _grad_central_3d(sx, dim=2, dx=self.dx)
        d_sy = _grad_central_3d(sy, dim=3, dx=self.dx)
        d_sz = _grad_central_3d(sz, dim=4, dx=self.dx)
        return d_sx + d_sy + d_sz

    def curl_residual(
        self, e_field: torch.Tensor, dadt: torch.Tensor
    ) -> torch.Tensor:
        f = e_field + self.dadt_sign * dadt
        fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        curl_x = _grad_central_3d(fz, 3, self.dx) - _grad_central_3d(fy, 4, self.dx)
        curl_y = _grad_central_3d(fx, 4, self.dx) - _grad_central_3d(fz, 2, self.dx)
        curl_z = _grad_central_3d(fy, 2, self.dx) - _grad_central_3d(fx, 3, self.dx)
        return torch.cat([curl_x, curl_y, curl_z], dim=1)

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        dadt: torch.Tensor | None = None,
        conductivity: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = self._build_mask(target, prediction)
        mask = mask.to(prediction.dtype)
        C    = prediction.shape[1]

        zero = prediction.new_tensor(0.0)
        l_data = zero
        l_cont = zero
        l_curl = zero
        l_bg   = zero

        # ---- data fidelity (masked, per-channel normalised) ----
        if self.w_data > 0:
            if self.data_loss_kind == "mse":
                l_data = self._masked_mean(
                    (prediction - target) ** 2, mask, C
                )
            else:  # "l1"
                l_data = self._masked_mean(
                    (prediction - target).abs(), mask, C
                )

        # ---- current continuity ∇·(σE) = 0 ----
        if self.w_continuity > 0 and conductivity is not None:
            r_cont = self.continuity_residual(prediction, conductivity)
            l_cont = self._masked_mean(r_cont ** 2, mask, 1)

        # ---- curl-free secondary field ∇×(E + ∂A/∂t) = 0 ----
        if self.w_curl > 0 and dadt is not None:
            r_curl = self.curl_residual(prediction, dadt)
            l_curl = self._masked_mean(r_curl ** 2, mask, 3)

        # ---- background suppression: pred → 0 outside the mask ----
        # Without this term the model has no training signal in the
        # background, predictions drift, and any unmasked validation
        # metric (like rel-L2 on the whole volume) explodes even while
        # the masked data loss falls. Mean-of-|pred| weighted by how
        # much of the volume is background, per channel.
        if self.w_bg > 0:
            inv      = 1.0 - mask
            bg_denom = inv.sum() * C + 1e-8
            l_bg     = (prediction.abs() * inv).sum() / bg_denom

        total = (
            self.w_data       * l_data
            + self.w_continuity * l_cont
            + self.w_curl       * l_curl
            + self.w_bg         * l_bg
        )

        # Stash unweighted components for logging. Detached — no autograd impact.
        self.last_components = {
            "data":       l_data.detach(),
            "continuity": l_cont.detach(),
            "curl":       l_curl.detach(),
            "bg":         l_bg.detach(),
        }
        return total


# =============================================================================
# Convenience constructors
# =============================================================================
def hybrid(
    w_data: float = 1.0,
    w_continuity: float = 0.01,
    w_curl: float = 0.01,
    w_bg: float = 0.15,
    dx: float = 1.0,
    data_loss: str = "mse",
) -> PhysicsInformedLoss:
    """
    Data anchor + physics residuals + background suppression.

    Default physics weights are 0.01 (not 1.0) to keep physics as a soft
    regularizer rather than a co-dominant term — without this the model
    tends to collapse toward the trivial curl=0 minimum (E ≈ -∂A/∂t)
    early in training and never recovers.
    """
    return PhysicsInformedLoss(
        w_data=w_data,
        w_continuity=w_continuity,
        w_curl=w_curl,
        w_bg=w_bg,
        dx=dx,
        data_loss=data_loss,
    )


def pure_pinn(
    w_continuity: float = 1.0,
    w_curl: float = 1.0,
    w_bg: float = 0.15,
    dx: float = 1.0,
) -> PhysicsInformedLoss:
    """
    No data anchor — physics residuals + background suppression only.
    Underdetermined; best used as fine-tuning from a hybrid checkpoint.
    """
    return PhysicsInformedLoss(
        w_data=0.0,
        w_continuity=w_continuity,
        w_curl=w_curl,
        w_bg=w_bg,
        dx=dx,
    )
