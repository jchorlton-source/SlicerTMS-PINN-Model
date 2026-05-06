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

Two exact consequences of (1) are useful as PDE residuals:

  (A) Current continuity (charge conservation in steady state):

          ∇·(σE) = 0                                             (2)

      E itself is *not* divergence-free in heterogeneous tissue —
      ∇·E concentrates at conductivity discontinuities.

  (B) The secondary field is curl-free:

          ∇×(E + ∂A/∂t) = -∇×∇φ ≡ 0                              (3)

These two residuals form the physics core of this loss. A small data
fidelity term (MSE or L1) anchors the gauge of φ; without it (2) and
(3) admit a family of solutions related by harmonic potentials.

Usage
-----
The ``forward`` signature matches how ``newTrain.py`` already unpacks
batches, so the only change required at the call site is:

    loss = loss_fn(pred, target, dadt=dadt, conductivity=cond, mask=mask)

Set ``w_data = 0`` (or use the ``pure_pinn`` factory) for a fully
unsupervised "pure" PINN run. Be aware that without a data anchor the
problem is underdetermined: pure-physics fine-tuning works best as a
second stage, after a hybrid run has converged.
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

    Edge slabs use one-sided (forward / backward) diffs so the derivative
    is defined everywhere — at the cost of slightly biased edges. With
    masked losses the bias is mostly outside the foreground anyway.
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
    Quasi-static TMS PINN loss with optional data anchor.

    Parameters
    ----------
    w_data, w_continuity, w_curl : float
        Term weights. With E in V/m, σ in S/m and dx in mm, the three
        terms have different magnitudes — log ``last_components`` after
        the first epoch and rebalance so contributions are within ~1
        order of magnitude before drawing conclusions.
    dx : float
        Voxel spacing in physical units (mm for SimNIBS ``ernie``
        defaults). Affects the absolute magnitude of the residuals; if
        you change dx and want the same effective weighting, scale
        ``w_continuity`` and ``w_curl`` by 1/dx² accordingly.
    data_loss : {"mse", "l1"}
        L2 vs L1 data fidelity. Use "l1" if you want the anchor to match
        the MAE-dominated benchmark CompositeLoss; "mse" pairs naturally
        with the rel-L2 validation metric.
    dadt_sign : {+1.0, -1.0}
        Sign convention for the input ``dadt`` channel. SimNIBS
        ``scalar_D`` outputs +∂A/∂t — keep the default. Flip to -1.0 if
        your dataset stores the primary E-field instead.
    use_auto_mask : bool
        If True and no explicit mask is provided, build one from the
        target's per-voxel magnitude. This restricts every term to the
        foreground (the brain), which matters: ∇·(σE) ≡ 0 trivially
        outside the head and would otherwise dominate the average.
    auto_mask_threshold : float
        Magnitude threshold (V/m) above which a target voxel counts as
        foreground.
    """

    _VALID_DATA_LOSS = ("mse", "l1")

    def __init__(
        self,
        w_data: float = 1.0,
        w_continuity: float = 1.0,
        w_curl: float = 1.0,
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
        if min(w_data, w_continuity, w_curl) < 0:
            raise ValueError("All loss weights must be non-negative")

        self.w_data              = float(w_data)
        self.w_continuity        = float(w_continuity)
        self.w_curl              = float(w_curl)
        self.dx                  = float(dx)
        self.data_loss_kind      = data_loss
        self.dadt_sign           = float(dadt_sign)
        self.use_auto_mask       = bool(use_auto_mask)
        self.auto_mask_threshold = float(auto_mask_threshold)

        # Last-forward components (unweighted), exposed for CSV logging.
        # Detached, so logging them is graph-safe.
        self.last_components: dict[str, torch.Tensor] = {}

    # ---- API compatibility shim --------------------------------------------
    # newTrain.py calls loss_fn.set_epoch(epoch). The benchmark CompositeLoss
    # uses it for a weight schedule; we use constant weights, so it's a no-op.
    def set_epoch(self, e: int) -> None:
        return

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _masked_mean(
        x: torch.Tensor, mask: torch.Tensor, channels: int
    ) -> torch.Tensor:
        """
        Mean over voxels where mask>0, normalised per channel so the loss
        magnitude does not depend on how much of the volume is foreground.

        x:    (B, C, D, H, W)
        mask: (B, 1, D, H, W) — broadcast over channels
        """
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
        """
        ∇·(σE) in conservative form:

            ∂(σ Ex)/∂x + ∂(σ Ey)/∂y + ∂(σ Ez)/∂z

        We differentiate σE directly rather than expanding to
        σ ∇·E + E·∇σ. The product form keeps the discretisation
        consistent at sharp σ discontinuities (CSF–GM, GM–WM); the
        expanded form would amplify errors via a separate ∇σ term.
        """
        sx = conductivity * e_field[:, 0:1]
        sy = conductivity * e_field[:, 1:2]
        sz = conductivity * e_field[:, 2:3]
        d_sx = _grad_central_3d(sx, dim=2, dx=self.dx)
        d_sy = _grad_central_3d(sy, dim=3, dx=self.dx)
        d_sz = _grad_central_3d(sz, dim=4, dx=self.dx)
        return d_sx + d_sy + d_sz                            # (B, 1, D, H, W)

    def curl_residual(
        self, e_field: torch.Tensor, dadt: torch.Tensor
    ) -> torch.Tensor:
        """
        ∇×(E + ∂A/∂t) — vanishes in the QS regime since
        E + ∂A/∂t = -∇φ. Axes: x = D (dim 2), y = H (3), z = W (4).
        """
        f = e_field + self.dadt_sign * dadt
        fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        curl_x = _grad_central_3d(fz, 3, self.dx) - _grad_central_3d(fy, 4, self.dx)
        curl_y = _grad_central_3d(fx, 4, self.dx) - _grad_central_3d(fz, 2, self.dx)
        curl_z = _grad_central_3d(fy, 2, self.dx) - _grad_central_3d(fx, 3, self.dx)
        return torch.cat([curl_x, curl_y, curl_z], dim=1)    # (B, 3, D, H, W)

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

        zero = prediction.new_tensor(0.0)
        l_data = zero
        l_cont = zero
        l_curl = zero

        # ---- data fidelity (masked, per-channel normalised) ----
        if self.w_data > 0:
            if self.data_loss_kind == "mse":
                l_data = self._masked_mean(
                    (prediction - target) ** 2, mask, prediction.shape[1]
                )
            else:  # "l1"
                l_data = self._masked_mean(
                    (prediction - target).abs(), mask, prediction.shape[1]
                )

        # ---- current continuity ∇·(σE) = 0 ----
        if self.w_continuity > 0 and conductivity is not None:
            r_cont = self.continuity_residual(prediction, conductivity)
            l_cont = self._masked_mean(r_cont ** 2, mask, 1)

        # ---- curl-free secondary field ∇×(E + ∂A/∂t) = 0 ----
        if self.w_curl > 0 and dadt is not None:
            r_curl = self.curl_residual(prediction, dadt)
            l_curl = self._masked_mean(r_curl ** 2, mask, 3)

        total = (
            self.w_data       * l_data
            + self.w_continuity * l_cont
            + self.w_curl       * l_curl
        )

        # Stash unweighted components for logging. Detached — no autograd impact.
        self.last_components = {
            "data":       l_data.detach(),
            "continuity": l_cont.detach(),
            "curl":       l_curl.detach(),
        }
        return total


# =============================================================================
# Convenience constructors
# =============================================================================
def hybrid(
    w_data: float = 1.0,
    w_continuity: float = 1.0,
    w_curl: float = 1.0,
    dx: float = 1.0,
    data_loss: str = "mse",
) -> PhysicsInformedLoss:
    """Data anchor + physics residuals. Recommended for the first run."""
    return PhysicsInformedLoss(
        w_data=w_data,
        w_continuity=w_continuity,
        w_curl=w_curl,
        dx=dx,
        data_loss=data_loss,
    )


def pure_pinn(
    w_continuity: float = 1.0,
    w_curl: float = 1.0,
    dx: float = 1.0,
) -> PhysicsInformedLoss:
    """
    No data anchor — physics residuals only. The problem is then
    underdetermined (gauge of φ is free up to harmonic functions); works
    best as a fine-tuning stage starting from a hybrid checkpoint rather
    than from random init.
    """
    return PhysicsInformedLoss(
        w_data=0.0,
        w_continuity=w_continuity,
        w_curl=w_curl,
        dx=dx,
    )
