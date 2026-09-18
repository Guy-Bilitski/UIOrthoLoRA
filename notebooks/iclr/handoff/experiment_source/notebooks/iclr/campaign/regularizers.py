"""Training penalties with cached fixed projectors and measured raw mixing."""

import math

import torch

from .protocol import P1_CONDITIONS


class CachedRegularizer:
    """Cache constants only; scaler/core tensors are always read at current state.

    Raw left/right mixing is evaluated on every call, including UNREG. Additional
    nuisance terms are evaluated only for the corresponding intervention, and
    absent terms are not emitted as false zeros. Random bases must be persisted
    in the run's immutable reference extras before optimization.
    """

    def __init__(self, layers, condition, coefficient, random_bases=None):
        if not layers or condition not in P1_CONDITIONS:
            raise ValueError("Nonempty spectral layers and a valid P1 condition are required")
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Coefficient must be finite and nonnegative")
        if condition in {"P1_UNREG", "P1_HEAD_BASE", "P1_HEAD_INIT"} and coefficient != 0:
            raise ValueError("An unpenalized reference cannot have a nonzero coefficient")
        self.layers = dict(layers)
        self.condition, self.coefficient = condition, coefficient
        self.projectors = {}
        self.random_projectors = {}
        self.pretrained_energy = {}
        for name, layer in layers.items():
            with torch.no_grad():
                self.projectors[name] = tuple(b[:, : layer.k] @ b[:, : layer.k].T for b in (layer.u_ref, layer.v_ref))
                self.pretrained_energy[name] = layer.w_pre.square().sum().detach()
                if self.pretrained_energy[name] <= 0:
                    raise ValueError("Relative-norm penalty requires nonzero pretrained matrices")
                if condition == "P1_RANDPROJ":
                    if random_bases is None or set(random_bases) != set(layers):
                        raise ValueError("Random-projector bases must cover exactly the adapted layers")
                    q, r = random_bases[name]
                    for basis, size in ((q, layer.e.numel()), (r, layer.d.numel())):
                        if (
                            basis.shape != (size, layer.k)
                            or basis.device != layer.e.device
                            or basis.dtype != layer.e.dtype
                        ):
                            raise ValueError("Random basis rank, dtype or device mismatch")
                        gram = basis.T @ basis
                        torch.testing.assert_close(
                            gram, torch.eye(layer.k, device=gram.device, dtype=gram.dtype), rtol=1e-5, atol=1e-6
                        )
                    self.random_projectors[name] = (q @ q.T, r @ r.T)

    def __call__(self):
        first = next(iter(self.layers.values()))
        zero = first.e.new_zeros(())
        left, right = zero, zero
        selected = zero
        for name, layer in self.layers.items():
            # Raw diagnostics remain real observations without retaining a large
            # unused autograd graph for conditions whose objective is different.
            needs_left = self.condition in {"P1_LEFT", "P1_MIX"}
            needs_right = self.condition == "P1_MIX"
            pe, pd = self.projectors[name]
            with torch.set_grad_enabled(torch.is_grad_enabled() and needs_left):
                a = ((layer.e[:, None] - layer.e[None, :]) * pe).square().sum() / 2
            with torch.set_grad_enabled(torch.is_grad_enabled() and needs_right):
                b = ((layer.d[:, None] - layer.d[None, :]) * pd).square().sum() / 2
            left, right = left + a, right + b
            if self.condition == "P1_NORM":
                selected = selected + layer.delta_total().square().sum() / self.pretrained_energy[name] / len(
                    self.layers
                )
            elif self.condition == "P1_CENTER":
                for scale in (layer.e, layer.d):
                    d, k = scale.numel(), layer.k
                    selected = selected + k * (d - k) / ((d - 1) * (d + 2)) * (scale - scale.mean()).square().sum()
            elif self.condition == "P1_DECAY_INIT":
                selected = selected + (layer.e - layer.e_init).square().sum() + (layer.d - layer.d_init).square().sum()
            elif self.condition == "P1_RANDPROJ":
                for scale, projector in zip((layer.e, layer.d), self.random_projectors[name]):
                    selected = selected + ((scale[:, None] - scale[None, :]) * projector).square().sum() / 2
        if self.condition == "P1_LEFT":
            selected = left
        elif self.condition == "P1_MIX":
            selected = left + right
        return self.coefficient * selected, dict(
            left=left.detach(), right=right.detach(), selected_unweighted=selected.detach()
        )
