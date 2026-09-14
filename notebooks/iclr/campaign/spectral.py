"""Explicit additive spectral layer and original-frame diagnostic primitives.

This is the prospective runner's implementation, not a modification of the legacy
adapter. All reference tensors persist in state_dict; reload must restore them.
Only dense, unquantized nn.Linear weights are supported in this implementation.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpectralConfig:
    tail_size: int
    rotation_size: int = 0
    use_scalers: bool = True
    leading_identity: bool = True
    initial_scaler: float = 0.01
    initial_coefficient: float = 0.01
    dense_tail: bool = False


class SpectralLinear(nn.Module):
    def __init__(self, base: nn.Linear, config: SpectralConfig, reference=None):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.dtype not in (torch.float32, torch.float64):
            raise TypeError("P0 supports unquantized float32/float64 nn.Linear; use autocast for mixed precision")
        m, n = base.weight.shape
        rank = min(m, n)
        if not 0 < config.tail_size < rank or not 0 <= config.rotation_size <= config.tail_size:
            raise ValueError("Require 0 < tail_size < min(shape), 0 <= rotation_size <= tail_size")
        if config.dense_tail and config.rotation_size:
            raise ValueError("A dense core and partial rotations are separate parameterizations")
        self.config = config
        self.base = base.requires_grad_(False)
        self.k = rank - config.tail_size
        # Full matrices preserve rectangular unmatched complements for diagnostics.
        with torch.no_grad():
            if reference is None:
                u, s, vh = torch.linalg.svd(base.weight, full_matrices=True)
            else:
                if (
                    reference["u_ref"].shape != (m, m)
                    or reference["v_ref"].shape != (n, n)
                    or reference["s_ref"].shape != (rank,)
                ):
                    raise ValueError("Saved reference must contain complete bases and the original spectrum")
                if not torch.equal(reference["w_pre"].to(base.weight), base.weight):
                    raise ValueError("Saved reference does not match the original base weight")
                u, s, vh = (
                    reference["u_ref"].to(base.weight),
                    reference["s_ref"].to(base.weight),
                    reference["v_ref"].T.to(base.weight),
                )
        for name, tensor in (("u_ref", u), ("v_ref", vh.T), ("s_ref", s), ("w_pre", base.weight.detach().clone())):
            self.register_buffer(name, tensor.detach().clone())
        factory = dict(dtype=base.weight.dtype, device=base.weight.device)
        scales = config.initial_scaler if config.use_scalers else 1.0
        self.e = nn.Parameter(torch.full((m,), scales, **factory), requires_grad=config.use_scalers)
        self.d = nn.Parameter(torch.full((n,), scales, **factory), requires_grad=config.use_scalers)
        self.register_buffer("e_init", self.e.detach().clone())
        self.register_buffer("d_init", self.d.detach().clone())
        h = torch.full((config.tail_size,), config.initial_coefficient, **factory)
        self.h = nn.Parameter(torch.diag(h) if config.dense_tail else h)
        self.left_rotation = self.right_rotation = None
        if config.rotation_size:

            def rotation():
                module = nn.Linear(config.rotation_size, config.rotation_size, bias=False, **factory)
                with torch.no_grad():
                    module.weight.copy_(torch.eye(config.rotation_size, **factory))
                return nn.utils.parametrizations.orthogonal(module, orthogonal_map="matrix_exp")

            self.left_rotation, self.right_rotation = rotation(), rotation()
        self.register_buffer("delta_init", self.delta_total().detach().clone())
        self.adapter_enabled = True
        self._merge_backup = None

    def get_extra_state(self):
        return asdict(self.config)

    def set_extra_state(self, state):
        if state != asdict(self.config):
            raise ValueError("Checkpoint configuration does not match this spectral layer")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self._merge_backup is not None:
            raise RuntimeError("Unmerge before saving to prevent double application on reload")
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def factors(self):
        r = len(self.s_ref)
        u = self.u_ref[:, self.k : r]
        v = self.v_ref[:, self.k : r]
        q = self.config.rotation_size
        if q:
            u = torch.cat((u[:, :-q], u[:, -q:] @ self.left_rotation.weight), dim=1)
            v = torch.cat((v[:, :-q], v[:, -q:] @ self.right_rotation.weight), dim=1)
        return u, v

    def tail_core(self):
        h = self.h if self.config.dense_tail else torch.diag(self.h)
        q = self.config.rotation_size
        if q:
            # Partial rotation changes only the last q directions of the tail.
            h = torch.block_diag(h[:-q, :-q], self.left_rotation.weight @ h[-q:, -q:] @ self.right_rotation.weight.T)
        return h

    def exact_leading_term(self):
        raw = self.u_ref[:, : self.k] @ self.v_ref[:, : self.k].T
        return self.e[:, None] * raw * self.d[None, :] if self.config.leading_identity else torch.zeros_like(raw)

    def delta_total(self):
        u, v = self.factors()
        tail = u @ self.h @ v.T if self.config.dense_tail else (u * self.h) @ v.T
        return self.exact_leading_term() + self.e[:, None] * tail * self.d[None, :]

    def delta_learned(self):
        return self.delta_total() - self.delta_init

    def forward(self, x):
        if not self.adapter_enabled:
            if self._merge_backup is not None:
                return F.linear(x, self._merge_backup, self.base.bias)
            return self.base(x)
        result = self.base(x)
        if self._merge_backup is not None:
            return result
        u, v = self.factors()
        z = (x * self.d) @ v
        z = z @ self.h.T if self.config.dense_tail else z * self.h
        update = z @ u.T
        if self.config.leading_identity:
            update = update + ((x * self.d) @ self.v_ref[:, : self.k]) @ self.u_ref[:, : self.k].T
        return result + update * self.e

    @torch.no_grad()
    def merge(self):
        if self.training:
            raise RuntimeError("Merge is an evaluation operation; call eval() first")
        if self._merge_backup is not None:
            raise RuntimeError("Already merged")
        merged = self.base.weight + self.delta_total()
        if not torch.isfinite(merged).all():
            raise ValueError("Nonfinite merged weight")
        self._merge_backup = self.base.weight.detach().clone()
        self.base.weight.copy_(merged)

    @torch.no_grad()
    def unmerge(self):
        if self._merge_backup is None:
            raise RuntimeError("Not merged")
        # Exact restoration also avoids low-precision cancellation on repeated merges.
        self.base.weight.copy_(self._merge_backup)
        self._merge_backup = None


def mixing_squared(scales, leading):
    """Half the squared commutator, using the full complementary projector.

    A Gram expression costs O(d k^2) without forming a full complement. The
    explicit commutator below is nonnegative and avoids cancellation near scalars.
    Projector construction can be cached by a runner, since bases are frozen.
    """
    projector = leading @ leading.T
    return ((scales[:, None] - scales[None, :]) * projector).square().sum() / 2


def haar_basis(d, k, seed, *, dtype=torch.float64, device="cpu"):
    """Fixed independent draw, with local RNG and QR diagonal sign correction."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, k, generator=generator, dtype=dtype), mode="reduced")
    signs = torch.where(r.diag() < 0, -1.0, 1.0)
    return (q * signs).to(device)


def regularization(layers, condition, coefficient=0.0, random_bases=None):
    """Exact P1 losses: MIX/LEFT/CENTER/DECAY/RAND sums; NORM module mean.

    Unweighted mixing diagnostics are computed even with zero coefficients.
    For runtime accounting, this initial implementation deliberately computes
    every nuisance quantity; timed calibration must measure the actual overhead.
    """
    from .protocol import P1_CONDITIONS

    if condition not in P1_CONDITIONS or coefficient < 0:
        raise ValueError("Invalid P1 condition or negative coefficient")
    if not layers:
        raise ValueError("Pass the common diagnostic layers, including for HEAD_INIT")
    first = next(iter(layers.values()))
    terms = {name: first.e.new_zeros(()) for name in ("left", "right", "norm", "center", "decay_init", "randproj")}
    for name, layer in layers.items():
        k = layer.k
        terms["left"] = terms["left"] + mixing_squared(layer.e, layer.u_ref[:, :k])
        terms["right"] = terms["right"] + mixing_squared(layer.d, layer.v_ref[:, :k])
        terms["norm"] = terms["norm"] + layer.delta_total().square().sum() / layer.w_pre.square().sum()
        for scales, initial in ((layer.e, layer.e_init), (layer.d, layer.d_init)):
            d = scales.numel()
            c = k * (d - k) / ((d - 1) * (d + 2))
            terms["center"] = terms["center"] + c * (scales - scales.mean()).square().sum()
            terms["decay_init"] = terms["decay_init"] + (scales - initial).square().sum()
        if condition == "P1_RANDPROJ":
            q, r = random_bases[name]
            if q.shape != (layer.e.numel(), k) or r.shape != (layer.d.numel(), k):
                raise ValueError("Random projector rank/dimension mismatch")
            terms["randproj"] = terms["randproj"] + mixing_squared(layer.e, q) + mixing_squared(layer.d, r)
    terms["norm"] = terms["norm"] / len(layers)
    selected = {
        "P1_LEFT": terms["left"],
        "P1_MIX": terms["left"] + terms["right"],
        "P1_NORM": terms["norm"],
        "P1_CENTER": terms["center"],
        "P1_DECAY_INIT": terms["decay_init"],
        "P1_RANDPROJ": terms["randproj"],
    }
    loss = coefficient * selected.get(condition, first.e.new_zeros(()))
    return loss, terms


def coordinate_blocks(delta, u, v, k):
    if u.shape != (delta.shape[0], delta.shape[0]) or v.shape != (delta.shape[1], delta.shape[1]):
        raise ValueError("Diagnostics require complete square reference bases")
    if not 0 < k < min(delta.shape):
        raise ValueError("Cutoff must leave a nonempty paired tail")
    c = u.T @ delta @ v
    return c, {"LL": c[:k, :k], "LT": c[:k, k:], "TL": c[k:, :k], "TT": c[k:, k:]}


def block_summary(delta, u, v, k):
    c, blocks = coordinate_blocks(delta, u, v, k)
    energies = {name: b.square().sum().item() for name, b in blocks.items()}
    energy = delta.square().sum().item()
    fractions = {name: value / energy if energy > 0 else None for name, value in energies.items()}
    ll_energy = energies["LL"]
    identity_coefficient = blocks["LL"].trace() / k
    residual = blocks["LL"] - identity_coefficient * torch.eye(k, dtype=c.dtype, device=c.device)
    return dict(
        energy=energy,
        block_energy=energies,
        fractions=fractions,
        p_cross=None if not energy else fractions["LT"] + fractions["TL"],
        p_off=None if not energy else fractions["LL"] + fractions["LT"] + fractions["TL"],
        identity_alignment=blocks["LL"].trace().square().item() / (k * ll_energy) if ll_energy else None,
        identity_residual_energy=residual.square().sum().item(),
        reconstruction_error=torch.linalg.vector_norm(u @ c @ v.T - delta).item(),
        energy_accounting_error=abs(sum(energies.values()) - energy),
    )
