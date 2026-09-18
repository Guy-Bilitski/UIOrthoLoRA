"""Strict band-restricted adapters for the 2026-09-15 band-by-flexibility study.

Update family: Delta = U_B H V_B^T for an equal-sized singular band
B = [band_start, band_start + band_size) of the pretrained SVD, added to the
unchanged pretrained weight. No ambient diagonal scalers and no additive fixed
leading/complement core. H is diagonal (band_size signed coefficients), with an
optional partial rotation of the band's LAST `rotation_size` directions on both
sides via the same tested orthogonal parameterization as the practical adapter:
H = blockdiag(diag(h_first), R_U diag(h_last) R_V^T). Coefficients initialize to
zero and rotations to identity, so every condition starts at the exact
pretrained effective backbone.

BandLinear reuses SpectralLinear's frozen references, rotation machinery,
merge/unmerge, reload guards and delta accounting; only the band column
selection differs. BandConfig is a separate schema so the active focused
study's registered configuration comparisons are untouched.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .modeling import attention_modules
from .spectral import SpectralConfig, SpectralLinear

BAND_SIZE = 256
ROTATION_CHOICES = (0, 64)
BAND_STARTS = {"lead": 0, "mid": 256, "tail": 512}


@dataclass(frozen=True)
class BandConfig:
    band_start: int
    band_size: int = BAND_SIZE
    rotation_size: int = 0

    def validate(self, rank=768):
        if rank % 3 or self.band_size != rank // 3:
            raise ValueError("Bands are the three equal thirds of the spectrum")
        if self.band_start not in (0, self.band_size, 2 * self.band_size):
            raise ValueError("band_start must select the leading, middle, or tail third")
        if not 0 <= self.rotation_size <= self.band_size:
            raise ValueError("rotation_size must fit inside the band")
        if rank == 768 and (self.band_size != BAND_SIZE or self.rotation_size not in ROTATION_CHOICES):
            raise ValueError("The registered study fixes band_size=256 and rotation in {0, 64}")


class BandLinear(SpectralLinear):
    def __init__(self, base: nn.Linear, config: BandConfig, reference=None):
        config.validate(min(base.weight.shape))
        self.band_config = config
        inner = SpectralConfig(
            tail_size=config.band_size,
            rotation_size=config.rotation_size,
            use_scalers=False,
            leading_identity=False,
            initial_scaler=1.0,
            initial_coefficient=0.0,
        )
        super().__init__(base, inner, reference=reference)
        if not torch.equal(self.delta_init, torch.zeros_like(self.delta_init)):
            raise ValueError("Band adapters must start at the exact pretrained backbone")

    def factors(self):
        start, size = self.band_config.band_start, self.band_config.band_size
        u = self.u_ref[:, start : start + size]
        v = self.v_ref[:, start : start + size]
        q = self.band_config.rotation_size
        if q:
            u = torch.cat((u[:, :-q], u[:, -q:] @ self.left_rotation.weight), dim=1)
            v = torch.cat((v[:, :-q], v[:, -q:] @ self.right_rotation.weight), dim=1)
        return u, v

    def get_extra_state(self):
        return dict(band=asdict(self.band_config), inner=asdict(self.config))

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint configuration does not match this band layer")

    def off_band_energy_fraction(self):
        """Enforcement check: energy of Delta outside the band, over its total."""
        delta = self.delta_total()
        start, size = self.band_config.band_start, self.band_config.band_size
        c = self.u_ref.T @ delta @ self.v_ref
        total = c.square().sum()
        if total == 0:
            return None
        inside = c[start : start + size, start : start + size].square().sum()
        return float((total - inside) / total)


def insert_band(model, config, references=None):
    model.requires_grad_(False)
    layers = {}
    for name, base in attention_modules(model).items():
        if not isinstance(base, nn.Linear):
            raise TypeError("Band insertion expected unwrapped nn.Linear")
        layer = BandLinear(base, config, reference=None if references is None else references[name])
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], layer)
        layers[name] = layer
    model.classifier.requires_grad_(True)
    return layers
