from __future__ import annotations
from typing import Dict, List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight
from transformers.pytorch_utils import Conv1D

__all__ = ["UIOrthoLoRALayer", "Linear"]

class IdentityWithTranspose(nn.Identity):
    @property
    def T(self):
        return self

    def __matmul__(self, other):
        return other  # Identity @ other

    def __rmatmul__(self, other):
        return other  # other @ Identity


class UIOrthoLoRALayer(BaseTunerLayer):
    adapter_layer_names = ("uiortholora_sigma", "uiortholora_D", "uiortholora_E")
    other_param_names = ("uiortholora_alpha", "uiortholora_dropout", "rank", "scaling_factor", "enforce_sv_positive")

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
         # ---- handle Linear vs Conv1D ----------------------------------
        if isinstance(base_layer, Conv1D):
            self.in_features  = base_layer.weight.shape[0]
            self.out_features = base_layer.weight.shape[1]
        else:
            self.in_features  = base_layer.in_features
            self.out_features = base_layer.out_features

        self.uiortholora_sigma = nn.ParameterDict()
        self.uiortholora_D = nn.ParameterDict()
        self.uiortholora_E = nn.ParameterDict()
        self.uiortholora_left_unitary = nn.ParameterDict()
        self.uiortholora_right_unitary = nn.ParameterDict()
        self.uiortholora_dropout = nn.ModuleDict()
        self.device = self.base_layer.weight.device
        self._meta: Dict[str, dict] = {}
        self.kwargs = kwargs
        self.num_svalues_to_adapt = kwargs.pop("num_svalues_to_adapt")
        self.num_svectors_to_adapt = kwargs.pop("num_svectors_to_adapt")
        self.dtype = self.base_layer.weight.dtype


    def update_layer(
        self,
        adapter_name: str,
        *,
        scaling_factor: float = 1.0,
        enforce_sv_positive: bool = False,
        uiortholora_dropout: float = 0.0,
        initial_scaler: Optional[float] = 1e-1,
        initial_sigma: Optional[float] = 1e-1,
        **kwargs,
    ):
            
        if adapter_name in self.uiortholora_sigma.keys():
            return

        base_w = self.get_pretrained_matrix()

        rank = min(self.in_features, self.out_features)
        # rank_to_preserve = rank - self.num_svectors_to_adapt
        self.major_component_size = rank - self.num_svalues_to_adapt
        self.medium_component_size = rank - self.num_svectors_to_adapt

        # Compute SVD and slice the smallest singular vectors
        if not self.buffers_loaded(adapter_name):
            with torch.no_grad():
                U, S, Vt = torch.linalg.svd(base_w.float(), full_matrices=False)
                U = U.to(dtype=self.dtype)
                Vt = Vt.to(dtype=self.dtype)
                S = S.to(dtype=self.dtype)
            print(f"Calculated SVD!")

        # Major component between svalues and svectors
        print(f"keeping major: {self.major_component_size}")
        self.register_buffer(f"{adapter_name}_U1", U[:, :self.major_component_size].detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_S1", torch.ones(self.major_component_size, dtype=self.dtype).detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_Vt1", Vt[:self.major_component_size, :].detach(), persistent=True)

        # Medium component between svalues and svectors
        self.register_buffer(f"{adapter_name}_U2", U[:, self.major_component_size:self.medium_component_size].detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_S2", S[self.major_component_size:self.medium_component_size].to(self.dtype).detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_Vt2", Vt[self.major_component_size:self.medium_component_size, :].detach(), persistent=True)

        # Small component
        self.register_buffer(f"{adapter_name}_U3", U[:, self.medium_component_size:].detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_S3", S[self.medium_component_size:].to(self.dtype).detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_Vt3", Vt[self.medium_component_size:, :].detach(), persistent=True)

        self.uiortholora_sigma[adapter_name] = nn.Parameter(torch.full((self.num_svalues_to_adapt,), initial_sigma, dtype=self.dtype))

        # Initialize D and E with provided scaler or default of 1
        self.uiortholora_D[adapter_name] = nn.Parameter(torch.full((self.in_features,), initial_scaler, dtype=self.dtype))
        self.uiortholora_E[adapter_name] = nn.Parameter(torch.full((self.out_features,), initial_scaler, dtype=self.dtype))

        if self.num_svectors_to_adapt == 0:
            left_orthogonal = IdentityWithTranspose()
            right_orthogonal = IdentityWithTranspose()

        else:
            left_orthogonal = torch.nn.utils.parametrizations.orthogonal(
                nn.Linear(self.num_svectors_to_adapt, self.num_svectors_to_adapt, bias=False)
            )
            right_orthogonal = torch.nn.utils.parametrizations.orthogonal(
                nn.Linear(self.num_svectors_to_adapt, self.num_svectors_to_adapt, bias=False)
            )
        
        self.uiortholora_left_unitary[adapter_name] = left_orthogonal
        self.uiortholora_right_unitary[adapter_name] = right_orthogonal

        self._meta[adapter_name] = dict(sf=scaling_factor, pos=enforce_sv_positive)

        # Add dropout
        self.uiortholora_dropout.update(nn.ModuleDict({
            adapter_name: nn.Dropout(uiortholora_dropout) if uiortholora_dropout > 0.0 else nn.Identity()
        }))

        # Move newly added parameters to the base layer's device
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # Activate the adapter
        self.set_adapter(self.active_adapters)

    def buffers_loaded(self, adapter: str) -> bool:
        # just check one of the core buffers
        return hasattr(self, f"{adapter}_U1") and isinstance(getattr(self, f"{adapter}_U1"), torch.Tensor)



    def get_base_layer(self) -> nn.Module:
        return self.base_layer if not hasattr(self.base_layer, "get_base_layer") else self.base_layer.get_base_layer()
    
    def get_pretrained_matrix(self):
        base_w = self.get_base_layer().weight

        # try:
        #     base_w = dequantize_module_weight(self.get_base_layer())
        # except (TypeError, AttributeError):
        #     pass

        if base_w.dim() > 2:
            base_w = base_w.view(base_w.size(0), -1)
        elif base_w.dim() == 1:
            base_w = base_w.view(1, -1)

        if self.fan_in_fan_out:
            base_w = base_w.T  # transpose before doing SVD

        return base_w

class Linear(nn.Linear, UIOrthoLoRALayer):
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        *,
        uiortholora_alpha: float = 1.0,
        uiortholora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ) -> None:
        # Initialize nn.Linear with the base layer's parameters
        super(nn.Linear, self).__init__()
        # Initialize UIOrthoLoRALayer
        UIOrthoLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            scaling_factor=kwargs.pop("scaling_factor"),
            enforce_sv_positive=kwargs.pop("enforce_sv_positive"),
            uiortholora_alpha=uiortholora_alpha,
            uiortholora_dropout=uiortholora_dropout,
            initial_scaler=kwargs.pop("initial_scaler"),
            initial_sigma=kwargs.pop("initial_sigma"))

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Return ΔW = E[:,None] * (U @ diag(S) @ Vt) * D[None,:]
        Uses the learned parameters for merging.
        """
        D = self.uiortholora_D[adapter]                 # (in,)
        E = self.uiortholora_E[adapter]                 # (out,)

        # Use trainable=False to take learned sigma and rotations
        U, S, Vt = self._calc_tuner_internal(adapter)  # U:(out,r), S:(r,), Vt:(r,in)

        # Internal adapter matrix: (out,r) * (r,) -> broadcast, then @ (r,in) -> (out,in)
        internal = (U * S.unsqueeze(0)) @ Vt

        # Match dtype and device of the base layer weights
        w = self.get_base_layer().weight
        internal = internal.to(dtype=w.dtype, device=w.device)

        # Apply row and column scaling
        return (E[:, None] * internal) * D[None, :]


    def merge(self, *, safe_merge: bool = False, adapter_names: Optional[List[str]] = None):
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.uiortholora_sigma.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    print("GEtting delta weight")
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.uiortholora_sigma.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _calc_tuner_internal(self, adapter: str):
        U1 = getattr(self, f"{adapter}_U1")
        Vt1 = getattr(self, f"{adapter}_Vt1")
        S1 = getattr(self, f"{adapter}_S1")
        U2 = getattr(self, f"{adapter}_U2")
        Vt2 = getattr(self, f"{adapter}_Vt2")
        U3 = getattr(self, f"{adapter}_U3")
        Vt3 = getattr(self, f"{adapter}_Vt3")

        sigma = self.uiortholora_sigma[adapter]
        s2_size = self.medium_component_size - self.major_component_size
        S2 = sigma[:s2_size] if s2_size > 0 else torch.tensor([], device=self.device, dtype=self.dtype)
        S3 = sigma[s2_size:] if len(sigma) > s2_size else torch.tensor([], device=self.device, dtype=self.dtype)

        # Ensure S2/S3 match the dtype (sometimes sigma is initialized differently)
        S2 = S2.to(self.dtype)
        S3 = S3.to(self.dtype)

        if self.num_svectors_to_adapt > 0:
            left_orthogonal = self.uiortholora_left_unitary[adapter].weight.to(self.dtype)
            right_orthogonal = self.uiortholora_right_unitary[adapter].weight.to(self.dtype)
            
            # Cast inputs to self.dtype before matmul to avoid errors
            new_U3 = U3.to(self.dtype) @ left_orthogonal
            new_Vt3 = (Vt3.to(self.dtype).T @ right_orthogonal).T
        else:
            new_U3 = U3.to(self.dtype)
            new_Vt3 = Vt3.to(self.dtype)

        # Concatenate
        U_cat  = torch.cat([U1.to(self.dtype), U2.to(self.dtype), new_U3],  dim=1)
        S_cat  = torch.cat([S1.to(self.dtype), S2, S3])
        Vt_cat = torch.cat([Vt1.to(self.dtype), Vt2.to(self.dtype), new_Vt3], dim=0)

        # Final safety cast to ensure output is strictly BFloat16 (or whatever self.dtype is)
        return U_cat.to(self.dtype), S_cat.to(self.dtype), Vt_cat.to(self.dtype)
    

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        out = self.base_layer(x, *args, **kwargs)  # shape (..., out_features)

        for name in self.active_adapters:
            if name not in self.uiortholora_sigma:
                continue
            D = self.uiortholora_D[name].to(self.dtype)
            E = self.uiortholora_E[name].to(self.dtype)

            x_scaled = self.uiortholora_dropout[name](x) * D
            x_scaled = x_scaled.to(self.dtype)

            U, S, Vt = self._calc_tuner_internal(name)
            mid = torch.matmul(x_scaled, Vt.transpose(-2, -1))

            #multiply S from left with mid
            mid = mid * S
            mid = mid.to(self.dtype)

            y = torch.matmul(mid, U.transpose(-2, -1))

            out = out + y * E

        return out


    def get_base_layer(self):
        return self.base_layer

    def __repr__(self):
        return f"UIOrthoLoRALayer({self.get_base_layer().__repr__()})"

