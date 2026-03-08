from typing import Iterable, List, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ControlNet(nn.Module):
    """Simple ControlNet-style adapter that produces per-layer control tokens."""

    def __init__(
        self,
        hint_channels: int,
        model_channels: int,
        layers_num: int,
        base_channels: int = 64,
        control_scales: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.hint_channels = hint_channels
        self.model_channels = model_channels
        self.layers_num = layers_num

        self.input_hint = nn.Sequential(
            nn.Conv2d(hint_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.control_projections = nn.ModuleList(
            [nn.Conv2d(base_channels, model_channels, kernel_size=1) for _ in range(layers_num)]
        )

        if control_scales is None:
            control_scales = [1.0] * layers_num
        elif len(control_scales) == 1:
            control_scales = list(control_scales) * layers_num
        elif len(control_scales) != layers_num:
            raise ValueError("control_scales length must be 1 or match layers_num")

        self.register_buffer("control_scales", torch.tensor(control_scales, dtype=torch.float32))

    def forward(
        self,
        x: Iterable[torch.Tensor],
        hint: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        _ = timesteps
        _ = context
        if isinstance(x, torch.Tensor):
            layer_feats = [x] * self.layers_num
        else:
            layer_feats = list(x)

        if len(layer_feats) != self.layers_num:
            raise ValueError("Expected features for each layer to match layers_num")

        base_hint = self.input_hint(hint)
        controls: List[torch.Tensor] = []
        for idx, feat in enumerate(layer_feats):
            if feat.dim() != 3:
                raise ValueError("ControlNet expects patch token tensors with shape [B, L, D]")
            bsz, length, dim = feat.shape
            grid = int((length - 1) ** 0.5)
            if grid * grid + 1 != length:
                raise ValueError("Patch token length must be 1 + grid_size**2")

            control_map = F.interpolate(
                base_hint,
                size=(grid, grid),
                mode="bilinear",
                align_corners=False,
            )
            control_map = self.control_projections[idx](control_map)
            control_map = control_map * self.control_scales[idx]

            control_tokens = control_map.flatten(2).transpose(1, 2)
            class_token = torch.zeros(
                bsz,
                1,
                dim,
                dtype=control_tokens.dtype,
                device=control_tokens.device,
            )
            control_tokens = torch.cat([class_token, control_tokens], dim=1)
            controls.append(control_tokens)

        return controls