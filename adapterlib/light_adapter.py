import torch
import torch.nn as nn

class LightweightBottleneckAdapter(nn.Module):
    def __init__(self, d_model=768, bottleneck_dim=64):
        super().__init__()
        self.d_model = d_model

        # 1. 降维全连接层
        self.down_proj = nn.Linear(d_model, bottleneck_dim)

        # 2. 局部感知 DW-Conv (专攻螺栓等高频细节)
        self.dw_conv = nn.Conv2d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            kernel_size=3,
            padding=1,
            groups=bottleneck_dim
        )

        self.act = nn.GELU()

        # 3. 升维全连接层
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # ⭐ 核心修复：零初始化。保证训练初始状态模型等价于原 CLIP
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, hw_shape=None):
        B, N, C = x.shape
        z = self.down_proj(x)

        if hw_shape is not None:
            H, W = hw_shape
            cls_token, spatial_tokens = z[:, :1, :], z[:, 1:, :]
            spatial_tokens = spatial_tokens.transpose(1, 2).reshape(B, -1, H, W)
            spatial_tokens = self.dw_conv(spatial_tokens)
            spatial_tokens = spatial_tokens.flatten(2).transpose(1, 2)
            z = torch.cat([cls_token, spatial_tokens], dim=1)

        z = self.act(z)
        return self.up_proj(z)