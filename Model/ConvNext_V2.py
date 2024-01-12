# %%
from numpy import outer
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam
import math

# %%
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """


    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
 
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
 
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)  # weight bias对应γ β
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height]
            # 对channels 维度求均值
            mean = x.mean(1, keepdim=True)
            # 方差
            var = (x - mean).pow(2).mean(1, keepdim=True)
            # 减均值，除以标准差的操作
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_rate=0.2, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # gamma 针对layer scale的操作
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()  # nn.Identity() 恒等映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 2, 1)
        
        x = shortcut + self.drop_path(x)

        return x

# %%
# ConvNextBlock(8)(torch.randn(3, 8, 2048)).shape
# %%
class Downsample(nn.Module):
    def __init__(self, in_chans, dim) -> None:
        super(Downsample, self).__init__()
        self.norm = LayerNorm(in_chans, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv1d(in_chans, dim, kernel_size=4, stride=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.conv(x)
        # x = x.permute(0, 2, 1)
        return x
# %%
class ConvNext_T(nn.Module):
    def __init__(self, depths = [1,1,1,1]) -> None:
        super(ConvNext_T, self).__init__()

        blocks = []
        self.stem = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=7, stride=4, padding=3),
            LayerNorm(24, eps=1e-6, data_format="channels_first"),
        )
        blocks.append(self.stem)

        for _ in range(depths[0]):
            blocks.append(ConvNextBlock(24))
        
        blocks.append(Downsample(24, 72))
        for _ in range(depths[1]):
            blocks.append(ConvNextBlock(72))
        
        blocks.append(Downsample(72, 216))
        for _ in range(depths[2]):
            blocks.append(ConvNextBlock(216))
        
        blocks.append(Downsample(216, 648))
        for _ in range(depths[3]):
            blocks.append(ConvNextBlock(648))
        

        self.blocks = nn.Sequential(*blocks)

        # 定义分类层
        self.Dropout_1 = nn.Dropout(0.6)
        self.Dropout_2 = nn.Dropout(0.6)
        self.linear_1 = nn.Linear(648*19, 128)
        self.linear_2 = nn.Linear(128, 30)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        out = out.view(out.size(0), -1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.Dropout_1(out)
        out = self.linear_2(out)
        # out = self.Dropout_2(out)

        return out
# %%
ConvNext_T()(torch.randn(1, 1, 2048)).shape
# %%