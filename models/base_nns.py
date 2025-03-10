import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
    def get_last_layer(self, x):
        return self.layers[1](self.layers[0](x))


class CLF(nn.Module):
    def __init__(self, int_dim, out_dim, bias=False):
        super(CLF, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(int_dim, int_dim),
            nn.ReLU(),
            nn.Linear(int_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Proj(nn.Module):
    def __init__(self, in_dim, out_dim, norm, scaler, bias=True):
        super(Proj, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm, self.scaler = norm, scaler
        if norm and scaler is not None:
            self.scale = torch.nn.Parameter(torch.Tensor([scaler]), requires_grad=True)

    def forward(self, x):
        x = self.layer(x)
        if self.norm and self.scaler is None:
            x = torch.nn.functional.normalize(x) * self.scale

        return x


class MSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x