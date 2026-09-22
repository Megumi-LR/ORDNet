import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import *

class RGCA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(RGCA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.linear = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, y, reliability=None):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        if reliability is not None:
            rel_down = F.interpolate(
                reliability, size=(h, w), mode='bilinear', align_corners=False)
            rel_w = torch.clamp(
                rel_down.flatten(2).unsqueeze(1), min=1e-4, max=1.0).to(
                dtype=k.dtype, device=k.device)
            k = F.normalize(k * rel_w, dim=-1, eps=1e-12)
        else:
            k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.nn.functional.softmax(attn, dim=-1)

        z_t = (attn @ v)
        z_t = rearrange(z_t, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        t_prime = self.linear(z_t)  # T'_T
        t_1 = self.beta * t_prime + self.alpha * y

        ffn_out = self.ffn(t_1)
        t_2 = self.gamma * t_1 + self.delta * ffn_out

        t_2 = self.project_out(t_2)
        return t_2
    

class MFEM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(MFEM, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias),    
            nn.ReLU(inplace=True),   
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),           
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(1, 3), padding=(0, 1), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 1), padding=(1, 0), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3, dilation=3, groups=hidden_features, bias=bias),
            nn.ReLU(inplace=True),
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),           
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 1), padding=(1, 0), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(1, 3), padding=(0, 1), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3, dilation=3, groups=hidden_features, bias=bias),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
        )
 
        self.conv_cat = nn.Conv2d(hidden_features * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv_cat(x_cat)
        return out


class RGCA_AB(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(RGCA_AB, self).__init__()
        self.gdfn = MFEM(dim)
        self.norm = LayerNorm(dim)
        self.ffn = RGCA(dim, num_heads, bias)

    def forward(self, x, y, reliability=None):
        # 仅 K 抑制(RGCA 内部)，无输出门控
        x = x + self.ffn(self.norm(x), self.norm(y), reliability=reliability)
        x = x + self.gdfn(self.norm(x))
        return x

class RGCA_LC(nn.Module):
    def __init__(self, dim, num_heads, bias=False, hf_floor=0.5):
        super(RGCA_LC, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = MFEM(dim)
        self.ffn = RGCA(dim, num_heads, bias)
        # 亮度流“轻门控”地板：g_hf = hf_floor + (1-hf_floor)*g
        # 低频(曝光/亮度恢复)永不门控；高频(纹理/细节,噪声藏身处)仅轻门控，
        # 低可信暗区也保留 hf_floor 比例细节修正 -> 既抑噪又不欠增强。纯标量、无参数。
        self.hf_floor = float(hf_floor)

    def forward(self, x, y, reliability=None):
        corr = self.ffn(self.norm(x), self.norm(y), reliability=reliability)
        if reliability is not None:
            g = F.interpolate(reliability, size=corr.shape[2:],
                              mode='bilinear', align_corners=False)
            corr_lf = F.avg_pool2d(corr, kernel_size=5, stride=1, padding=2)  # 低频=亮度恢复
            corr_hf = corr - corr_lf                                          # 高频=细节/噪声
            g_hf = self.hf_floor + (1.0 - self.hf_floor) * g                  # 轻门控
            corr = corr_lf + g_hf * corr_hf                                   # 低频不门控
        x = x + corr
        x = x + self.gdfn(self.norm(x))
        return x
