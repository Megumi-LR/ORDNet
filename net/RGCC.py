import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class LongRangeLowFreq(nn.Module):
    """可靠性门控的低分辨率非局部聚合，作为 corr_lf 的“长程借取”版本。
    Q/K 来自 base(内容描述子)，V = corr(被借取的校正)；key 按【源可靠性 g】门控
    -> 只从高可信位置借取平滑校正。在低分辨率上做(低频本就不需要高分辨率，且压成本)。
    """
    def __init__(self, dim, lr_size=32, key_dim=None):
        super(LongRangeLowFreq, self).__init__()
        self.lr_size = lr_size
        kd = key_dim if key_dim is not None else max(dim // 4, 8)
        self.q = nn.Conv2d(dim, kd, 1, bias=False)
        self.k = nn.Conv2d(dim, kd, 1, bias=False)
        self.scale = kd ** -0.5

    def forward(self, corr, base, g):
        B, C, H, W = corr.shape
        th, tw = min(H, self.lr_size), min(W, self.lr_size)
        cb = F.adaptive_avg_pool2d(base, (th, tw))
        cv = F.adaptive_avg_pool2d(corr, (th, tw))
        gd = F.adaptive_avg_pool2d(g, (th, tw)).clamp(1e-4, 1.0)   # 源可靠性
        q = self.q(cb).flatten(2)                       # [B,kd,N]
        k = (self.k(cb) * gd).flatten(2)                # [B,kd,N]  g 门控 key
        v = cv.flatten(2)                               # [B,C,N]
        attn = torch.softmax((q.transpose(1, 2) @ k) * self.scale, dim=-1)  # [B,N,N]
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(B, C, th, tw)
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)


class RGCC(nn.Module):
    """Reliability-Gated Cross-modal Correction (替代 baseline 的 MAFM 融合)。
    互补基底 (x+y) + 可靠性门控的跨模态校正 g·corr。
    """
    def __init__(self, dim, reduction=8, use_longrange=False, hf_floor=0.0):
        super(RGCC, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        # 跨模态校正：从两支差异生成校正方向（亮度引导修正色度）
        self.conv_corr = nn.Conv2d(dim, dim, 1, bias=True)
        # 可选：corr_lf 改用“长程借取”(低分辨率 + g 门控空间注意力)；默认关闭=局部平滑
        self.use_longrange = use_longrange
        if use_longrange:
            self.lr_block = LongRangeLowFreq(dim)
        # 流特异高频门控强度：g_hf = hf_floor + (1-hf_floor)*g
        #   hf_floor=0.0(默认) -> 色度流“重门控”：低可信处高频几乎清零，强杀伪色/色噪；
        #   hf_floor>0          -> “轻门控”：低可信处仍保留 hf_floor 比例高频(留细节)。
        # 纯标量、无 nn.Parameter -> 不改变 state_dict，旧 checkpoint 照常加载。
        self.hf_floor = float(hf_floor)

    def forward(self, x, y, reliability=None):
        # 互补基底：两支信息都保留，永不丢弃任一模态
        base = x + y

        # 内容注意力（沿用 baseline 的 CA/SA/PA）：决定“在哪里校正”(WHERE)
        cattn = self.ca(base)
        cattn = base + cattn
        sattn = self.sa(base)
        sattn = base + sattn
        pattn1 = self.sigmoid(self.pa(base, cattn))
        pattn2 = self.sigmoid(self.pa(base, sattn))
        content_weight = self.alpha * pattn1 + self.beta * pattn2

        # 跨模态校正：两支差异作为校正方向，内容注意力做空间定位
        corr = content_weight * self.conv_corr(y - x)

        # 频率分离门控（解耦“强恢复暗区”与“抑制不可信信息”）：
        #   低频 corr_lf -> 安全的“真内容”，永远全施加（恢复暗区，可选长程借取）；
        #   高频 corr_hf = 细节，也是噪声藏身处 -> 仅这部分被 g 门控（暗噪区压掉，不注入色噪）。
        if reliability is not None:
            g = F.interpolate(
                reliability, size=x.shape[2:], mode='bilinear', align_corners=False)
            if self.use_longrange:
                corr_lf = self.lr_block(corr, base, g)   # 长程借取（低分辨率 + g 门控 key）
            else:
                corr_lf = F.avg_pool2d(corr, kernel_size=5, stride=1, padding=2)  # 局部平滑
            corr_hf = corr - corr_lf
            g_hf = self.hf_floor + (1.0 - self.hf_floor) * g  # 流特异：色度重门控/亮度轻门控
            result = base + corr_lf + g_hf * corr_hf   # 低频不门控；高频按可靠性门控
        else:
            result = base + corr

        result = self.conv(result)
        return result

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) 
        pattn1 = pattn1.unsqueeze(dim=2) 
        x2 = torch.cat([x, pattn1], dim=2) 
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
