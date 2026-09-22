import torch
import torch.nn as nn
import torch.nn.functional as F


class ORPE(nn.Module):
    """ORPE - Oklab Reliability Prior Estimator（Oklab 可靠性先验估计器）。
    输出逐像素可靠度 g∈[0,1]，供 RGCA(K 抑制) 与 RGCC(频率门控) 共享。
    多证据 4 信号: [亮度 sig_lc, 局部SNR sig_snr, 相干真彩 sig_coherence, 边缘 sig_edge]；
    其中 sig_coherence(色彩相干性) 能区分"真彩 vs 色噪"——单一强度 SNR 原理上做不到。
    注：色度幅度 sig_chroma 不单独进 g（对色噪投正票、有害且与 coher 冗余），
        只作为 sig_coherence 的门控（coher = coherence × sig_chroma）。
    """
    def __init__(self, window_size=5, bias=False, eps=1e-6, num_scales=4):
        super(ORPE, self).__init__()
        pad = window_size // 2
        self.eps = eps
        self.register_buffer(
            'avg_kernel',
            torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        )
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)
        )
        # 聚合：每尺度独立的可学习 softmax 加权 + 固定对比拉伸。
        # signal_weights[scale] 是该 UNet 层级专属的一组权重 -> 每层可侧重不同信号。
        # 信号在各尺度上分别重算（forward 的 lc/ab 已是该尺度下采样版），凸组合不可塌缩。
        # 顺序: [亮度, 局部SNR, 相干真彩, 边缘]，各尺度初始化均匀(各 0.25)。
        self.num_scales = num_scales
        self.signal_weights = nn.Parameter(torch.zeros(num_scales, 4))
        self.gain = 8.0
        self.mid = 0.5
        self.pad = pad

    def forward(self, lc, ab, scale=0):
        kernel = self.avg_kernel.to(dtype=lc.dtype)
        sx = self.sobel_x.to(dtype=lc.dtype)
        sy = self.sobel_y.to(dtype=lc.dtype)
        pad = self.pad

        # 信号1：亮度（保留，作为独立证据）
        sig_lc = lc

        # 信号2：局部相对平滑/洁净度。平滑且干净 -> σ 小 -> μ/σ 大 -> 高；
        # 噪声区 -> σ 大 -> 低。自动拒绝“被噪声搅乱的假平滑”。
        mu = F.conv2d(lc, kernel, padding=pad)
        mu2 = F.conv2d(lc ** 2, kernel, padding=pad)
        var = (mu2 - mu ** 2).clamp(min=0)
        std = (var + self.eps).sqrt()
        sig_snr = torch.sigmoid(mu / std - 1.0)

        # 信号3：色度幅度（有没有颜色）
        chroma = ab.norm(dim=1, keepdim=True)
        sig_chroma = torch.sigmoid(chroma * 20.0 - 0.5)

        # 信号4：相干真彩 = 色度相干度 ∧ 色度幅度。真实暗物体 ab 成片同向 -> R≈1；
        # 随机色噪互相抵消 -> R≈0；用色度幅度门控消除灰区 0/0 误报。
        mean_a = F.conv2d(ab[:, 0:1], kernel, padding=pad)
        mean_b = F.conv2d(ab[:, 1:2], kernel, padding=pad)
        mag_of_mean = (mean_a ** 2 + mean_b ** 2 + self.eps).sqrt()
        mean_of_mag = F.conv2d(chroma, kernel, padding=pad)
        coherence = (mag_of_mean / (mean_of_mag + self.eps)).clamp(0.0, 1.0)
        sig_coherence = (coherence * sig_chroma).clamp(0.0, 1.0)

        # 信号5：边缘/结构强度
        gx = F.conv2d(lc, sx, padding=1)
        gy = F.conv2d(lc, sy, padding=1)
        sig_edge = torch.sigmoid((gx ** 2 + gy ** 2 + 1e-8).sqrt() * 15.0 - 0.5)

        # 聚合：该尺度专属 softmax 权重对 4 信号做凸组合 + 固定对比拉伸（不可塌缩）。
        # 注：sig_chroma 不进 g（避免对色噪投正票），只作为 sig_coherence 的门控。
        sigs = torch.cat([sig_lc, sig_snr,
                          sig_coherence, sig_edge], dim=1)        # [B,4,H,W]
        w = torch.softmax(self.signal_weights[scale], dim=0).view(1, 4, 1, 1)
        g = (sigs * w).sum(dim=1, keepdim=True)                  # 凸组合（权重和=1）
        g = torch.sigmoid(self.gain * (g - self.mid))           # 固定对比拉伸 -> 全程
        return g.clamp(0.0, 1.0)
