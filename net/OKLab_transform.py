import torch
import torch.nn as nn


class RGB_OKLab(nn.Module):
    def __init__(self):
        super(RGB_OKLab, self).__init__()
        self.compression_k = torch.nn.Parameter(torch.full([1], 0.2))
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.this_k = 0
        self.ab_scale = 0.5

    def _compression_gamma(self):
        gamma = torch.clamp(torch.exp(self.compression_k), min=1.0, max=4.0)
        self.this_k = gamma.detach().item()
        return gamma

    @staticmethod
    def _srgb_to_linear(img):
        threshold = 0.04045
        return torch.where(
            img <= threshold,
            img / 12.92,
            ((img + 0.055) / 1.055).pow(2.4),
        )

    @staticmethod
    def _linear_to_srgb(img):
        img = torch.clamp(img, min=0.0)
        threshold = 0.0031308
        return torch.where(
            img <= threshold,
            img * 12.92,
            1.055 * torch.clamp(img, min=threshold).pow(1.0 / 2.4) - 0.055,
        )

    @staticmethod
    def _rgb_to_oklab(img):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]

        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = torch.clamp(l, min=1e-8).pow(1.0 / 3.0)
        m_ = torch.clamp(m, min=1e-8).pow(1.0 / 3.0)
        s_ = torch.clamp(s, min=1e-8).pow(1.0 / 3.0)

        lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        return lab_l, lab_a, lab_b

    @staticmethod
    def _oklab_to_rgb(lab_l, lab_a, lab_b):
        l_ = lab_l + 0.3963377774 * lab_a + 0.2158037573 * lab_b
        m_ = lab_l - 0.1055613458 * lab_a - 0.0638541728 * lab_b
        s_ = lab_l - 0.0894841775 * lab_a - 1.2914855480 * lab_b

        l_ = torch.clamp(l_, -2.0, 2.0)
        m_ = torch.clamp(m_, -2.0, 2.0)
        s_ = torch.clamp(s_, -2.0, 2.0)

        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_

        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        return torch.cat([r, g, b], dim=1)

    def rgb_to_oklab(self, img):
        img_linear = self._srgb_to_linear(torch.clamp(img, 0.0, 1.0))
        lab_l, lab_a, lab_b = self._rgb_to_oklab(img_linear)

        gamma = self._compression_gamma()
        compressed_l = torch.clamp(lab_l, 0.0, 1.0).pow(gamma.view(1, 1, 1, 1))

        a_plane = torch.clamp(lab_a / self.ab_scale, -1.0, 1.0)
        b_plane = torch.clamp(lab_b / self.ab_scale, -1.0, 1.0)
        return torch.cat([a_plane, b_plane, compressed_l], dim=1)

    def oklab_to_rgb(self, img):
        a_plane = torch.clamp(img[:, 0:1], -1.0, 1.0)
        b_plane = torch.clamp(img[:, 1:2], -1.0, 1.0)
        compressed_l = torch.clamp(img[:, 2:3], 0.0, 1.0)

        if self.gated:
            a_plane = a_plane * 1.3
            b_plane = b_plane * 1.3

        gamma = self._compression_gamma()
        lab_l = torch.clamp(compressed_l, min=1e-6).pow(
            1.0 / gamma.view(1, 1, 1, 1))
        lab_l = torch.clamp(lab_l, 0.0, 1.0)
        lab_a = a_plane * self.ab_scale
        lab_b = b_plane * self.ab_scale

        rgb_linear = self._oklab_to_rgb(lab_l, lab_a, lab_b)
        rgb = torch.clamp(self._linear_to_srgb(rgb_linear), 0.0, 1.0)

        if self.gated2:
            rgb = torch.clamp(rgb * self.alpha, 0.0, 1.0)
        return rgb
