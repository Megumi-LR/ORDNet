import torch
import torch.nn as nn
import torch.nn.functional as F
from net.OKLab_transform import RGB_OKLab
from net.transformer_utils import NormDownsample, NormUpsample
from net.RGCA import RGCA_AB, RGCA_LC
from net.RGCC import RGCC
from net.ORPE import ORPE

class ORDNet(nn.Module):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 alpha_l=0.5
        ):
        super(ORDNet, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # The released checkpoints use dilation=1 in both streams.
        ab_dil = 1

        # ab branch（色度流：空洞，大感受野）
        self.ab_enc_block0 = nn.Sequential(
            nn.ReplicationPad2d(ab_dil),
            nn.Conv2d(2, ch1, 3, stride=1, padding=0, dilation=ab_dil, bias=False)
            )
        self.ab_enc_block1 = NormDownsample(ch1, ch2, scale=0.5, use_norm = norm, dilation=ab_dil)
        self.ab_enc_block2 = NormDownsample(ch2, ch3, scale=0.5, use_norm = norm, dilation=ab_dil)
        self.ab_enc_block3 = NormDownsample(ch3, ch4, scale=0.5, use_norm = norm, dilation=ab_dil)

        self.ab_dec_block3 = NormUpsample(ch4, ch3, scale=2.0, use_norm = norm, dilation=ab_dil)
        self.ab_dec_block2 = NormUpsample(ch3, ch2, scale=2.0, use_norm = norm, dilation=ab_dil)
        self.ab_dec_block1 = NormUpsample(ch2, ch1, scale=2.0, use_norm = norm, dilation=ab_dil)
        self.ab_dec_block0 = nn.Sequential(
            nn.ReplicationPad2d(ab_dil),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, dilation=ab_dil, bias=False)
        )
        
        
        # Lc branch
        self.lc_enc_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.lc_enc_block1 = NormDownsample(ch1, ch2, scale=0.5, use_norm = norm)
        self.lc_enc_block2 = NormDownsample(ch2, ch3, scale=0.5, use_norm = norm)
        self.lc_enc_block3 = NormDownsample(ch3, ch4, scale=0.5, use_norm = norm)
        
        self.lc_dec_block3 = NormUpsample(ch4, ch3, scale=2.0,use_norm = norm)
        self.lc_dec_block2 = NormUpsample(ch3, ch2, scale=2.0,use_norm = norm)
        self.lc_dec_block1 = NormUpsample(ch2, ch1, scale=2.0,use_norm = norm)
        self.lc_dec_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.RGCA_AB1 = RGCA_AB(ch2, head2)
        self.RGCA_AB2 = RGCA_AB(ch3, head3)
        self.RGCA_AB3 = RGCA_AB(ch4, head4)
        self.RGCA_AB4 = RGCA_AB(ch4, head4)
        self.RGCA_AB5 = RGCA_AB(ch3, head3)
        self.RGCA_AB6 = RGCA_AB(ch2, head2)
        
        self.RGCA_LC1 = RGCA_LC(ch2, head2, hf_floor=alpha_l)
        self.RGCA_LC2 = RGCA_LC(ch3, head3, hf_floor=alpha_l)
        self.RGCA_LC3 = RGCA_LC(ch4, head4, hf_floor=alpha_l)
        self.RGCA_LC4 = RGCA_LC(ch4, head4, hf_floor=alpha_l)
        self.RGCA_LC5 = RGCA_LC(ch3, head3, hf_floor=alpha_l)
        self.RGCA_LC6 = RGCA_LC(ch2, head2, hf_floor=alpha_l)
        
        self.trans = RGB_OKLab()
        self.reliability_estimator = ORPE(window_size=5, num_scales=4)
        self.fusion1 = RGCC(ch1, use_longrange=True, hf_floor=0.0)
        self.fusion2 = RGCC(ch2, use_longrange=True, hf_floor=0.0)
        self.fusion3 = RGCC(ch3, use_longrange=True, hf_floor=0.0)
        self.fusion4 = RGCC(ch4, use_longrange=True, hf_floor=0.0)


    def forward(self, x):
        dtypes = x.dtype
        oklab = self.trans.rgb_to_oklab(x)
        lc = oklab[:,2,:,:].unsqueeze(1).to(dtypes)
        ab = oklab[:,0:2,:,:].to(dtypes)
        ab_raw = ab
        lc1 = F.avg_pool2d(lc, 2); ab1 = F.avg_pool2d(ab_raw, 2)
        lc2 = F.avg_pool2d(lc, 4); ab2 = F.avg_pool2d(ab_raw, 4)
        lc3 = F.avg_pool2d(lc, 8); ab3 = F.avg_pool2d(ab_raw, 8)
        g3 = self.reliability_estimator(lc3, ab3, scale=3)
        g2 = self.reliability_estimator(lc2, ab2, scale=2)
        g1 = self.reliability_estimator(lc1, ab1, scale=1)
        g0 = self.reliability_estimator(lc, ab_raw, scale=0)
        ga0, ga1, ga2, ga3 = g0, g1, g2, g3

        lc_enc0 = self.lc_enc_block0(lc)
        ab_0 = self.ab_enc_block0(ab)
        ab_0 = self.fusion1(ab_0, lc_enc0, reliability=g0)
        # low
        lc_enc1 = self.lc_enc_block1(lc_enc0)
        ab_1 = self.ab_enc_block1(ab_0)
        ab_1 = self.fusion2(ab_1, lc_enc1, reliability=g1)
        lc_jump0 = lc_enc0
        ab_jump0 = ab_0

        lc_enc2 = self.RGCA_LC1(lc_enc1, ab_1, reliability=ga1)
        ab_2 = self.RGCA_AB1(ab_1, lc_enc1, reliability=ga1)
        ab_2 = self.fusion2(ab_2, lc_enc2, reliability=g1)
        lc_jump1 = lc_enc2
        ab_jump1 = ab_2
        lc_enc2 = self.lc_enc_block2(lc_enc2)
        ab_2 = self.ab_enc_block2(ab_2)
        ab_2 = self.fusion3(ab_2, lc_enc2, reliability=g2)

        lc_enc3 = self.RGCA_LC2(lc_enc2, ab_2, reliability=ga2)
        ab_3 = self.RGCA_AB2(ab_2, lc_enc2, reliability=ga2)
        ab_3 = self.fusion3(ab_3, lc_enc3, reliability=g2)
        lc_jump2 = lc_enc3
        ab_jump2 = ab_3
        lc_enc3 = self.lc_enc_block3(lc_enc2)
        ab_3 = self.ab_enc_block3(ab_2)
        ab_3 = self.fusion4(ab_3, lc_enc3, reliability=g3)

        lc_enc4 = self.RGCA_LC3(lc_enc3, ab_3, reliability=ga3)
        ab_4 = self.RGCA_AB3(ab_3, lc_enc3, reliability=ga3)
        ab_4 = self.fusion4(ab_4, lc_enc4, reliability=g3)

        lc_dec4 = self.RGCA_LC4(lc_enc4, ab_4, reliability=ga3)
        ab_4 = self.RGCA_AB4(ab_4, lc_enc4, reliability=ga3)
        ab_4 = self.fusion4(ab_4, lc_dec4, reliability=g3)

        ab_3 = self.ab_dec_block3(ab_4, ab_jump2)
        lc_dec3 = self.lc_dec_block3(lc_dec4, lc_jump2)
        ab_3 = self.fusion3(ab_3, lc_dec3, reliability=g2)
        lc_dec2 = self.RGCA_LC5(lc_dec3, ab_3, reliability=ga2)
        ab_2 = self.RGCA_AB5(ab_3, lc_dec3, reliability=ga2)
        ab_2 = self.fusion3(ab_2, lc_dec2, reliability=g2)

        ab_2 = self.ab_dec_block2(ab_2, ab_jump1)
        lc_dec2 = self.lc_dec_block2(lc_dec3, lc_jump1)
        ab_2 = self.fusion2(ab_2, lc_dec2, reliability=g1)

        lc_dec1 = self.RGCA_LC6(lc_dec2, ab_2, reliability=ga1)
        ab_1 = self.RGCA_AB6(ab_2, lc_dec2, reliability=ga1)
        ab_1 = self.fusion2(ab_1, lc_dec1, reliability=g1)

        lc_dec1 = self.lc_dec_block1(lc_dec1, lc_jump0)
        ab_1 = self.ab_dec_block1(ab_1, ab_jump0)
        ab_1 = self.fusion1(ab_1, lc_dec1, reliability=g0)
        lc_dec0 = self.lc_dec_block0(lc_dec1)
        ab_0 = self.ab_dec_block0(ab_1)
        
        output_oklab = torch.cat([ab_0, lc_dec0], dim=1) + oklab
        output_rgb = self.trans.oklab_to_rgb(output_oklab)

        return output_rgb
    
    def rgb_to_oklab(self, x):
        oklab = self.trans.rgb_to_oklab(x)
        return oklab
    

    
