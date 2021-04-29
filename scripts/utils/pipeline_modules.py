import torch
import torch.nn.functional as F

from utils.stft import STFT


class NetFeeder(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.eps = torch.finfo(torch.float32).eps
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, mix, sph):
        real_mix, imag_mix = self.stft.stft(mix)
        mag_mix = torch.sqrt(real_mix**2 + imag_mix**2)
        feat = mag_mix
        
        real_sph, imag_sph = self.stft.stft(sph)
        mag_sph = torch.sqrt(real_sph**2 + imag_sph**2)
        lbl = mag_sph

        return feat, lbl


class Resynthesizer(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, est, mix):
        real_mix, imag_mix = self.stft.stft(mix)
        pha_mix = torch.atan2(imag_mix.data, real_mix.data)
        real_est = est * torch.cos(pha_mix)
        imag_est = est * torch.sin(pha_mix)
        sph_est = self.stft.istft(torch.stack([real_est, imag_est], dim=1))
        sph_est = F.pad(sph_est, [0, mix.shape[1]-sph_est.shape[1]])

        return sph_est
