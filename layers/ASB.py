import torch
import torch.nn as nn


class ImprovedASB(nn.Module):
    """Improved Adaptive Spectral Block: per-channel gating + band attention + adaptive threshold.

    Performs frequency-domain enhancement as a pre-processing step before
    the main attention mechanism.
    """
    def __init__(self, d_model, d_channel, n_bands=4):
        super().__init__()
        self.d_model = d_model
        self.d_channel = d_channel
        freq_bins = d_model // 2 + 1

        # Adaptive threshold (learnable, per-channel)
        self.threshold = nn.Parameter(torch.zeros(1, d_channel, 1))

        # Global filter (learns global frequency weights on raw spectrum)
        self.global_weight = nn.Parameter(
            torch.randn(1, d_channel, freq_bins, 2) * 0.02
        )

        # Local filter (learns frequency weights on denoised spectrum)
        self.local_weight = nn.Parameter(
            torch.randn(1, d_channel, freq_bins, 2) * 0.02
        )

        # Per-channel residual gate (initialized ~0.88 via sigmoid(2.0))
        self.channel_gate = nn.Parameter(torch.ones(1, d_channel, 1) * 2.0)

        # Band attention (squeeze-excitation style)
        self.band_attn = nn.Sequential(
            nn.Linear(freq_bins, n_bands),
            nn.ReLU(),
            nn.Linear(n_bands, freq_bins),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, d_model]
        residual = x

        # FFT
        X_freq = torch.fft.rfft(x, dim=-1)  # [B, C, freq_bins] complex

        # Adaptive soft mask (denoising)
        power = X_freq.abs()  # [B, C, freq_bins]
        mask = torch.sigmoid(power - torch.exp(self.threshold))
        X_filtered = X_freq * mask

        # Band attention weights
        band_w = self.band_attn(power)  # [B, C, freq_bins]
        X_filtered = X_filtered * band_w

        # Global filter (on raw spectrum)
        g_w = torch.view_as_complex(self.global_weight.contiguous())
        F_global = X_freq * g_w

        # Local filter (on denoised spectrum)
        l_w = torch.view_as_complex(self.local_weight.contiguous())
        F_local = X_filtered * l_w

        # Merge + IFFT
        F_out = F_global + F_local
        enhanced = torch.fft.irfft(F_out, n=x.shape[-1])

        # Per-channel gated residual
        gate = torch.sigmoid(self.channel_gate)  # ~0.88
        output = gate * enhanced + (1 - gate) * residual

        return output
