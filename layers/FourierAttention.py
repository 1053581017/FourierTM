import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class FourierEmbedding(nn.Module):
    """FFT-based multi-band tokenization, replacing SWT.

    Splits the frequency spectrum into m+1 bands (low-to-high),
    producing output shape [B, C, m+1, d_model] identical to SWT.
    """
    def __init__(self, d_channel=16, m=2, decompose=True):
        super().__init__()
        self.d_channel = d_channel
        self.m = m
        self.decompose = decompose

    def forward(self, x):
        if self.decompose:
            return self.fft_decomposition(x)
        else:
            return self.fft_reconstruction(x)

    def fft_decomposition(self, x):
        # x: [B, C, d_model]
        n = x.shape[-1]
        X_freq = torch.fft.rfft(x, dim=-1)  # [B, C, freq_bins]
        freq_bins = X_freq.shape[-1]
        n_bands = self.m + 1

        boundaries = self._get_band_boundaries(freq_bins, n_bands, X_freq.device)

        band_signals = []
        for i in range(n_bands):
            mask = torch.zeros(freq_bins, device=X_freq.device, dtype=X_freq.dtype)
            start = boundaries[i]
            end = boundaries[i + 1]
            mask[start:end] = 1.0
            band_freq = X_freq * mask.unsqueeze(0).unsqueeze(0)
            band_time = torch.fft.irfft(band_freq, n=n)
            band_signals.append(band_time)

        return torch.stack(band_signals, dim=-2)  # [B, C, m+1, d_model]

    def fft_reconstruction(self, coeffs):
        # coeffs: [B, C, m+1, d_model]
        return coeffs.sum(dim=-2)  # [B, C, d_model]

    def _get_band_boundaries(self, freq_bins, n_bands, device):
        """Logarithmic band boundaries: finer at low frequencies."""
        if freq_bins <= n_bands:
            boundaries = torch.linspace(0, freq_bins, n_bands + 1, device=device).long()
            return boundaries

        log_boundaries = torch.logspace(0, torch.log10(torch.tensor(float(freq_bins))),
                                        n_bands + 1, device=device)
        log_boundaries = (log_boundaries - 1.0) / (log_boundaries[-1] - 1.0) * freq_bins
        boundaries = log_boundaries.long()
        boundaries[0] = 0
        boundaries[-1] = freq_bins

        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 1

        return boundaries


# ============================================================
# Attention Mechanisms
# ============================================================

class DualPathGeomAttention(nn.Module):
    """Dual-path attention: dot-product and wedge-product computed separately, then fused.
    Takes 4 inputs: (Q, K, V_dot, V_wedge).
    """
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, alpha=1.0):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.alpha = alpha
        # Initialize fusion_gate from alpha: sigmoid(gate) ≈ alpha (wedge weight)
        # so gate = logit(alpha). Clamp alpha to avoid inf.
        alpha_clamped = max(min(alpha, 0.999), 0.001)
        init_val = torch.log(torch.tensor(alpha_clamped / (1 - alpha_clamped)))
        # gate controls wedge weight: output = (1-g)*dot + g*wedge
        self.fusion_gate = nn.Parameter(init_val)

    def forward(self, queries, keys, values_dot, values_wedge):
        B, L, H, E = queries.shape
        _, S, _, D = values_dot.shape
        scale = self.scale or 1. / sqrt(E)

        # Path A: Dot-product
        dot_scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Path B: Wedge-product
        queries_norm2 = torch.sum(queries ** 2, dim=-1).permute(0, 2, 1).unsqueeze(-1)
        keys_norm2 = torch.sum(keys ** 2, dim=-1).permute(0, 2, 1).unsqueeze(-2)
        wedge_norm2 = F.relu(queries_norm2 * keys_norm2 - dot_scores ** 2)
        wedge_scores = torch.sqrt(wedge_norm2 + 1e-8)

        dot_scores = dot_scores * scale
        wedge_scores = wedge_scores * scale

        if self.mask_flag:
            attn_mask = torch.tril(torch.ones(L, S, device=queries.device))
            dot_scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
            wedge_scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        dot_attn = self.dropout(torch.softmax(dot_scores, dim=-1))
        wedge_attn = self.dropout(torch.softmax(wedge_scores, dim=-1))

        dot_out = torch.einsum("bhls,bshd->blhd", dot_attn, values_dot)
        wedge_out = torch.einsum("bhls,bshd->blhd", wedge_attn, values_wedge)

        # g = sigmoid(fusion_gate) ≈ alpha initially
        # Matches original: (1-alpha)*dot + alpha*wedge
        g = torch.sigmoid(self.fusion_gate)
        output = (1 - g) * dot_out + g * wedge_out

        attn_reg = (dot_scores.abs().mean() + wedge_scores.abs().mean()) / 2

        if self.output_attention:
            return output.contiguous()
        else:
            return output.contiguous(), attn_reg


class CombinedGeomAttention(nn.Module):
    """Original GeomAttention with combined dot+wedge scores.
    Takes 3 inputs: (Q, K, V). Same as original SimpleTM but without pywt dependency.
    """
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, alpha=1.0):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.alpha = alpha

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        queries_norm2 = torch.sum(queries ** 2, dim=-1).permute(0, 2, 1).unsqueeze(-1)
        keys_norm2 = torch.sum(keys ** 2, dim=-1).permute(0, 2, 1).unsqueeze(-2)
        wedge_norm2 = F.relu(queries_norm2 * keys_norm2 - dot_product ** 2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)

        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale

        if self.mask_flag:
            attn_mask = torch.tril(torch.ones(L, S, device=scores.device))
            scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous()
        else:
            return V.contiguous(), scores.abs().mean()


# ============================================================
# Flexible Attention Layer (supports all ablation variants)
# ============================================================

class FourierAttentionLayer(nn.Module):
    """Configurable attention layer supporting ablation experiments.

    Args:
        tokenization: 'fft' or 'swt'
        attention_mode: 'dual_path' or 'combined'
    """
    def __init__(self, attention, d_model, d_channel=None, m=2,
                 tokenization='fft', attention_mode='dual_path',
                 geomattn_dropout=0.5,
                 # SWT-specific params (only used when tokenization='swt')
                 requires_grad=True, wv='db2', kernel_size=None):
        super().__init__()

        self.d_channel = d_channel
        self.inner_attention = attention
        self.attention_mode = attention_mode

        # --- Tokenization ---
        if tokenization == 'fft':
            self.decompose = FourierEmbedding(d_channel=d_channel, m=m, decompose=True)
            self.reconstruct = FourierEmbedding(d_channel=d_channel, m=m, decompose=False)
        else:  # swt
            from layers.SWTAttention_Family import WaveletEmbedding
            self.decompose = WaveletEmbedding(
                d_channel=d_channel, swt=True,
                requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size
            )
            self.reconstruct = WaveletEmbedding(
                d_channel=d_channel, swt=False,
                requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size
            )

        # --- Projections ---
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
        )

        if attention_mode == 'dual_path':
            # Two separate V projections
            self.value_dot_projection = nn.Sequential(
                nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
            )
            self.value_wedge_projection = nn.Sequential(
                nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
            )
        else:
            # Single V projection (same as original)
            self.value_projection = nn.Sequential(
                nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
            )

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # Decompose: [B, C, d_model] -> [B, C, m+1, d_model]
        queries = self.decompose(queries)
        keys = self.decompose(keys)
        values = self.decompose(values)

        # Project
        Q = self.query_projection(queries).permute(0, 3, 2, 1)
        K = self.key_projection(keys).permute(0, 3, 2, 1)

        if self.attention_mode == 'dual_path':
            V1 = self.value_dot_projection(values).permute(0, 3, 2, 1)
            V2 = self.value_wedge_projection(values).permute(0, 3, 2, 1)
            out, attn = self.inner_attention(Q, K, V1, V2)
        else:
            V = self.value_projection(values).permute(0, 3, 2, 1)
            out, attn = self.inner_attention(Q, K, V)

        # Reconstruct: [B, d_model, m+1, C] -> [B, C, d_model]
        out = self.out_linear(out.permute(0, 3, 2, 1))
        out = self.reconstruct(out)

        return out, attn
