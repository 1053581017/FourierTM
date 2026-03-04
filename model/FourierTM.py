import torch
import torch.nn as nn
from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.FourierAttention import (
    FourierAttentionLayer,
    DualPathGeomAttention,
    CombinedGeomAttention,
)
from layers.ASB import ImprovedASB
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """FourierTM: configurable model supporting all ablation variants.

    Controlled by three config switches:
        --use_asb:          0 or 1 (default 1)
        --tokenization:     'fft' or 'swt' (default 'fft')
        --attention_mode:   'dual_path' or 'combined' (default 'dual_path')

    Ablation variants:
        +FFT only:       use_asb=0, tokenization=fft, attention_mode=combined
        +ASB only:       use_asb=1, tokenization=swt, attention_mode=combined
        +Dual-Path only: use_asb=0, tokenization=swt, attention_mode=dual_path
        FourierTM Full:  use_asb=1, tokenization=fft, attention_mode=dual_path
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.alpha = configs.alpha

        # Read ablation switches (with safe defaults)
        self.use_asb = bool(getattr(configs, 'use_asb', 1))
        tokenization = getattr(configs, 'tokenization', 'fft')
        attention_mode = getattr(configs, 'attention_mode', 'dual_path')

        # 1. Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # 2. ASB (optional)
        if self.use_asb:
            self.asb = ImprovedASB(
                d_model=configs.d_model,
                d_channel=configs.dec_in,
                n_bands=getattr(configs, 'asb_n_bands', 4)
            )
        else:
            self.asb = None

        # 3. Build inner attention based on mode
        def _build_inner_attention():
            if attention_mode == 'dual_path':
                return DualPathGeomAttention(
                    False, configs.factor,
                    attention_dropout=configs.dropout,
                    output_attention=configs.output_attention,
                    alpha=self.alpha
                )
            else:
                return CombinedGeomAttention(
                    False, configs.factor,
                    attention_dropout=configs.dropout,
                    output_attention=configs.output_attention,
                    alpha=self.alpha
                )

        # 4. Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FourierAttentionLayer(
                        _build_inner_attention(),
                        configs.d_model,
                        d_channel=configs.dec_in,
                        m=configs.m,
                        tokenization=tokenization,
                        attention_mode=attention_mode,
                        geomattn_dropout=getattr(configs, 'geomattn_dropout', 0.5),
                        # SWT params (used only when tokenization='swt')
                        requires_grad=getattr(configs, 'requires_grad', True),
                        wv=getattr(configs, 'wv', 'db1'),
                        kernel_size=getattr(configs, 'kernel_size', None),
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 5. Output Projection
        self.projector = nn.Linear(configs.d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        if self.asb is not None:
            enc_out = self.asb(enc_out)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, None, None, None)
        return dec_out, attns
