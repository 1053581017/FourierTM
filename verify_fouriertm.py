"""FourierTM verification script.

Checks:
1. FourierEmbedding decomposition/reconstruction
2. ImprovedASB forward + gradient
3. DualPathGeomAttention forward + gradient
4. All 4 ablation variants + full model
5. Separate optimizer for ASB
"""
import torch
import argparse
import sys


def create_configs(**overrides):
    """Create minimal config for testing."""
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    defaults = dict(
        seq_len=96, pred_len=96, output_attention=False, use_norm=True,
        alpha=0.5, d_model=32, d_ff=32, e_layers=1, m=3,
        enc_in=7, dec_in=7, c_out=7, embed='timeF', freq='h',
        dropout=0.1, geomattn_dropout=0.5, factor=1, activation='gelu',
        asb_n_bands=4, use_asb=1, tokenization='fft', attention_mode='dual_path',
        requires_grad=True, wv='db1', kernel_size=None,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def test_fourier_embedding():
    from layers.FourierAttention import FourierEmbedding
    B, C, D, m = 4, 7, 32, 3
    x = torch.randn(B, C, D)

    fft_dec = FourierEmbedding(d_channel=C, m=m, decompose=True)
    coeffs = fft_dec(x)
    assert coeffs.shape == (B, C, m + 1, D)

    fft_rec = FourierEmbedding(d_channel=C, m=m, decompose=False)
    x_rec = fft_rec(coeffs)
    error = (x - x_rec).abs().max().item()
    assert error < 1e-5
    print(f"[PASS] FourierEmbedding: {x.shape} -> {coeffs.shape} -> {x_rec.shape}, error={error:.2e}")


def test_asb():
    from layers.ASB import ImprovedASB
    B, C, D = 4, 7, 32
    x = torch.randn(B, C, D)
    asb = ImprovedASB(d_model=D, d_channel=C, n_bands=4)
    out = asb(x)
    assert out.shape == x.shape
    out.sum().backward()
    for name, p in asb.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"No grad: {name}"
    print(f"[PASS] ImprovedASB: {x.shape} -> {out.shape}, all grads OK")


def test_dual_path_attention():
    from layers.FourierAttention import DualPathGeomAttention
    B, L, H, E = 4, 4, 1, 32
    Q = torch.randn(B, L, H, E, requires_grad=True)
    K = torch.randn(B, L, H, E)
    V1 = torch.randn(B, L, H, E)
    V2 = torch.randn(B, L, H, E)
    attn = DualPathGeomAttention(attention_dropout=0.0)
    out, reg = attn(Q, K, V1, V2)
    assert out.shape == (B, L, H, E)
    out.sum().backward()
    assert attn.fusion_gate.grad is not None
    print(f"[PASS] DualPathGeomAttention: {out.shape}, fusion_gate grad OK")


def test_combined_attention():
    from layers.FourierAttention import CombinedGeomAttention
    B, L, H, E = 4, 4, 1, 32
    Q = torch.randn(B, L, H, E)
    K = torch.randn(B, L, H, E)
    V = torch.randn(B, L, H, E)
    attn = CombinedGeomAttention(attention_dropout=0.0)
    out, reg = attn(Q, K, V)
    assert out.shape == (B, L, H, E)
    print(f"[PASS] CombinedGeomAttention: {out.shape}")


def test_ablation_variant(name, use_asb, tokenization, attention_mode):
    """Test a specific ablation configuration."""
    from model.FourierTM import Model

    configs = create_configs(
        use_asb=use_asb,
        tokenization=tokenization,
        attention_mode=attention_mode,
    )
    model = Model(configs)
    B = 2
    x = torch.randn(B, configs.seq_len, configs.enc_in)
    out, _ = model(x, None, None, None)
    assert out.shape == (B, configs.pred_len, configs.dec_in)

    out.sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"No gradient: {no_grad}"

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    has_asb = model.asb is not None
    print(f"[PASS] {name}: asb={has_asb}, tok={tokenization}, attn={attention_mode}, "
          f"params={total:,}, out={out.shape}")


def test_separate_optimizer():
    from model.FourierTM import Model
    from torch import optim

    configs = create_configs(use_asb=1)
    model = Model(configs)
    asb_ids = set(id(p) for p in model.asb.parameters())
    asb_params = list(model.asb.parameters())
    other_params = [p for p in model.parameters() if id(p) not in asb_ids]
    lr = 0.02
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': lr},
        {'params': asb_params, 'lr': lr * 0.1},
    ])
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]['lr'] == lr * 0.1
    print(f"[PASS] Separate optimizer: main lr={lr}, ASB lr={lr*0.1}")


if __name__ == '__main__':
    print("=" * 60)
    print("FourierTM Verification")
    print("=" * 60)

    tests = [
        ("FourierEmbedding", test_fourier_embedding),
        ("ImprovedASB", test_asb),
        ("DualPathGeomAttention", test_dual_path_attention),
        ("CombinedGeomAttention", test_combined_attention),
        # All 4 ablation variants + full
        ("FFT-only",       lambda: test_ablation_variant("FFT-only",       0, 'fft', 'combined')),
        ("ASB-only",       lambda: test_ablation_variant("ASB-only",       1, 'swt', 'combined')),
        ("DualPath-only",  lambda: test_ablation_variant("DualPath-only",  0, 'swt', 'dual_path')),
        ("FourierTM-Full", lambda: test_ablation_variant("FourierTM-Full", 1, 'fft', 'dual_path')),
        ("Separate Optimizer", test_separate_optimizer),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All verifications passed!")
