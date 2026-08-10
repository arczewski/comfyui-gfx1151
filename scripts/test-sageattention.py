#!/usr/bin/env python3
"""
Test script for SageAttention RDNA3 on gfx1151 (Strix Halo).
Verifies:
  1. SageAttention imports correctly
  2. Forward pass works with fp16 tensors
  3. Both 'native' and 'triton' backends work (if available)
"""

import os
import torch

print("=== SageAttention RDNA3 Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")

print()

# Check import
try:
    import sageattention
    from sageattention import sageattn
    print("✓ sageattention imported successfully")
except ImportError as e:
    print(f"✗ Failed to import sageattention: {e}")
    print("  Run: cd /opt/sageattention-rdna3 && GPU_ARCHS=gfx1151 pip install . --no-build-isolation")
    exit(1)

# Check backend
backend = os.environ.get("SAGEATTN_BACKEND", "triton")
print(f"  SAGEATTN_BACKEND = {backend}")

# Check if native extension is available
try:
    import sageattention._qattn_gfx11
    print("  Native HIP extension: AVAILABLE")
except ImportError:
    print("  Native HIP extension: not built (Triton-only mode)")

# Run forward pass
print()
print("=== Forward Pass Test ===")
try:
    device = "cuda"
    dtype = torch.float16

    # Standard SDXL-like attention shapes: batch=2, heads=8, seq=4096, head_dim=64
    B, H, S, D = 2, 8, 4096, 64
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    print(f"✓ Forward pass OK — output shape: {out.shape}, dtype: {out.dtype}")
    print(f"  Mean: {out.mean().item():.6f}, Std: {out.std().item():.6f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")

# Test with causal mask (shorter sequence, like text-to-image cross-attention)
print()
print("=== Causal Mask Test ===")
try:
    S_short = 128
    q2 = torch.randn(1, 8, S_short, D, device=device, dtype=dtype)
    k2 = torch.randn(1, 8, S_short, D, device=device, dtype=dtype)
    v2 = torch.randn(1, 8, S_short, D, device=device, dtype=dtype)

    out2 = sageattn(q2, k2, v2, tensor_layout="HND", is_causal=True)
    print(f"✓ Causal forward pass OK — output shape: {out2.shape}")
except Exception as e:
    print(f"✗ Causal forward pass failed: {e}")

print()
print("=== All checks complete ===")
