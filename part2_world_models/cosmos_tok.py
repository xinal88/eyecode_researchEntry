"""Thin wrapper around the official NVIDIA Cosmos Tokenizer TorchScript checkpoints.

Cosmos-0.1-Tokenizer-CV4x8x8 is a *continuous, causal* video tokenizer:
  video [B, 3, 1 + 4k, H, W] in [-1, 1]  ->  latent [B, 16, 1 + k, H/8, W/8]
Because it is causal, the first latent frame depends only on the first image, which is
what makes image-conditioned (Video2World-style) generation in latent space possible.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "checkpoints/cosmos_tokenizer_cv4x8x8"


class CosmosTokenizer:
    def __init__(self, ckpt_dir=DEFAULT_DIR, device="cuda", dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.enc = torch.jit.load(str(Path(ckpt_dir) / "encoder.jit"), map_location=device).eval()
        self.dec = torch.jit.load(str(Path(ckpt_dir) / "decoder.jit"), map_location=device).eval()
        for m in (self.enc, self.dec):
            m.to(dtype)

    @staticmethod
    def to_tensor(frames_uint8, size=None, device="cuda"):
        """(T, H, W, 3) uint8 -> (1, 3, T, h, w) float in [-1, 1], optionally resized (on GPU)."""
        x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(device).float().permute(3, 0, 1, 2)[None] / 127.5 - 1
        if size is not None and size != x.shape[-1]:
            t = x.shape[2]
            x = x[0].permute(1, 0, 2, 3)  # T,3,H,W
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(-1, 1)
            x = x.permute(1, 0, 2, 3)[None]
            assert x.shape[2] == t
        return x

    @staticmethod
    def to_uint8(x, size=None):
        """(B, 3, T, h, w) in [-1, 1] -> (B, T, H, W, 3) uint8, optionally resized back."""
        x = x.float()
        if size is not None and size != x.shape[-1]:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.interpolate(x, size=(size, size), mode="area")
            x = x.reshape(b, t, c, size, size).permute(0, 2, 1, 3, 4)
        x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte()
        return x.permute(0, 2, 3, 4, 1).cpu().numpy()

    @torch.no_grad()
    def encode(self, x):
        out = self.enc(x.to(self.device, self.dtype))
        return out[0] if isinstance(out, (tuple, list)) else out

    @torch.no_grad()
    def decode(self, z):
        return self.dec(z.to(self.device, self.dtype))
