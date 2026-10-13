"""Siamese U-Net for multi-class bi-temporal change segmentation.

A shared-weight encoder embeds the before and after images. At every scale the two feature maps
are fused as [f_before, f_after, |f_before - f_after|] -> 1x1 conv, and a U-Net decoder turns the
fused pyramid into per-pixel change-class logits. Input H and W must be divisible by 8.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import N_CLASSES


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class _Fuse(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(3 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

    def forward(self, fa, fb):
        return self.proj(torch.cat([fa, fb, (fa - fb).abs()], dim=1))


class SiameseUNet(nn.Module):
    def __init__(self, in_ch: int = 6, n_classes: int = N_CLASSES, base: int = 16):
        super().__init__()
        b = base
        self.enc1, self.enc2 = _block(in_ch, b), _block(b, 2 * b)
        self.enc3, self.enc4 = _block(2 * b, 4 * b), _block(4 * b, 8 * b)
        self.fuse1, self.fuse2, self.fuse3, self.fuse4 = _Fuse(b), _Fuse(2 * b), _Fuse(4 * b), _Fuse(8 * b)
        self.up3, self.dec3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2), _block(8 * b, 4 * b)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), _block(4 * b, 2 * b)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), _block(2 * b, b)
        self.head = nn.Conv2d(b, n_classes, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        return e1, e2, e3, e4

    def forward(self, before, after):
        ea, eb = self.encode(before), self.encode(after)
        f1, f2, f3, f4 = (fuse(x, y) for fuse, x, y in zip((self.fuse1, self.fuse2, self.fuse3, self.fuse4), ea, eb))
        d3 = self.dec3(torch.cat([self.up3(f4), f3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), f2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), f1], 1))
        return self.head(d1)


def build_model(in_ch: int = 6, n_classes: int = N_CLASSES, base: int = 16) -> SiameseUNet:
    return SiameseUNet(in_ch, n_classes, base)


def save_model(model: SiameseUNet, path, meta: dict | None = None) -> None:
    cfg = {"in_ch": model.enc1[0].in_channels, "n_classes": model.head.out_channels,
           "base": model.enc1[0].out_channels}
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "meta": meta or {}}, path)


def load_model(path, device: str = "cpu") -> SiameseUNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(**ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
