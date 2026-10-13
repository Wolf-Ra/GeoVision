"""Explainability: Grad-CAM saliency from the fused Siamese features + index-based evidence/narrative."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_NAMES, N_CLASSES
from .indices import normalize, spectral_indices, valid_mask
from .infer import pad_min, tile_origins


def gradcam_tile(model, a: torch.Tensor, b: torch.Tensor, target_class: int,
                 mask: torch.Tensor | None = None, layer: str = "fuse3") -> torch.Tensor:
    """Grad-CAM of `target_class` (averaged over `mask` pixels) on a fusion layer -> [H, W] in 0..1."""
    store: dict = {}
    module = getattr(model, layer)

    def hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = module.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.zero_grad()
            score = model(a, b)[:, target_class]
            score = (score * mask).sum() / mask.sum().clamp(min=1) if mask is not None else score.mean()
            score.backward()
    finally:
        handle.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=a.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    return cam / cam.max().clamp(min=1e-8)


def explain_map(model, before, after, change_map, tile: int = 256, overlap: int = 32,
                device: str = "cpu", min_pixels: int = 20) -> np.ndarray:
    """Stitched Grad-CAM [H, W]: per tile, explains the dominant predicted change class."""
    model.eval().to(device)
    h, w = change_map.shape
    a, b = normalize(pad_min(before, tile)), normalize(pad_min(after, tile))
    cm = pad_min(change_map, tile)
    H, W = cm.shape
    acc, cnt = np.zeros((H, W), "float32"), np.zeros((H, W), "float32")
    for y, x in tile_origins(H, W, tile, overlap):
        sub = cm[y:y + tile, x:x + tile]
        counts = np.bincount(sub[sub > 0], minlength=N_CLASSES)
        if counts.sum() < min_pixels:
            continue
        c = int(counts.argmax())
        ta = torch.from_numpy(a[:, y:y + tile, x:x + tile])[None].to(device)
        tb = torch.from_numpy(b[:, y:y + tile, x:x + tile])[None].to(device)
        mask = torch.from_numpy((sub == c).astype("float32"))[None].to(device)
        acc[y:y + tile, x:x + tile] += gradcam_tile(model, ta, tb, c, mask).cpu().numpy()
        cnt[y:y + tile, x:x + tile] += 1
    return (acc / np.maximum(cnt, 1))[:h, :w]


_EVIDENCE = {
    1: "NDBI changed {ndbi:+.2f} and NDVI {ndvi:+.2f} on average: vegetation or farmland was replaced by built-up surfaces.",
    2: "NDVI fell {ndvi:+.2f} and NBR {nbr:+.2f}: dense vegetation was cleared to bare or low cover.",
    3: "NDWI rose {ndwi:+.2f} (NDVI {ndvi:+.2f}): land that was dry is now covered by open water.",
    4: "NDVI shifted {ndvi:+.2f} without built-up or water signatures: a change of cover such as cropping, regrowth or conversion to bare soil.",
}


def summarize(change_map, before, after, res_m: float, probs: np.ndarray | None = None) -> dict:
    """Per-class area, spectral evidence, model confidence and a plain-language narrative."""
    ib, ia = spectral_indices(before), spectral_indices(after)
    valid = valid_mask(before) & valid_mask(after)
    total = max(int(valid.sum()), 1)
    ha_per_px = res_m * res_m / 10_000
    classes, lines = [], []
    for c in range(1, N_CLASSES):
        m = (change_map == c) & valid
        n = int(m.sum())
        delta = {k: (float((ia[k][m] - ib[k][m]).mean()) if n else 0.0) for k in ("ndvi", "ndwi", "ndbi", "nbr")}
        entry = {
            "class_id": c, "name": CLASS_NAMES[c], "pixels": n,
            "hectares": round(n * ha_per_px, 2), "pct_of_aoi": round(100 * n / total, 2),
            "mean_delta": {k: round(v, 3) for k, v in delta.items()},
            "confidence": round(float(probs[c][m].mean()), 3) if (probs is not None and n) else None,
        }
        classes.append(entry)
        if n:
            conf = f" (mean confidence {entry['confidence']:.0%})" if entry["confidence"] is not None else ""
            lines.append(f"{CLASS_NAMES[c]}: {entry['hectares']:.1f} ha, {entry['pct_of_aoi']:.1f}% of the area{conf}. "
                         + _EVIDENCE[c].format(**delta))
    changed = sum(e["pixels"] for e in classes)
    top = max(classes, key=lambda e: e["pixels"])
    headline = (f"{100 * changed / total:.1f}% of the analysed area changed; dominant change: {top['name']}."
                if changed else "No significant change detected.")
    return {"headline": headline, "valid_pixels": total, "resolution_m": res_m, "classes": classes, "narrative": lines}
