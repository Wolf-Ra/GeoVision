"""Training. Sources: synthetic (offline, exact labels) or stac (streamed scenes + rule-based weak labels)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import N_CLASSES, PRESET_WINDOWS
from .data.dataset import ArrayPairDataset
from .data.synthetic import synthetic_scene
from .infer import predict_full
from .labels import rule_based_labels
from .models import build_model, save_model


def dice_loss(logits, y, n_classes: int = N_CLASSES):
    p = torch.softmax(logits, 1)
    onehot = F.one_hot(y, n_classes).permute(0, 3, 1, 2).float()
    inter, denom = (p * onehot).sum((0, 2, 3)), (p + onehot).sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    dice = (2 * inter + 1) / (denom + 1)
    return 1 - dice[1:][present[1:]].mean() if present[1:].any() else logits.sum() * 0


def iou_per_class(pred: np.ndarray, truth: np.ndarray, n_classes: int = N_CLASSES) -> list[float]:
    out = []
    for c in range(n_classes):
        inter = np.logical_and(pred == c, truth == c).sum()
        union = np.logical_or(pred == c, truth == c).sum()
        out.append(float(inter / union) if union else float("nan"))
    return out


def mean_iou_changed(pred, truth) -> float:
    return float(np.nanmean(iou_per_class(pred, truth)[1:]))


def synthetic_items(n: int, size: int, seed0: int):
    return [synthetic_scene(size, seed0 + i) for i in range(n)]


def stac_items(res: int = 20, presets=None, log=print):
    """Stream the preset AOIs and label them with the rule-based weak labeller."""
    from .data.stac_loader import PRESETS
    from .pipeline import fetch_pair

    items = []
    for key in presets or PRESET_WINDOWS:
        before, after, cloud = PRESET_WINDOWS[key]
        try:
            log(f"[stream] {key} ...")
            pair = fetch_pair(PRESETS[key].bbox, before, after, res=res, max_cloud=cloud, name=key)
            items.append((pair.before, pair.after, rule_based_labels(pair.before, pair.after)))
        except Exception as e:  # keep going if one AOI fails
            log(f"[skip] {key}: {type(e).__name__}: {str(e)[:100]}")
    if not items:
        raise RuntimeError("Could not stream any scene - check your network / date windows.")
    return items


def train(source: str = "synthetic", steps: int = 600, batch: int = 16, lr: float = 2e-3, tile: int = 64,
          base: int = 16, out: str = "checkpoints/geovision.pt", seed: int = 0, res: int = 20,
          device: str | None = None, log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_items, val_items = [], []
    if source in ("synthetic", "both"):
        train_items += synthetic_items(24, 192, seed0=1000)
        val_items += synthetic_items(4, 192, seed0=9000)
    if source in ("stac", "both"):
        real = stac_items(res, log=log)
        train_items += real
        val_items += real[:1]

    loader = DataLoader(ArrayPairDataset(train_items, tile=tile, length=steps * batch), batch_size=batch,
                        shuffle=False, num_workers=0)
    model = build_model(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    weights = torch.tensor([1.0] + [4.0] * (N_CLASSES - 1), device=device)

    model.train()
    t0, history = time.time(), []
    for step, (a, b, y) in enumerate(loader, 1):
        a, b, y = a.to(device), b.to(device), y.to(device)
        logits = model(a, b)
        loss = F.cross_entropy(logits, y, weight=weights) + dice_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if step % max(steps // 10, 1) == 0 or step == steps:
            log(f"step {step:4d}/{steps}  loss {np.mean(history[-20:]):.4f}  ({time.time() - t0:.0f}s)")

    model.eval()
    scores = []
    for a, b, y in val_items:
        pred = predict_full(model, a, b, tile=min(256, (min(y.shape) // 8) * 8), overlap=16, device=device).argmax(0)
        scores.append(mean_iou_changed(pred, y))
    miou = float(np.nanmean(scores))
    log(f"validation mIoU (change classes): {miou:.3f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, out, meta={"source": source, "steps": steps, "val_miou": miou})
    log(f"saved {out}")
    return {"loss": history, "val_miou": miou, "checkpoint": out}
