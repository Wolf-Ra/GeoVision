"""Command line: python -m geovision {demo,train,analyze,presets}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .config import PRESET_WINDOWS
from .models import load_model
from .pipeline import export_geotiff, fetch_pair, run_analysis, synthetic_pair
from .train import mean_iou_changed, train
from .viz import make_report

DEFAULT_CKPT = "checkpoints/geovision.pt"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_presets(_):
    from .data.stac_loader import PRESETS

    for k, a in PRESETS.items():
        b, af, cl = PRESET_WINDOWS[k]
        print(f"{k:20s} {a.description:28s} bbox={a.bbox}  before={b}  after={af}  cloud<{cl}%")


def cmd_train(a):
    train(a.source, a.steps, a.batch, a.lr, a.tile, a.base, a.ckpt, a.seed, a.res)


def cmd_demo(a):
    """Fully offline end-to-end run on synthetic data (trains a small model if needed)."""
    ckpt = Path(a.ckpt)
    if a.retrain or not ckpt.exists():
        print("== Training a small model on synthetic scenes ==")
        train("synthetic", a.steps, out=str(ckpt))
    model = load_model(ckpt, _device())
    pair = synthetic_pair(256, seed=a.seed)
    res_model = run_analysis(pair, "model", model, _device())
    res_rules = run_analysis(pair, "rules")
    print("\n== Mean IoU over change classes vs ground truth ==")
    print(f"  rule-based baseline : {mean_iou_changed(res_rules.change_map, pair.label):.3f}")
    print(f"  Siamese U-Net       : {mean_iou_changed(res_model.change_map, pair.label):.3f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    make_report(pair, res_model, str(out / "demo_report.png"))
    (out / "demo_insights.json").write_text(json.dumps(res_model.insights, indent=2))
    print("\n" + res_model.insights["headline"])
    for line in res_model.insights["narrative"]:
        print(" -", line)
    print(f"\nSaved {out / 'demo_report.png'} and {out / 'demo_insights.json'}")


def cmd_analyze(a):
    from .data.stac_loader import PRESETS

    if a.aoi:
        bbox = PRESETS[a.aoi].bbox
        before, after, cloud = PRESET_WINDOWS[a.aoi]
        before, after = a.before or before, a.after or after
        cloud = a.max_cloud if a.max_cloud is not None else cloud
        name = a.aoi
    else:
        if not (a.bbox and a.before and a.after):
            sys.exit("Provide --aoi PRESET, or --bbox W S E N with --before and --after")
        bbox, before, after, cloud, name = a.bbox, a.before, a.after, a.max_cloud or 30.0, "custom"

    method = a.method or ("model" if Path(a.ckpt).exists() else "rules")
    model = None
    if method == "model":
        if not Path(a.ckpt).exists():
            sys.exit(f"No checkpoint at {a.ckpt}. Run: python -m geovision train --source synthetic")
        model = load_model(a.ckpt, _device())
    print(f"Streaming Sentinel-2 for {name} {bbox}\n  before={before}  after={after}  res={a.res}m  method={method}")
    pair = fetch_pair(bbox, before, after, a.res, cloud, a.provider, name)
    print(f"  scenes used: {pair.meta['scenes_before']} before / {pair.meta['scenes_after']} after; "
          f"grid {pair.before.shape[1]}x{pair.before.shape[2]}")
    result = run_analysis(pair, method, model, _device(), explain=not a.no_explain)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    make_report(pair, result, str(out / f"{name}_report.png"))
    (out / f"{name}_insights.json").write_text(json.dumps(result.insights, indent=2))
    try:
        export_geotiff(str(out / f"{name}_change.tif"), result.change_map, pair)
    except Exception as e:
        print(f"  (GeoTIFF export skipped: {e})")
    print("\n" + result.insights["headline"])
    for line in result.insights["narrative"]:
        print(" -", line)
    print(f"\nOutputs in {out}/")


def main(argv=None):
    p = argparse.ArgumentParser(prog="geovision", description="GeoVision AI - satellite change detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("presets", help="list preset areas of interest"); sp.set_defaults(fn=cmd_presets)

    sp = sub.add_parser("train", help="train the Siamese U-Net")
    sp.add_argument("--source", choices=["synthetic", "stac", "both"], default="synthetic")
    sp.add_argument("--steps", type=int, default=600); sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--lr", type=float, default=2e-3); sp.add_argument("--tile", type=int, default=64)
    sp.add_argument("--base", type=int, default=16); sp.add_argument("--res", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0); sp.add_argument("--ckpt", default=DEFAULT_CKPT)
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("demo", help="offline end-to-end demo on synthetic data")
    sp.add_argument("--steps", type=int, default=500); sp.add_argument("--seed", type=int, default=123)
    sp.add_argument("--retrain", action="store_true"); sp.add_argument("--ckpt", default="checkpoints/demo.pt")
    sp.add_argument("--out", default="outputs"); sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("analyze", help="stream real Sentinel-2 data and detect change")
    sp.add_argument("--aoi", choices=list(PRESET_WINDOWS))
    sp.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    sp.add_argument("--before"); sp.add_argument("--after")
    sp.add_argument("--res", type=int, default=20); sp.add_argument("--max-cloud", type=float, dest="max_cloud")
    sp.add_argument("--provider", choices=["earth-search", "planetary"], default="earth-search")
    sp.add_argument("--method", choices=["rules", "model"]); sp.add_argument("--ckpt", default=DEFAULT_CKPT)
    sp.add_argument("--no-explain", action="store_true"); sp.add_argument("--out", default="outputs")
    sp.set_defaults(fn=cmd_analyze)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
