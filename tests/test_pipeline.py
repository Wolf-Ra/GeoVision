"""Offline tests - no network needed (synthetic scenes + fake xarray cubes)."""
import json

import numpy as np
import pytest
import torch
import xarray as xr

from geovision.data.synthetic import synthetic_scene
from geovision.explain import explain_map, summarize
from geovision.infer import predict_full
from geovision.labels import rule_based_labels
from geovision.models import build_model, load_model, save_model
from geovision.pipeline import run_analysis, synthetic_pair
from geovision.train import mean_iou_changed, train


def test_synthetic_scene_shapes_and_labels():
    a, b, y = synthetic_scene(128, seed=1)
    assert a.shape == b.shape == (6, 128, 128) and y.shape == (128, 128)
    assert set(np.unique(y)) <= {0, 1, 2, 3, 4}
    assert 0.03 < (y > 0).mean() < 0.6
    assert a.min() >= 0 and a.max() <= 1


def test_rule_labels_beat_chance():
    scores = []
    for s in range(3):
        a, b, y = synthetic_scene(192, seed=s)
        scores.append(mean_iou_changed(rule_based_labels(a, b), y))
    assert np.nanmean(scores) > 0.4


def test_model_forward_shape():
    m = build_model(base=8)
    out = m(torch.randn(2, 6, 64, 64), torch.randn(2, 6, 64, 64))
    assert out.shape == (2, 5, 64, 64)


def test_train_save_load_predict_explain(tmp_path):
    ckpt = tmp_path / "m.pt"
    info = train("synthetic", steps=12, batch=4, base=8, out=str(ckpt), log=lambda *_: None)
    assert np.isfinite(info["loss"]).all()
    model = load_model(ckpt)
    a, b, _ = synthetic_scene(100, seed=7)  # odd size, smaller than tile
    probs = predict_full(model, a, b, tile=64, overlap=16)
    assert probs.shape == (5, 100, 100)
    assert np.allclose(probs.sum(0), 1, atol=1e-4)
    cm = probs.argmax(0)
    cam = explain_map(model, a, b, cm, tile=64, overlap=16, min_pixels=1)
    assert cam.shape == (100, 100) and np.isfinite(cam).all()


def test_run_analysis_both_methods_and_json(tmp_path):
    pair = synthetic_pair(128, seed=5)
    rules = run_analysis(pair, "rules")
    json.dumps(rules.insights)
    m = build_model(base=8)
    save_model(m, tmp_path / "x.pt")
    res = run_analysis(pair, "model", load_model(tmp_path / "x.pt"), tile=64, overlap=16)
    assert res.change_map.shape == (128, 128) and res.cam is not None
    json.dumps(res.insights)


def test_summarize_hectares():
    a, b, y = synthetic_scene(64, seed=2)
    ins = summarize(y, a, b, res_m=10.0)
    total_ha = sum(e["hectares"] for e in ins["classes"])
    assert total_ha == pytest.approx((y > 0).sum() * 100 / 10_000, abs=0.05)


def test_stac_preprocessing_helpers():
    from geovision.data.stac_loader import add_indices, mask_clouds, to_reflectance

    times = np.array(["2021-06-01", "2023-06-01"], dtype="datetime64[ns]")
    shape = (2, 4, 4)
    dn = np.full(shape, 3000, dtype="uint16")
    ds = xr.Dataset({b: (("time", "y", "x"), dn.copy()) for b in
                     ["blue", "green", "red", "nir", "swir16", "swir22"]}, coords={"time": times})
    ds["scl"] = (("time", "y", "x"), np.full(shape, 4, dtype="uint8"))
    ds["scl"][1, 0, 0] = 9  # cloud
    refl = to_reflectance(ds)
    assert float(refl["red"][0, 0, 0]) == pytest.approx(0.30)   # pre-2022: no offset
    assert float(refl["red"][1, 1, 1]) == pytest.approx(0.20)   # post-2022: -1000 offset
    masked = mask_clouds(refl)
    assert np.isnan(masked["red"][1, 0, 0]) and not np.isnan(masked["red"][0, 0, 0])
    assert "ndvi" in add_indices(masked.median("time"))
