# GeoVision AI

Computer vision + geospatial AI that compares satellite images from two time periods and detects
**urban expansion, deforestation, flooding and other land-use change**, then explains what it found.

- **No local copies of satellite data.** Sentinel-2 L2A is searched through public STAC catalogs and read as
  Cloud-Optimized GeoTIFFs over HTTP range requests, straight into memory. Only the pixels for your area,
  bands and dates are fetched (Element84 Earth Search on AWS by default, no account; Microsoft Planetary Computer optional).
- **Deep learning change detection.** A Siamese U-Net (shared encoder, fused feature pyramid) segments each pixel into
  `No change / Urban expansion / Deforestation / Flooding / Land-use change`.
- **Explainable.** Grad-CAM shows where the network looked, and per-class spectral evidence (ΔNDVI, ΔNDWI, ΔNDBI, ΔNBR),
  hectares, confidence and a plain-language narrative say why a pixel was flagged.
- **Rule-based baseline** from index differencing, usable without any training (also the weak-label source for real data).

## Project layout

```
geovision/
  data/stac_loader.py   STAC search + lazy COG cube, cloud mask (SCL), reflectance, indices, median composite
  data/synthetic.py     offline synthetic before/after scenes with exact labels
  data/dataset.py       in-memory random-crop tile dataset
  models/siamese_unet.py  model + save/load
  labels.py             rule-based change labels
  train.py  infer.py  explain.py  viz.py
  pipeline.py           fetch_pair() / run_analysis() / export_geotiff()
  cli.py                python -m geovision {demo,train,analyze,presets}
app.py                  Streamlit app
tests/                  offline pytest suite
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
(For a smaller CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.)

## Run

```bash
# 1. Verify everything works offline (synthetic data, ~3 min CPU to train a small model)
python -m pytest -q
python -m geovision demo              # -> outputs/demo_report.png, outputs/demo_insights.json

# 2. Web app (works offline in "Offline synthetic demo" mode)
streamlit run app.py

# 3. Stream real Sentinel-2 (needs internet)
python -m geovision presets
python -m geovision analyze --aoi urban_bengaluru --method rules
python -m geovision analyze --aoi deforest_rondonia --before 2019-06-15/2019-09-15 --after 2025-06-15/2025-09-15
python -m geovision analyze --bbox 77.62 12.86 77.74 12.97 --before 2019-01-01/2019-03-31 --after 2026-01-01/2026-03-31

# 4. Train the model
python -m geovision train --source synthetic               # offline, exact labels
python -m geovision train --source both --steps 1500       # + streamed real scenes with rule-based weak labels
python -m geovision analyze --aoi flood_assam --method model --ckpt checkpoints/geovision.pt
```
`analyze` writes `outputs/<aoi>_report.png`, `<aoi>_insights.json` and a georeferenced `<aoi>_change.tif`.
`make test | demo | app | train | analyze` are shortcuts for the same commands.

## Use as a library

```python
from geovision.pipeline import fetch_pair, run_analysis
from geovision.models import load_model

pair = fetch_pair((77.62, 12.86, 77.74, 12.97), "2019-01-01/2019-03-31", "2026-01-01/2026-03-31", res=20)
result = run_analysis(pair, "model", load_model("checkpoints/geovision.pt"))
print(result.insights["headline"])
```

## Honest limitations

- The synthetic demo scores ~1.0 IoU because its classes are cleanly separable by construction. It proves the pipeline works;
  it is **not** a real-world accuracy figure.
- A model trained only on synthetic data will not transfer well to real imagery. For real use, train with `--source both`
  (weak labels from index rules, so the model largely learns to imitate and denoise them) and ideally fine-tune on
  human-labelled change masks (e.g. OSCD, LEVIR-CD, or your own polygons) by adding them to `train.py` as extra `(before, after, label)` items.
- Compare the same season on both dates; phenology looks like change. Flooding presets deliberately compare dry vs monsoon.
- Grad-CAM explains the dominant change class per tile at coarse (1/4) resolution; treat it as a guide, not ground truth.
- Sentinel-2 offset handling uses a fixed 2022-01-25 processing-baseline cutoff.
- `analyze` and the app's streaming mode were not exercised against live catalogs while building this (the build sandbox had
  no access to them); everything else is covered by the offline tests and demo.
