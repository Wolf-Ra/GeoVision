PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

demo:            # offline end-to-end run on synthetic data
	$(PY) -m geovision demo

train:           # synthetic labels; use SOURCE=both to add streamed real scenes
	$(PY) -m geovision train --source $(or $(SOURCE),synthetic)

analyze:         # stream real Sentinel-2 (needs internet)
	$(PY) -m geovision analyze --aoi $(or $(AOI),urban_bengaluru)

app:
	streamlit run app.py

.PHONY: setup test demo train analyze app
