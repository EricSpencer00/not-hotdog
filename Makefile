PY := .venv/bin/python
SITE ?= ../EricSpencer00.github.io

.PHONY: help setup data teacher student export calibrate build test test-fast eval samples deploy clean

help:
	@echo "setup      create the venv and install dependencies"
	@echo "data       download and manifest every dataset (slow, ~6 GB)"
	@echo "teacher    fine-tune EfficientNet-B0   (gate: F1)"
	@echo "student    distil + QAT the 136K model"
	@echo "export     emit int8 weights for both engines"
	@echo "calibrate  pick the decision threshold on val, fold it into the bias"
	@echo "build      bundle dist/ (index.html + hotdog.js)"
	@echo "test       kernel + fixed-point + full parity on the val set"
	@echo "test-fast  same, but parity on 300 synthetic images (what CI runs)"
	@echo "eval       accuracy of the int8 model on val, test and the hard set"
	@echo "deploy     rsync dist/ into \$$SITE/hotdog/ (default $(SITE))"

setup:
	uv venv --python 3.11 && uv sync

data:
	$(PY) data/ingest_food101.py
	$(PY) data/ingest_imagenette.py
	$(PY) data/ingest_openimages.py
	$(PY) data/ingest_oi_labels.py --mode both
	$(PY) data/make_samples.py

teacher:
	$(PY) train/teacher.py

student:
	$(PY) train/distill.py

export:
	$(PY) train/export.py

calibrate:
	$(PY) train/calibrate.py
	$(PY) train/export.py

build: export
	$(PY) build.py

samples:
	$(PY) data/make_samples.py

# Full local run: parity over every held-out validation image.
test:
	$(PY) reference/gen_fixedpoint_vectors.py
	$(PY) reference/gen_kernel_vectors.py
	$(PY) reference/gen_parity_vectors.py --source val --n 0
	node --test engine/test/*.test.js

# What CI runs: no dataset needed, parity on synthetic images.
test-fast:
	$(PY) reference/gen_fixedpoint_vectors.py
	$(PY) reference/gen_kernel_vectors.py
	$(PY) reference/gen_parity_vectors.py --source random --n 300
	node --test engine/test/*.test.js

eval:
	$(PY) train/eval.py

deploy: build
	@test -d "$(SITE)" || { echo "site repo not found at $(SITE)"; exit 1; }
	@cd "$(SITE)" && git rev-parse --abbrev-ref HEAD | grep -qx dev || \
		{ echo "refusing to deploy: $(SITE) is not on the dev branch"; exit 1; }
	mkdir -p "$(SITE)/hotdog"
	rsync -a --delete dist/ "$(SITE)/hotdog/"
	@echo "copied dist/ -> $(SITE)/hotdog/  (review on dev, then merge to main)"

clean:
	rm -rf dist/hotdog.js dist/index.html dist/samples engine/test/vectors
