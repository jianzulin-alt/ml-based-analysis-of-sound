PY := .venv/bin/python
CONFIG := src/configs/audio_params.yaml
LABELS_CONFIG := src/configs/labels.yaml
# FEATURE options: mel | cqt | mel_cqt 
FEATURE ?= mel
WORKERS ?= 12
# DATASET options: irmas | chinese_instruments
DATASET ?= irmas
MIX_DATASET ?= chinese_instruments
MIX_FEATURE ?= mel
MIX_WORKERS ?= 8

.PHONY: extract extract-help irmas chinese mix clean


extract:
	$(PY) src/scripts/extract_features.py --config $(CONFIG) --dataset $(DATASET) --feature $(FEATURE) --num_workers $(WORKERS) --labels_config $(LABELS_CONFIG)

irmas:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

chinese: DATASET := chinese_instruments
chinese:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

extract-help:
	$(PY) src/scripts/extract_features.py --help

mix:
	$(PY) src/scripts/mix_and_extract.py --config $(CONFIG) --dataset $(MIX_DATASET) --feature $(MIX_FEATURE) --num_workers $(MIX_WORKERS) --labels_config $(LABELS_CONFIG) $(if $(MIX_NUM_MIXES),--num_mixes $(MIX_NUM_MIXES),) $(if $(MIX_SEED),--seed $(MIX_SEED),)

clean:
	rm -rf data/processed
