PY := .venv/bin/python
CONFIG := src/configs/audio_params.yaml
LABELS_CONFIG := src/configs/labels.yaml
# FEATURE options: mel | cqt 
FEATURE ?= mel
WORKERS ?= 12
# DATASET options: irmas | chinese_instruments
DATASET ?= irmas
MIX_DATASET ?= chinese_instruments
MIX_FEATURE ?= mel
MIX_WORKERS ?= 8
ALL_DATASETS := irmas chinese_instruments
ALL_FEATURES := mel cqt

.PHONY: extract extract-help irmas chinese all mix clean


extract:
	$(PY) src/scripts/extract_features.py --config $(CONFIG) --dataset $(DATASET) --feature $(FEATURE) --num_workers $(WORKERS) --labels_config $(LABELS_CONFIG)

irmas:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

chinese: DATASET := chinese_instruments
chinese:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

extract-help:
	$(PY) src/scripts/extract_features.py --help

# generate all features combinations for all datasets
all:
	@set -e; \
	for ds in $(ALL_DATASETS); do \
		for feat in $(ALL_FEATURES); do \
			echo ">>> extracting dataset=$$ds feature=$$feat"; \
			$(MAKE) extract DATASET=$$ds FEATURE=$$feat WORKERS=$(WORKERS); \
		done; \
	done

clean:
	rm -rf data/processed

# mix:
# 	$(PY) src/scripts/mix_and_extract.py --config $(CONFIG) --dataset $(MIX_DATASET) --feature $(MIX_FEATURE) --num_workers $(MIX_WORKERS) --labels_config $(LABELS_CONFIG) $(if $(MIX_NUM_MIXES),--num_mixes $(MIX_NUM_MIXES),) $(if $(MIX_SEED),--seed $(MIX_SEED),)
