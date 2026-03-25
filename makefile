VENV_PY_WIN := .venv/Scripts/python.exe
VENV_PY_UNIX := .venv/bin/python

ifeq ($(wildcard $(VENV_PY_WIN)),$(VENV_PY_WIN))
PY := $(VENV_PY_WIN)
else ifeq ($(wildcard $(VENV_PY_UNIX)),$(VENV_PY_UNIX))
PY := $(VENV_PY_UNIX)
else
PY := python
endif

AUDIO_CONFIG := src/configs/audio_params.yaml
CONFIG := $(AUDIO_CONFIG)
LABELS_CONFIG := src/configs/labels.yaml
TRAIN_CONFIG ?= src/configs/train_params.yaml
TRAIN_OUTPUT_DIR ?=
# FEATURE options: mel | cqt | mfcc | chroma
FEATURE ?= mel
WORKERS ?= 12
# DATASET options: irmas | chinese_instruments
DATASET ?= irmas
MIX_DATASET ?= chinese_instruments
MIX_FEATURE ?= mel
MIX_WORKERS ?= 8
ALL_DATASETS := irmas chinese_instruments
ALL_FEATURES := mel cqt mfcc chroma
ALL_EXTRACT_TARGETS := $(foreach ds,$(ALL_DATASETS),$(foreach feat,$(ALL_FEATURES),extract-$(ds)-$(feat)))
TRAIN_ARGS := --config $(TRAIN_CONFIG) --audio_config $(AUDIO_CONFIG) --labels_config $(LABELS_CONFIG) $(if $(strip $(TRAIN_OUTPUT_DIR)),--output_dir $(TRAIN_OUTPUT_DIR),)

.PHONY: extract extract-help train train-dry train-resume train-help irmas chinese mfcc chroma all mix clean $(ALL_EXTRACT_TARGETS)


extract:
	"$(PY)" -m src.scripts.extract_features --config $(AUDIO_CONFIG) --dataset $(DATASET) --feature $(FEATURE) --num_workers $(WORKERS) --labels_config $(LABELS_CONFIG)

train:
	"$(PY)" -m src.train.run_train $(TRAIN_ARGS)

train-dry:
	"$(PY)" -m src.train.run_train $(TRAIN_ARGS) --dry_run

train-resume:
	"$(PY)" -m src.train.run_train $(TRAIN_ARGS) --resume

train-help:
	"$(PY)" -m src.train.run_train --help

irmas:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

chinese: DATASET := chinese_instruments
chinese:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

mfcc: FEATURE := mfcc
mfcc:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

chroma: FEATURE := chroma
chroma:
	$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

extract-help:
	"$(PY)" -m src.scripts.extract_features --help

define EXTRACT_template
extract-$(1)-$(2):
	@echo ">>> extracting dataset=$(1) feature=$(2)"
	$(MAKE) extract DATASET=$(1) FEATURE=$(2) WORKERS=$(WORKERS)
endef

$(foreach ds,$(ALL_DATASETS),$(foreach feat,$(ALL_FEATURES),$(eval $(call EXTRACT_template,$(ds),$(feat)))))

# generate all feature combinations for all datasets
all: $(ALL_EXTRACT_TARGETS)

clean:
	"$(PY)" -c "from pathlib import Path; import shutil; shutil.rmtree(Path('data/processed'), ignore_errors=True)"

# mix:
# 	$(PY) src/scripts/mix_and_extract.py --config $(CONFIG) --dataset $(MIX_DATASET) --feature $(MIX_FEATURE) --num_workers $(MIX_WORKERS) --labels_config $(LABELS_CONFIG) $(if $(MIX_NUM_MIXES),--num_mixes $(MIX_NUM_MIXES),) $(if $(MIX_SEED),--seed $(MIX_SEED),)
