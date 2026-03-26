#
# Python executable
#
VENV_PY_WIN := .venv/Scripts/python.exe
VENV_PY_UNIX := .venv/bin/python

ifeq ($(wildcard $(VENV_PY_WIN)),$(VENV_PY_WIN))
PY := $(VENV_PY_WIN)
else ifeq ($(wildcard $(VENV_PY_UNIX)),$(VENV_PY_UNIX))
PY := $(VENV_PY_UNIX)
else
PY := python
endif

#
# Python modules
#
EXTRACT_MODULE := src.scripts.extract_features
TRAIN_MODULE := src.scripts.run_train

#
# Config
#
AUDIO_CONFIG := src/configs/audio_params.yaml
LABELS_CONFIG := src/configs/labels.yaml
TRAIN_CONFIG ?= src/configs/train_params.yaml
TRAIN_OUTPUT_DIR ?=

#
# Extraction options
#
DATASET ?= irmas
FEATURE ?= mel
WORKERS ?= 12

ALL_DATASETS := irmas chinese_instruments
ALL_FEATURES := mel cqt mfcc chroma
ALL_EXTRACT_TARGETS := $(foreach ds,$(ALL_DATASETS),$(foreach feat,$(ALL_FEATURES),extract-$(ds)-$(feat)))

EXTRACT_ARGS := --config $(AUDIO_CONFIG) --dataset $(DATASET) --feature $(FEATURE) --num_workers $(WORKERS) --labels_config $(LABELS_CONFIG)
TRAIN_ARGS := --config $(TRAIN_CONFIG) --audio_config $(AUDIO_CONFIG) --labels_config $(LABELS_CONFIG) $(if $(strip $(TRAIN_OUTPUT_DIR)),--output_dir $(TRAIN_OUTPUT_DIR),)

.PHONY: extract extract-help help train train-dry train-resume train-help \
	irmas chinese mel cqt mfcc chroma all clean $(ALL_EXTRACT_TARGETS)


extract:
	"$(PY)" -m $(EXTRACT_MODULE) $(EXTRACT_ARGS)

help:
	@printf '%s\n' \
		'Common targets:' \
		'  make extract DATASET=irmas FEATURE=mel WORKERS=12' \
		'  make train TRAIN_CONFIG=src/configs/train_params.yaml' \
		'  make train-dry' \
		'  make train-resume TRAIN_OUTPUT_DIR=src/models/saved_weights/<run>' \
		'  make all' \
		'' \
		'Shortcuts:' \
		'  make irmas | make chinese' \
		'  make mel | make cqt | make mfcc | make chroma' \
		'' \
		'Helpers:' \
		'  make extract-help | make train-help | make clean'

train:
	"$(PY)" -m $(TRAIN_MODULE) $(TRAIN_ARGS)

train-dry:
	"$(PY)" -m $(TRAIN_MODULE) $(TRAIN_ARGS) --dry_run

train-resume:
	"$(PY)" -m $(TRAIN_MODULE) $(TRAIN_ARGS) --resume

train-help:
	"$(PY)" -m $(TRAIN_MODULE) --help

irmas: DATASET := irmas
irmas:
	@$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

chinese: DATASET := chinese_instruments
chinese:
	@$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

mel: FEATURE := mel
mel:
	@$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

cqt: FEATURE := cqt
cqt:
	@$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

mfcc: FEATURE := mfcc
mfcc:
	@$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

chroma: FEATURE := chroma
chroma:
	@$(MAKE) extract DATASET=$(DATASET) FEATURE=$(FEATURE) WORKERS=$(WORKERS)

extract-help:
	"$(PY)" -m $(EXTRACT_MODULE) --help

define EXTRACT_template
extract-$(1)-$(2):
	@echo ">>> extracting dataset=$(1) feature=$(2)"
	@$(MAKE) extract DATASET=$(1) FEATURE=$(2) WORKERS=$(WORKERS)
endef

$(foreach ds,$(ALL_DATASETS),$(foreach feat,$(ALL_FEATURES),$(eval $(call EXTRACT_template,$(ds),$(feat)))))

all: $(ALL_EXTRACT_TARGETS)

clean:
	"$(PY)" -c "from pathlib import Path; import shutil; shutil.rmtree(Path('data/processed'), ignore_errors=True)"
