VENV_PY := $(firstword $(wildcard .venv/bin/python) $(wildcard .venv/Scripts/python.exe))
PY ?= $(if $(VENV_PY),$(VENV_PY),python)

PY_SRC := PYTHONPATH=src $(PY)
PROCESSED_ROOT := data/processed
CONFIG_FILE := src/configs/audio_params.yaml
LABELS_CONFIG := src/configs/labels.yaml
NUM_WORKERS ?= 19
IRMAS_CONFIG := src/configs/audio_params_irmas.yaml
IRMAS_LABELS := src/configs/labels_irmas.yaml
IRMAS_TRAIN_DIR := data/audio/IRMAS/IRMAS-TrainingData/IRMAS-TrainingData
IRMAS_TRAIN_MELS_MANIFEST := $(PROCESSED_ROOT)/irmas_train_mels.csv
CQT_CACHE_ROOT := $(PROCESSED_ROOT)/log_cqt
IRMAS_CQT_CACHE_ROOT := $(PROCESSED_ROOT)/irmas_cqt
IRMAS_TEST_CQT_CACHE_ROOT := $(PROCESSED_ROOT)/irmas_cqt_test
IRMAS_FMAX := 20000
ALLOW_UNSAFE_RM ?= 0

# Mix train mels and gennerate spectrogram
NUM_MIXES ?= 20000
MIN_SOURCES ?= 2
MAX_SOURCES ?= 2
SNR_DB_MIN ?= -3
SNR_DB_MAX ?= 6
NUM_MIXES ?= 12000 # start with ~150% of dataset size

MIXED_CACHE_ROOT := $(PROCESSED_ROOT)/log_mels_mixed
MIXED_CQT_CACHE_ROOT := $(PROCESSED_ROOT)/log_cqt_mixed
MIXED_MANIFEST := $(PROCESSED_ROOT)/train_mels_mixed.csv
TRAIN_DIR := data/train
GENERATE_TASK ?=
TRAIN_DATASET ?= chinese
TRAIN_FEATURE ?= mel
TEST_DATASET ?= chinese
TEST_FEATURE ?= mel
SUPPORTED_DATASETS := chinese irmas
SUPPORTED_FEATURES := mel cqt mel_cqt

TRAIN_KEY := $(TRAIN_DATASET)_$(TRAIN_FEATURE)
TEST_KEY := $(TEST_DATASET)_$(TEST_FEATURE)

TRAIN_GEN_TASKS_chinese_mel := convert_mp3_wav generate_train_mels
TRAIN_GEN_TASKS_chinese_cqt := convert_mp3_wav generate_train_mels generate_chinese_train_cqt
TRAIN_GEN_TASKS_chinese_mel_cqt := convert_mp3_wav generate_train_mels generate_chinese_train_cqt
TRAIN_GEN_TASKS_irmas_mel := generate_irmas_train_mels
TRAIN_GEN_TASKS_irmas_cqt := generate_irmas_train_mels generate_irmas_train_cqt
TRAIN_GEN_TASKS_irmas_mel_cqt := generate_irmas_train_mels generate_irmas_train_cqt

TEST_GEN_TASKS_chinese_mel := test_manifest_az
TEST_GEN_TASKS_chinese_cqt := test_manifest_az
TEST_GEN_TASKS_chinese_mel_cqt := test_manifest_az
TEST_GEN_TASKS_irmas_mel := test_manifest_irmas
TEST_GEN_TASKS_irmas_cqt := test_manifest_irmas generate_irmas_test_cqt
TEST_GEN_TASKS_irmas_mel_cqt := test_manifest_irmas generate_irmas_test_cqt

TRAIN_GEN_TARGETS := $(TRAIN_GEN_TASKS_$(TRAIN_KEY))
TEST_GEN_TARGETS := $(TEST_GEN_TASKS_$(TEST_KEY))

.PHONY: \
	all all_full all_selected clean generate \
	prepare_train_features prepare_test_features validate_targets \
	generate_train_mels generate_irmas_train_mels \
	generate_chinese_train_cqt generate_irmas_train_cqt \
	generate_mixed_train_mels generate_mixed_train_mel_cqt \
	test_manifest test_manifest_az test_manifest_irmas \
	generate_irmas_test_cqt convert_mp3_wav

# NOTE: premixing will be replaced with mixing at train time to save storage
generate_mixed_train_mels:
	NUM_WORKERS=$(NUM_WORKERS) NUM_MIXES=$(NUM_MIXES) $(PY_SRC) src/scripts/generate.py --task mixed_mel

generate_mixed_train_mel_cqt:
	NUM_WORKERS=$(NUM_WORKERS) NUM_MIXES=$(NUM_MIXES) $(PY_SRC) src/scripts/generate.py --task mixed_mel_cqt

generate_train_mels:
	NUM_WORKERS=$(NUM_WORKERS) $(PY_SRC) src/scripts/generate.py --task chinese_mel

generate_irmas_train_mels:
	NUM_WORKERS=$(NUM_WORKERS) $(PY_SRC) src/scripts/generate.py --task irmas_mel

generate_chinese_train_cqt:
	NUM_WORKERS=$(NUM_WORKERS) $(PY_SRC) src/scripts/generate.py --task chinese_cqt

generate_irmas_train_cqt:
	NUM_WORKERS=$(NUM_WORKERS) $(PY_SRC) src/scripts/generate.py --task irmas_cqt

convert_mp3_wav:
	$(PY_SRC) src/scripts/generate.py --task convert_mp3_wav

generate:
ifeq ($(strip $(GENERATE_TASK)),)
	NUM_WORKERS=$(NUM_WORKERS) NUM_MIXES=$(NUM_MIXES) $(PY_SRC) src/scripts/generate.py
else
	NUM_WORKERS=$(NUM_WORKERS) NUM_MIXES=$(NUM_MIXES) $(PY_SRC) src/scripts/generate.py --task $(GENERATE_TASK)
endif

TEST_DIR_AZ := data/test/a-touch-of-zen
TEST_MANIFEST_AZ := $(TEST_DIR_AZ).csv
TEST_DIR_IRMAS := data/audio/IRMAS/IRMAS-TestingData-Part1/IRMAS-TestingData-Part1/Part1
TEST_MANIFEST_IRMAS := data/test/IRMAS/IRMAS-TestingData-Part1.csv

generate_features: generate_train_mels generate_mixed_train_mels

# Full preprocessing (legacy ordered pipeline)
all_full: \
	convert_mp3_wav \
	generate_train_mels \
	generate_chinese_train_cqt \
	generate_mixed_train_mel_cqt \
	generate_irmas_train_mels \
	generate_irmas_train_cqt \
	test_manifest_az \
	test_manifest_irmas \
	generate_irmas_test_cqt

validate_targets:
	@$(PY_SRC) -c "train_dataset='$(TRAIN_DATASET)'; train_feature='$(TRAIN_FEATURE)'; test_dataset='$(TEST_DATASET)'; test_feature='$(TEST_FEATURE)'; supported_datasets='$(SUPPORTED_DATASETS)'.split(); supported_features='$(SUPPORTED_FEATURES)'.split(); train_targets='$(TRAIN_GEN_TARGETS)'.split(); test_targets='$(TEST_GEN_TARGETS)'.split(); assert train_dataset in supported_datasets, f'Unsupported TRAIN_DATASET={train_dataset}. Expected one of {supported_datasets}'; assert test_dataset in supported_datasets, f'Unsupported TEST_DATASET={test_dataset}. Expected one of {supported_datasets}'; assert train_feature in supported_features, f'Unsupported TRAIN_FEATURE={train_feature}. Expected one of {supported_features}'; assert test_feature in supported_features, f'Unsupported TEST_FEATURE={test_feature}. Expected one of {supported_features}'; assert train_targets, f'No generation steps mapped for TRAIN={train_dataset}/{train_feature}'; assert test_targets, f'No generation steps mapped for TEST={test_dataset}/{test_feature}'; print('[make] TRAIN targets:', ' '.join(train_targets)); print('[make] TEST targets: ', ' '.join(test_targets))"

prepare_train_features: validate_targets $(TRAIN_GEN_TARGETS)

prepare_test_features: validate_targets $(TEST_GEN_TARGETS)

# Smart one-shot feature generation based on train/test selection.
# Example:
#   make all_selected TRAIN_DATASET=chinese TRAIN_FEATURE=mel_cqt TEST_DATASET=irmas TEST_FEATURE=mel
all_selected: prepare_train_features prepare_test_features

# Default one-shot generation: build all feature presets.
all: all_full

test_manifest:
	@echo "Creating test manifest..."
	$(PY_SRC) src/scripts/generate_test_manifest.py \
		--test_dir $(TEST_DIR) \
		--out_csv $(OUT_CSV)

test_manifest_az:
	$(PY_SRC) src/scripts/generate.py --task test_manifest_az

test_manifest_irmas:
	$(PY_SRC) src/scripts/generate.py --task test_manifest_irmas

generate_irmas_test_cqt:
	NUM_WORKERS=$(NUM_WORKERS) $(PY_SRC) src/scripts/generate.py --task irmas_test_cqt

clean:
	@PROCESSED_ROOT="$(PROCESSED_ROOT)" ALLOW_UNSAFE_RM="$(ALLOW_UNSAFE_RM)" $(PY) -c "import os, shutil, sys; from pathlib import Path; root = Path.cwd().resolve(); raw_target = Path(os.environ.get('PROCESSED_ROOT', 'data/processed')).expanduser(); target = (raw_target if raw_target.is_absolute() else (root / raw_target)).resolve(); allow = os.environ.get('ALLOW_UNSAFE_RM', '0') == '1'; is_drive_root = target.parent == target; blocked = (not allow) and (is_drive_root or target == root or root not in target.parents); sys.exit(f'[safe-guard] Refusing to delete: {target}') if blocked else None; print(f'[clean] Removing: {target}'); shutil.rmtree(target, ignore_errors=True)"
