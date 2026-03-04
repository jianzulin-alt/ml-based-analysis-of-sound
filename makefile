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

# One-shot pipeline (ordered)
all: \
	convert_mp3_wav \
	generate_train_mels \
	generate_chinese_train_cqt \
	generate_mixed_train_mel_cqt \
	generate_irmas_train_mels \
	generate_irmas_train_cqt \
	test_manifest_az \
	test_manifest_irmas \
	generate_irmas_test_cqt

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

train:
	$(PY_SRC) src/train/train.py --dataset $(TRAIN_DATASET) --feature $(TRAIN_FEATURE)

test:
	$(PY_SRC) src/test/test.py --dataset $(TEST_DATASET) --feature $(TEST_FEATURE)

clean:
	@PROCESSED_ROOT="$(PROCESSED_ROOT)" ALLOW_UNSAFE_RM="$(ALLOW_UNSAFE_RM)" $(PY) - <<'PY'
	import os
	from pathlib import Path
	root = Path(__file__).resolve().parent
	target = Path(os.environ.get("PROCESSED_ROOT", "data/processed")).expanduser().resolve()
	allow = os.environ.get("ALLOW_UNSAFE_RM", "0") == "1"
	def is_drive_root(p: Path) -> bool:
	    return p.parent == p
	if not allow:
	    if is_drive_root(target) or target == root or root not in target.parents:
	        raise SystemExit(f"[safe-guard] Refusing to delete: {target}")
	PY
	rm -rf $(PROCESSED_ROOT)
