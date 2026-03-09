VENV_PY := $(firstword $(wildcard .venv/bin/python) $(wildcard .venv/Scripts/python.exe))
PY ?= $(if $(VENV_PY),$(VENV_PY),python)

PY_MOD := $(PY) -m
PROCESSED_ROOT := data/processed
CONFIG_FILE := src/configs/audio_params.yaml
LABELS_CONFIG := src/configs/labels.yaml
IRMAS_TRAIN_DIR ?= data/IRMAS/Train
IRMAS_TRAIN_CACHE_ROOT ?= $(PROCESSED_ROOT)/irmas_train_mels
IRMAS_TRAIN_MEL_MANIFEST ?= data/IRMAS/IRMAS-TrainingData.csv

# Mix train mels and generate spectrograms (configured in src/configs/audio_params.yaml)

generate_mixed_train_mels:
	$(PY_MOD) src.scripts.generate_mixed_train_mels \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG)

generate_train_mels:
	$(PY_MOD) src.scripts.generate_log_mels \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG)

irmas_train_mels:
	$(PY_MOD) src.scripts.generate_irmas_train_mels \
		--irmas_train_dir $(IRMAS_TRAIN_DIR) \
		--cache_root $(IRMAS_TRAIN_CACHE_ROOT) \
		--mel_manifest_out $(IRMAS_TRAIN_MEL_MANIFEST)

TEST_DIR_AZ := data/test/a-touch-of-zen
TEST_MANIFEST_AZ := $(TEST_DIR_AZ).csv
TEST_DIR_IRMAS := data/test/IRMAS/IRMAS-TestingData-Part1
TEST_MANIFEST_IRMAS := $(TEST_DIR_IRMAS).csv
TEST_CHECKPOINT ?= src/models/saved_weights/MobileNetV3_v1/best_val.pt
TEST_OUTPUT_DIR ?=

generate_features: generate_train_mels generate_mixed_train_mels

test_manifest:
	@echo "Creating test manifest..."
	$(PY_MOD) src.scripts.generate_test_manifest \
		--test_dir $(TEST_DIR) \
		--out_csv $(OUT_CSV)

test_manifest_az:
	@$(MAKE) test_manifest TEST_DIR=$(TEST_DIR_AZ) OUT_CSV=$(TEST_MANIFEST_AZ)

test_manifest_irmas:
	@$(MAKE) test_manifest TEST_DIR=$(TEST_DIR_IRMAS) OUT_CSV=$(TEST_MANIFEST_IRMAS)

test_model:
	$(PY_MOD) src.test.test \
		--checkpoint $(TEST_CHECKPOINT) \
		--test_manifest $(TEST_MANIFEST_AZ) \
		$(if $(TEST_OUTPUT_DIR),--output_dir $(TEST_OUTPUT_DIR),) \
		--auto_threshold

clean:
	rm -rf $(PROCESSED_ROOT)
