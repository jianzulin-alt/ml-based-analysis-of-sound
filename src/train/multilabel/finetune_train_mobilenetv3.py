"""
Fine-tune an existing MobileNetV3 run on mixed mels.

python -m src.train.train_mobilenetv3_v1_to_mixed
"""

from pathlib import Path

import src
import yaml

from src.train.train_multilabel import multi_label_train_loop

PROJECT_ROOT = Path(src.__file__).resolve().parents[1]

# ---- Source run (where weights come from) ----
SOURCE_RUN_NAME = "MobileNetV3_v1"
SOURCE_WEIGHTS_DIR = PROJECT_ROOT / "src" / "models" / "saved_weights" / SOURCE_RUN_NAME
SOURCE_CKPT = SOURCE_WEIGHTS_DIR / "best_val.pt"  # or "last.pt"

# ---- New run (where fine-tune results will be written) ----
RUN_NAME = "MobileNetV3_v1_ft_mixed"
WEIGHTS_DIR = PROJECT_ROOT / "src" / "models" / "saved_weights" / RUN_NAME

# Train on mixed (and optionally pure)
MANIFESTS = [
    PROJECT_ROOT / "data" / "processed" / "train_mels_mixed.csv",
    PROJECT_ROOT / "data" / "processed" / "train_mels.csv",
]

LABELS_YAML = PROJECT_ROOT / "src" / "configs" / "labels.yaml"
AUDIO_CONFIG_YAML = PROJECT_ROOT / "src" / "configs" / "audio_params.yaml"
EXPERIMENT_LOG = PROJECT_ROOT / "src" / "models" / "experiments.csv"

MODEL_CONFIG = {
    "model_name": "mobilenet_v3_small",
    "in_ch": 2,
    "pretrained": False,              
    "freeze_backbone_epochs": 0,      
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "lr": 1e-4,                       
    "epochs": 40,                     
    "patience": 10,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "val_frac": 0.2,
    "seed": 1337,
    "threshold": 0.5,
    "num_workers": 2,
}


def main() -> None:
    with open(AUDIO_CONFIG_YAML, "r") as f:
        audio_params = yaml.safe_load(f) or {}
    with open(LABELS_YAML, "r") as f:
        label_config = yaml.safe_load(f) or {}

    classes = [c.strip().lower() for c in label_config.get("train_labels", [])]
    if not classes:
        raise ValueError(f"No train_labels found in {LABELS_YAML}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_CKPT.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {SOURCE_CKPT}. "
            f"Check SOURCE_RUN_NAME and whether best_val.pt exists."
        )

    print(f"Fine-tuning from: {SOURCE_CKPT}")
    print(f"Writing new run to: {WEIGHTS_DIR}")
    print(f"Training manifests: {[str(p) for p in MANIFESTS]}")

    results = multi_label_train_loop(
        manifest_csv=[str(p) for p in MANIFESTS],
        classes=classes,
        ckpt_dir=WEIGHTS_DIR,
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        val_frac=TRAIN_CONFIG["val_frac"],
        dropout=TRAIN_CONFIG["dropout"],
        patience=TRAIN_CONFIG["patience"],
        num_workers=TRAIN_CONFIG["num_workers"],
        threshold=TRAIN_CONFIG["threshold"],
        seed=TRAIN_CONFIG["seed"],
        audio_cfg=audio_params.get("audio", audio_params),
        resume_from=SOURCE_CKPT,                    # <-- key line
        model_name=MODEL_CONFIG["model_name"],
        in_ch=MODEL_CONFIG["in_ch"],
        pretrained=MODEL_CONFIG["pretrained"],
        freeze_backbone_epochs=MODEL_CONFIG["freeze_backbone_epochs"],
        run_name=RUN_NAME,
        experiment_log=EXPERIMENT_LOG,
    )

    print("Fine-tuning complete.")
    print("Summary:", results["summary"])


if __name__ == "__main__":
    main()
