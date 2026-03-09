"""
Notebook-style training entrypoint for single-label CNN.
Edit constants below, then run:

python -m src.train.train_cnn
"""

from pathlib import Path

import src
import yaml

from src.train.train_singlelabel import single_label_train_loop

PROJECT_ROOT = Path(src.__file__).resolve().parents[1]

RUN_NAME = "CNN_singlelabel_irmas_v1"
WEIGHTS_DIR = PROJECT_ROOT / "src" / "models" / "saved_weights" / RUN_NAME
MANIFESTS = [
    PROJECT_ROOT / "data" / "processed" / "train_mels.csv",
]
LABELS_YAML = PROJECT_ROOT / "src" / "configs" / "labels.yaml"
AUDIO_CONFIG_YAML = PROJECT_ROOT / "src" / "configs" / "audio_params.yaml"
EXPERIMENT_LOG = PROJECT_ROOT / "src" / "models" / "experiments_singlelabel.csv"
LABEL_KEY = None  # Set "irmas" to force class ordering from labels.yaml.

MODEL_CONFIG = {
    "model_name": "cnn",
    "in_ch": 2,
    "pretrained": False,
    "freeze_backbone_epochs": 0,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 120,
    "patience": 20,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "val_frac": 0.2,
    "seed": 1337,
    "num_workers": 0,
}


def main() -> None:
    with open(AUDIO_CONFIG_YAML, "r") as f:
        audio_params = yaml.safe_load(f) or {}
    with open(LABELS_YAML, "r") as f:
        label_config = yaml.safe_load(f) or {}

    classes = None
    class_source = "manifest"
    if LABEL_KEY is not None:
        classes = [c.strip().lower() for c in label_config.get(LABEL_KEY, [])]
        if not classes:
            raise ValueError(f"No labels found for key '{LABEL_KEY}' in {LABELS_YAML}")
        class_source = f"labels.yaml:{LABEL_KEY}"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    resume_ckpt = WEIGHTS_DIR / "last.pt"
    if not resume_ckpt.exists():
        resume_ckpt = None
        print("Starting fresh. No previous weights found.")
    else:
        print(f"Resuming from {resume_ckpt}")

    results = single_label_train_loop(
        manifest_csv=[str(p) for p in MANIFESTS],
        ckpt_dir=WEIGHTS_DIR,
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        val_frac=TRAIN_CONFIG["val_frac"],
        dropout=TRAIN_CONFIG["dropout"],
        patience=TRAIN_CONFIG["patience"],
        num_workers=TRAIN_CONFIG["num_workers"],
        seed=TRAIN_CONFIG["seed"],
        audio_cfg=audio_params.get("audio", audio_params),
        classes=classes,
        class_source=class_source,
        resume_from=resume_ckpt,
        model_name=MODEL_CONFIG["model_name"],
        in_ch=MODEL_CONFIG["in_ch"],
        pretrained=MODEL_CONFIG["pretrained"],
        freeze_backbone_epochs=MODEL_CONFIG["freeze_backbone_epochs"],
        run_name=RUN_NAME,
        experiment_log=EXPERIMENT_LOG,
    )
    print("Training complete.")
    print("Summary:", results["summary"])


if __name__ == "__main__":
    main()
