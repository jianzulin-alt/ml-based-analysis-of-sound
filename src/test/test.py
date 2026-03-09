"""Evaluate a trained model checkpoint on a test manifest and save results.

Example:
python -m src.test.test \
  --checkpoint src/models/saved_weights/MobileNetV3_v1/best_val.pt \
  --test_manifest data/test/a-touch-of-zen.csv \
  --auto_threshold

python -m src.test.test \
  --checkpoint src/models/saved_weights/CNN_v1/best_val.pt \
  --test_manifest data/test/a-touch-of-zen.csv \
  --auto_threshold
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

import src
from src.test.utils import (
    evaluate_multilabel_performance,
    find_best_threshold,
    run_inference,
    save_test_artifacts,
)

REPO_ROOT = Path(src.__file__).resolve().parents[1]
SUPPORTED_MODELS = ("auto", "cnn", "crnn", "mobilenet_v3_small")


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / p).resolve()


def detect_model_name(ckpt: dict, arg_model: str) -> str:
    if arg_model != "auto":
        return arg_model
    ckpt_name = str(ckpt.get("model_name", "cnn")).strip().lower()
    alias_map = {
        "mobilenet_v3": "mobilenet_v3_small",
        "mobilenetv3": "mobilenet_v3_small",
        "mobilenetv3_small": "mobilenet_v3_small",
    }
    return alias_map.get(ckpt_name, ckpt_name)


def model_factory(model_name: str, in_ch: int):
    if model_name == "cnn":
        from src.models.CNN import CNN

        return CNN, {"in_ch": in_ch}
    if model_name == "crnn":
        from src.models.CRNN import CRNN

        return CRNN, {}
    if model_name == "mobilenet_v3_small":
        from src.models.MobileNetV3 import MobileNetV3Small

        # Pretrained=False for eval load; checkpoint weights are loaded afterward.
        return MobileNetV3Small, {"in_ch": in_ch, "pretrained": False}
    raise ValueError(f"Unsupported model '{model_name}'.")


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    ap.add_argument("--test_manifest", required=True, help="CSV with wav_path and txt_path columns")
    ap.add_argument("--model", choices=SUPPORTED_MODELS, default="auto")
    ap.add_argument("--in_ch", type=int, default=2, help="Input channels for CNN/MobileNet models")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for multilabel outputs")
    ap.add_argument("--auto_threshold", action="store_true", help="Search threshold on the provided test set")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--output_dir", default=None, help="Output directory. Default: checkpoint-based timestamped dir")
    ap.add_argument(
        "--results_log",
        default="src/models/test_results.csv",
        help="CSV to append summary row. Pass empty string to disable.",
    )
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar")
    args = ap.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    test_manifest_path = resolve_path(args.test_manifest)
    device = get_device(args.device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = detect_model_name(ckpt, args.model)
    model_cls, model_kwargs = model_factory(model_name, args.in_ch)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Test manifest: {test_manifest_path}")
    print(f"Device: {device}")

    preds_arr, gts_arr, sample_ids, _, valid_labels, _ = run_inference(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        model_weights_path=checkpoint_path,
        device=device,
        test_manifest_csv=test_manifest_path,
        show_progress=not args.no_progress,
    )

    threshold = float(args.threshold)
    if args.auto_threshold:
        print("Searching best threshold on this test set...")
        threshold = float(find_best_threshold(preds_arr, gts_arr, valid_labels, show_plot=False))
    print(f"Using threshold: {threshold:.2f}")

    report = evaluate_multilabel_performance(
        all_preds=preds_arr,
        all_gt=gts_arr,
        class_list=valid_labels,
        sample_ids=sample_ids,
        threshold=threshold,
        debug=False,
    )

    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
    else:
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = checkpoint_path.parent / "test_results" / f"{test_manifest_path.stem}_{stamp}"

    results_log = resolve_path(args.results_log) if args.results_log.strip() else None

    summary = save_test_artifacts(
        output_dir=output_dir,
        report=report,
        preds_probs=preds_arr,
        gts=gts_arr,
        sample_ids=sample_ids,
        class_list=valid_labels,
        checkpoint_path=checkpoint_path,
        test_manifest_path=test_manifest_path,
        threshold=threshold,
        append_log_csv=results_log,
    )

    print(f"Saved test outputs to: {output_dir}")
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
