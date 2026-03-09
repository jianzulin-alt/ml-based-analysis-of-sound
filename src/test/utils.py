import torch
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def evaluate_multilabel_performance(
    all_preds,
    all_gt,
    class_list,
    sample_ids=None,
    threshold=0.5,
    debug=False,
    zero_division=0,
):
    classes = [c.strip().lower() for c in class_list]
    probs = np.asarray(all_preds)
    gts = np.asarray(all_gt).astype(int)

    if probs.shape != gts.shape:
        raise ValueError(f"Shape mismatch: preds {probs.shape} vs gts {gts.shape}")

    preds = (probs >= threshold).astype(int)

    num_samples, num_classes = preds.shape

    # ---- Key counts ----
    total_pos_gt = int(gts.sum())
    total_pos_pred = int(preds.sum())
    total_entries = int(num_samples * num_classes)

    # Confusion totals across ALL labels (micro)
    tp = int(((preds == 1) & (gts == 1)).sum())
    fp = int(((preds == 1) & (gts == 0)).sum())
    fn = int(((preds == 0) & (gts == 1)).sum())
    tn = int(((preds == 0) & (gts == 0)).sum())

    # ---- None prediction rate ----
    none_pred_mask = preds.sum(axis=1) == 0
    num_none = int(none_pred_mask.sum())

    # ---- Metrics ----
    subset_acc = accuracy_score(gts, preds)  # exact match
    hamming_acc = 1.0 - hamming_loss(gts, preds)  # label-wise accuracy

    report = classification_report(
        gts,
        preds,
        target_names=classes,
        output_dict=True,
        zero_division=zero_division,
    )

    # ---- Print summary ----
    print(f"Classification threshold probability: {threshold}")
    print(
        f"Samples: {num_samples} | Classes: {num_classes} | Decisions: {total_entries}"
    )
    print(
        f"GT positives: {total_pos_gt} ({total_pos_gt/total_entries:.2%} of all decisions)"
    )
    print(
        f"Pred positives: {total_pos_pred} ({total_pos_pred/total_entries:.2%} of all decisions)"
    )
    print(f"Predicted 'None' (all-zero): {num_none} ({num_none/num_samples:.2%})")
    print("")
    print(f"Hamming accuracy (label-wise): {hamming_acc:.2%}")
    print(f"Subset accuracy (exact match): {subset_acc:.2%}")
    print(f"Micro F1:  {report['micro avg']['f1-score']:.4f}")
    print(f"Macro F1:  {report['macro avg']['f1-score']:.4f}")
    # print(f"Micro Prec:{report['micro avg']['precision']:.4f} | Micro Rec:{report['micro avg']['recall']:.4f}")
    print("")

    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    if total_pos_pred < max(5, 0.02 * total_pos_gt):
        print("WARNING: Very few positive predictions relative to GT positives.")
        print(
            "         Your threshold is likely too high, or logits are miscalibrated.\n"
        )

    # ---- Per-class table ----
    pos_per_class = gts.sum(axis=0)
    print(
        f"{'Instrument':<15} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>7} | {'Pred':>5}"
    )
    print("-" * 70)

    for i, name in enumerate(classes):
        support = int(pos_per_class[i])
        pred_count = int(preds[:, i].sum())

        if support == 0:
            print(
                f"{name:<15} | {'  n/a':>6} | {'  n/a':>6} | {'  n/a':>6} | {support:>7} | {pred_count:>5}"
            )
            continue

        prec = report[name]["precision"]
        rec = report[name]["recall"]
        f1 = report[name]["f1-score"]
        print(
            f"{name:<15} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f} | {support:>7} | {pred_count:>5}"
        )

    # ---- Debug examples where GT had positives ----
    if debug and sample_ids is not None:
        print("\n--- DEBUG: Examples where GT has at least one label ---")
        gt_nonzero = np.where(gts.sum(axis=1) > 0)[0]
        for idx in gt_nonzero[:]:
            pred_names = [classes[j] for j, v in enumerate(preds[idx]) if v]
            gt_names = [classes[j] for j, v in enumerate(gts[idx]) if v]
            print(f"ID: {sample_ids[idx]}")
            print(f"  Predicted: {pred_names if pred_names else '(none)'}")
            print(f"  Actual:    {gt_names if gt_names else '(none)'}")
            print("-" * 30)

    return report


def parse_ground_truth(txt_path, label_to_idx):
    """
    Parses text files. Handles newlines, tabs, and commas. 
    Only keeps labels present in the training set.
    """
    path = Path(txt_path)
    gt_vector = np.zeros(len(label_to_idx))
    if not path.exists():
        return gt_vector

    with open(path, 'r') as f:
        content = f.read()
    
    # Regex split handles \n, \t, and commas simultaneously
    raw_labels = re.split(r'[\n,\t]', content)
    
    for label in raw_labels:
        clean = label.strip().lower()
        if clean in label_to_idx:
            gt_vector[label_to_idx[clean]] = 1.0
            
    return gt_vector

def find_best_threshold(preds_probs, gts, labels, show_plot=False):
    thresholds = np.arange(0.05, 1, 0.05)
    micro_f1s = []
    macro_f1s = []
    
    print(f"{'Threshold':<10} | {'Micro F1':<10} | {'Macro F1':<10} | {'Subset Acc':<10}")
    print("-" * 50)

    for t in thresholds:
        # Apply current threshold
        current_preds = (preds_probs > t).astype(int)
        
        # Calculate metrics
        rep = classification_report(gts, current_preds, target_names=labels, output_dict=True, zero_division=0)
        sub_acc = accuracy_score(gts, current_preds)
        
        micro_f1 = rep['micro avg']['f1-score']
        macro_f1 = rep['macro avg']['f1-score']
        
        micro_f1s.append(micro_f1)
        macro_f1s.append(macro_f1)
        
        print(f"{t:<10.2f} | {micro_f1:<10.4f} | {macro_f1:<10.4f} | {sub_acc:<10.2%}")

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, micro_f1s, label='Micro F1', marker='o')
        plt.plot(thresholds, macro_f1s, label='Macro F1', marker='s')
        plt.xlabel('Detection Threshold')
        plt.ylabel('F1 Score')
        plt.title('Threshold vs. Model Performance')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Identify best thresholds
    best_micro_t = thresholds[np.argmax(micro_f1s)]
    best_macro_t = thresholds[np.argmax(macro_f1s)]
    
    print(f"\nBest Threshold (Micro F1): {best_micro_t:.2f} (Score: {max(micro_f1s):.4f})")
    print(f"Best Threshold (Macro F1): {best_macro_t:.2f} (Score: {max(macro_f1s):.4f})")
    
    return best_micro_t

def display_formatted_results(results_dict):
    """
    Converts the classification report dictionary into a styled Pandas DataFrame
    for clear visualisation in Jupyter Notebooks.
    """
    # 1. Convert dictionary to DataFrame and Transpose
    df = pd.DataFrame(results_dict).transpose()

    # 2. Remove the global 'accuracy' row to keep the table focus on per-class metrics
    if 'accuracy' in df.index:
        df = df.drop('accuracy')

    # 3. Apply styling: 4 decimal places and a colour gradient for the F1-Score
    # This helps identify underperforming instruments (like pipa or sheng) at a glance.
    styled_df = df.style.format({
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1-score": "{:.4f}",
        "support": "{:.0f}"
    }).background_gradient(cmap='YlGnBu', subset=['f1-score'])

    print("\n--- Detailed Classification Report ---")
    return styled_df

def run_inference(
    *,
    model_cls,
    model_kwargs: dict,
    model_weights_path,
    device,
    test_manifest_csv,
    root: Path | None = None,
    state_key: str = "model_state",
    audio_cfg_key: str = "audio_config",
    classes_key: str = "classes",
    strict_load: bool = True,
    show_progress: bool = True,
):
    """
    Loads checkpoint + model, runs inference over a manifest CSV.

    Returns:
        preds_arr: (N, C) float array of predicted probabilities
        gts_arr:   (N, C) int/bool array of ground-truth multi-hot vectors
        sample_ids: list[str] stems of wav filenames
        audio_cfg: dict-like audio config from checkpoint
        valid_labels: list[str] normalised class names
        label_to_idx: dict[str, int]
    """
    # Local import keeps module import light for CLI --help usage.
    from src.inference.utils import get_prediction, load_and_preprocess

    ckpt = torch.load(model_weights_path, map_location=device)

    audio_cfg = ckpt[audio_cfg_key]
    valid_labels = [c.strip().lower() for c in ckpt[classes_key]]
    label_to_idx = {name: i for i, name in enumerate(valid_labels)}

    model = model_cls(**model_kwargs, num_classes=len(valid_labels)).to(device)
    model.load_state_dict(ckpt[state_key], strict=strict_load)
    model.eval()

    df = pd.read_csv(test_manifest_csv)
    required_cols = {"wav_path", "txt_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Test manifest must include columns {sorted(required_cols)}. Missing: {sorted(missing)}"
        )

    base_root = PROJECT_ROOT if root is None else Path(root)

    def _resolve_path(p):
        p = Path(p)
        return p if p.is_absolute() else (base_root / p).resolve()

    df["wav_path"] = df["wav_path"].apply(lambda p: str(_resolve_path(p)))
    df["txt_path"] = df["txt_path"].apply(lambda p: str(_resolve_path(p)))

    all_preds, all_gt, sample_ids = [], [], []

    print(f"Running inference on {len(df)} samples against {len(valid_labels)} classes...")

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df))

    with torch.no_grad():
        for _, row in iterator:
            gt_vec = parse_ground_truth(row["txt_path"], label_to_idx)
            mel = load_and_preprocess(row["wav_path"], audio_cfg)
            probs = get_prediction(model, mel, device)

            all_preds.append(probs)
            all_gt.append(gt_vec)
            sample_ids.append(Path(row["wav_path"]).stem)

    preds_arr = np.asarray(all_preds)
    gts_arr = np.asarray(all_gt)

    return preds_arr, gts_arr, sample_ids, audio_cfg, valid_labels, label_to_idx


def _vector_to_labels(vec: np.ndarray, class_list: list[str]) -> str:
    names = [class_list[i] for i, value in enumerate(vec) if int(value) == 1]
    return "|".join(names)


def predictions_to_dataframe(
    preds_probs: np.ndarray,
    gts: np.ndarray,
    sample_ids: list[str],
    class_list: list[str],
    threshold: float,
) -> pd.DataFrame:
    preds_bin = (preds_probs >= threshold).astype(int)
    gts_bin = gts.astype(int)

    base_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "gt_labels": [_vector_to_labels(row, class_list) for row in gts_bin],
            "pred_labels": [_vector_to_labels(row, class_list) for row in preds_bin],
            "exact_match": (preds_bin == gts_bin).all(axis=1).astype(int),
        }
    )

    prob_df = pd.DataFrame(preds_probs, columns=[f"prob__{c}" for c in class_list])
    gt_df = pd.DataFrame(gts_bin, columns=[f"gt__{c}" for c in class_list])
    pred_df = pd.DataFrame(preds_bin, columns=[f"pred__{c}" for c in class_list])
    return pd.concat([base_df, prob_df, gt_df, pred_df], axis=1)


def build_evaluation_summary(
    *,
    report: dict,
    preds_probs: np.ndarray,
    gts: np.ndarray,
    class_list: list[str],
    checkpoint_path: Path,
    test_manifest_path: Path,
    threshold: float,
) -> dict:
    preds_bin = (preds_probs >= threshold).astype(int)
    gts_bin = gts.astype(int)

    return {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": checkpoint_path.parent.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "test_manifest": str(test_manifest_path.resolve()),
        "num_samples": int(gts_bin.shape[0]),
        "num_classes": int(len(class_list)),
        "threshold": float(threshold),
        "hamming_accuracy": float(1.0 - hamming_loss(gts_bin, preds_bin)),
        "subset_accuracy": float(accuracy_score(gts_bin, preds_bin)),
        "micro_precision": float(report["micro avg"]["precision"]),
        "micro_recall": float(report["micro avg"]["recall"]),
        "micro_f1": float(report["micro avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


TEST_RESULT_LOG_FIELDS = [
    "timestamp_utc",
    "run_name",
    "checkpoint_path",
    "test_manifest",
    "num_samples",
    "num_classes",
    "threshold",
    "hamming_accuracy",
    "subset_accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]


def append_test_result_log(log_csv_path: Path, summary: dict) -> None:
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([{k: summary.get(k, "") for k in TEST_RESULT_LOG_FIELDS}])
    if log_csv_path.exists():
        row_df.to_csv(log_csv_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_csv_path, index=False)


def save_test_artifacts(
    *,
    output_dir: Path,
    report: dict,
    preds_probs: np.ndarray,
    gts: np.ndarray,
    sample_ids: list[str],
    class_list: list[str],
    checkpoint_path: Path,
    test_manifest_path: Path,
    threshold: float,
    append_log_csv: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_evaluation_summary(
        report=report,
        preds_probs=preds_probs,
        gts=gts,
        class_list=class_list,
        checkpoint_path=checkpoint_path,
        test_manifest_path=test_manifest_path,
        threshold=threshold,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report.csv", index=True)

    preds_df = predictions_to_dataframe(
        preds_probs=preds_probs,
        gts=gts,
        sample_ids=sample_ids,
        class_list=class_list,
        threshold=threshold,
    )
    preds_df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if append_log_csv is not None:
        append_test_result_log(append_log_csv, summary)

    return summary
