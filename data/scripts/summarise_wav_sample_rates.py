from __future__ import annotations

import argparse
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None


DEFAULT_ROOTS = [
    Path("data/IRMAS/IRMAS-TestingData-Part1"),
    Path("data/test/a-touch-of-zen"),
    Path("data/IRMAS/IRMAS-TrainingData"),
    Path("data/train"),
]


@dataclass(frozen=True)
class RootSummary:
    root: Path
    total_wavs: int
    sample_rate_counts: Counter[int]
    error_count: int


def display_relative(repo_root: Path, path: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def iter_wavs(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def read_sample_rate(path: Path) -> int:
    if sf is not None:
        return int(sf.info(str(path)).samplerate)

    with wave.open(str(path), "rb") as wav_handle:
        return int(wav_handle.getframerate())


def summarise_root(root: Path, pattern: str) -> RootSummary:
    sample_rate_counts: Counter[int] = Counter()
    error_count = 0
    wav_paths = iter_wavs(root, pattern)

    for wav_path in wav_paths:
        try:
            sample_rate = read_sample_rate(wav_path)
        except Exception:
            error_count += 1
            continue
        sample_rate_counts[sample_rate] += 1

    return RootSummary(
        root=root,
        total_wavs=len(wav_paths),
        sample_rate_counts=sample_rate_counts,
        error_count=error_count,
    )


def print_summary(summary: RootSummary, repo_root: Path) -> None:
    display_root = display_relative(repo_root, summary.root)
    print(f"Root: {display_root}")
    print(f"Total wavs: {summary.total_wavs}")
    print(f"Unique sample rates: {len(summary.sample_rate_counts)}")
    if summary.error_count:
        print(f"Unreadable wavs: {summary.error_count}")

    if not summary.sample_rate_counts:
        print("No readable wav files found.\n")
        return

    print("\nSample rate distribution:\n")
    width = max(len(str(sample_rate)) for sample_rate in summary.sample_rate_counts)
    for sample_rate, count in sorted(
        summary.sample_rate_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        pct = (count / summary.total_wavs) * 100 if summary.total_wavs else 0.0
        print(f"  {str(sample_rate).rjust(width)} Hz  {count}  ({pct:.1f}%)")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise WAV sample rates for one or more roots.")
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help=(
            "Dataset root to scan. Pass multiple times for multiple roots. "
            f"Defaults to: {', '.join(str(path) for path in DEFAULT_ROOTS)}"
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern to scan under each root (default: *.wav).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    roots = args.roots or DEFAULT_ROOTS

    combined_counts: Counter[int] = Counter()
    combined_total = 0
    combined_errors = 0

    for index, root in enumerate(roots):
        resolved_root = root if root.is_absolute() else (repo_root / root).resolve()
        if not resolved_root.exists():
            print(f"Root: {display_relative(repo_root, resolved_root)}")
            print("Path does not exist.\n")
            continue

        summary = summarise_root(resolved_root, args.pattern)
        print_summary(summary, repo_root)
        combined_counts.update(summary.sample_rate_counts)
        combined_total += summary.total_wavs
        combined_errors += summary.error_count

        if index != len(roots) - 1:
            print("-" * 50)

    if len(roots) <= 1:
        return

    print("Combined Summary")
    print(f"Total wavs: {combined_total}")
    print(f"Unique sample rates: {len(combined_counts)}")
    if combined_errors:
        print(f"Unreadable wavs: {combined_errors}")

    if not combined_counts:
        return

    print("\nSample rate distribution:\n")
    width = max(len(str(sample_rate)) for sample_rate in combined_counts)
    for sample_rate, count in sorted(combined_counts.items(), key=lambda item: (-item[1], item[0])):
        pct = (count / combined_total) * 100 if combined_total else 0.0
        print(f"  {str(sample_rate).rjust(width)} Hz  {count}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
