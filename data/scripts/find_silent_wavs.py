"""
python3 data/scripts/find_silent_wavs.py --root data/train
python3 data/scripts/find_silent_wavs.py --root data/train --report data/processed/silent_wavs.tsv
python3 data/scripts/find_silent_wavs.py --root data/train --delete

"""
from __future__ import annotations

import argparse
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None

try:
    from scipy.io import wavfile
except ModuleNotFoundError:
    wavfile = None

DEFAULT_THRESHOLD_DBFS = -70.0 # - 80 is true silence


@dataclass(frozen=True)
class AudioStats:
    path: Path
    sample_rate: int
    frames: int
    peak: float
    rms: float

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / self.sample_rate

    @property
    def peak_dbfs(self) -> float:
        return amplitude_to_dbfs(self.peak)

    @property
    def rms_dbfs(self) -> float:
        return amplitude_to_dbfs(self.rms)


def amplitude_to_dbfs(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(value)


def dbfs_to_amplitude(value: float) -> float:
    return 10.0 ** (value / 20.0)


def format_dbfs(value: float) -> str:
    if not math.isfinite(value):
        return "-inf dBFS"
    return f"{value:.1f} dBFS"


def display_relative(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def iter_wavs(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def to_float32_waveform(waveform: np.ndarray) -> np.ndarray:
    if np.issubdtype(waveform.dtype, np.floating):
        return waveform.astype(np.float32, copy=False)

    if waveform.dtype == np.uint8:
        return ((waveform.astype(np.float32) - 128.0) / 128.0).astype(np.float32, copy=False)

    if np.issubdtype(waveform.dtype, np.signedinteger):
        max_abs = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)
        return (waveform.astype(np.float32) / float(max_abs)).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported waveform dtype: {waveform.dtype}")


def read_wav_with_scipy(path: Path) -> tuple[np.ndarray, int]:
    if wavfile is None:
        raise RuntimeError("scipy is not available")

    sample_rate, waveform = wavfile.read(str(path))
    waveform = np.asarray(waveform)
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]
    return to_float32_waveform(waveform), int(sample_rate)


def read_wav_with_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_handle:
        if wav_handle.getcomptype() != "NONE":
            raise ValueError(f"Unsupported WAV compression: {wav_handle.getcomptype()}")

        sample_rate = wav_handle.getframerate()
        channels = wav_handle.getnchannels()
        sample_width = wav_handle.getsampwidth()
        raw_frames = wav_handle.readframes(wav_handle.getnframes())

    if sample_width == 1:
        waveform = np.frombuffer(raw_frames, dtype=np.uint8)
        waveform = waveform.reshape(-1, channels)
        return ((waveform.astype(np.float32) - 128.0) / 128.0), sample_rate

    if sample_width == 2:
        waveform = np.frombuffer(raw_frames, dtype="<i2")
        waveform = waveform.reshape(-1, channels)
        return waveform.astype(np.float32) / 32768.0, sample_rate

    if sample_width == 3:
        packed = np.frombuffer(raw_frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        waveform = signed.reshape(-1, channels)
        return waveform.astype(np.float32) / 8388608.0, sample_rate

    if sample_width == 4:
        waveform = np.frombuffer(raw_frames, dtype="<i4")
        waveform = waveform.reshape(-1, channels)
        return waveform.astype(np.float32) / 2147483648.0, sample_rate

    raise ValueError(f"Unsupported sample width: {sample_width} bytes")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    if sf is not None:
        waveform, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        return waveform, int(sample_rate)

    if wavfile is not None:
        return read_wav_with_scipy(path)

    return read_wav_with_wave(path)


def analyse_wav(path: Path) -> AudioStats:
    waveform, sample_rate = read_wav(path)
    frames = int(waveform.shape[0])

    if waveform.size == 0:
        return AudioStats(
            path=path,
            sample_rate=sample_rate,
            frames=frames,
            peak=0.0,
            rms=0.0,
        )

    waveform64 = waveform.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(waveform64)))
    rms = float(np.sqrt(np.mean(np.square(waveform64))))

    return AudioStats(
        path=path,
        sample_rate=sample_rate,
        frames=frames,
        peak=peak,
        rms=rms,
    )


def write_report(
    report_path: Path,
    repo_root: Path,
    rows: list[AudioStats],
    *,
    delete_requested: bool,
    delete_statuses: dict[Path, str],
    delete_errors: dict[Path, str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "path\tduration_s\tpeak\tpeak_dbfs\trms\trms_dbfs\t"
            "delete_requested\tdelete_status\tdelete_error\n"
        )
        for row in rows:
            path_value = row.path
            try:
                path_value = row.path.relative_to(repo_root)
            except ValueError:
                pass

            handle.write(
                f"{path_value}\t"
                f"{row.duration_s:.3f}\t"
                f"{row.peak:.8f}\t"
                f"{row.peak_dbfs:.3f}\t"
                f"{row.rms:.8f}\t"
                f"{row.rms_dbfs:.3f}\t"
                f"{str(delete_requested).lower()}\t"
                f"{delete_statuses.get(row.path, 'not-requested')}\t"
                f"{delete_errors.get(row.path, '')}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan .wav files under a dataset root, report silent files, "
            "and optionally delete them."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/train"),
        help="Root dataset directory to scan (default: data/train)",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern to match audio files under --root (default: *.wav)",
    )
    parser.add_argument(
        "--silence-threshold-dbfs",
        type=float,
        default=DEFAULT_THRESHOLD_DBFS,
        help=(
            "Treat a file as silent when its absolute peak never rises above this "
            "threshold in dBFS (default: -80.0)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional TSV output path for the silent-file report.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Delete silent files after logging them. Disabled by default.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    root = args.root
    if not root.is_absolute():
        root = (repo_root / root).resolve()

    if not root.exists():
        print(f"[warn] root does not exist: {root}")
        return

    report_path = args.report
    if report_path is not None and not report_path.is_absolute():
        report_path = (repo_root / report_path).resolve()

    threshold_amplitude = dbfs_to_amplitude(args.silence_threshold_dbfs)
    wav_paths = list(iter_wavs(root, args.pattern))

    print(f"Root: {root}")
    print(f"Matched files: {len(wav_paths)}")
    print(
        "Silence threshold: "
        f"{args.silence_threshold_dbfs:.1f} dBFS "
        f"(peak <= {threshold_amplitude:.8f})"
    )
    print(f"Mode: {'delete' if args.delete else 'dry-run'}")

    if not wav_paths:
        return

    silent_rows: list[AudioStats] = []
    failed_paths: list[tuple[Path, str]] = []

    for wav_path in wav_paths:
        try:
            stats = analyse_wav(wav_path)
        except Exception as exc:
            failed_paths.append((wav_path, str(exc)))
            continue

        if stats.peak <= threshold_amplitude:
            silent_rows.append(stats)

    print(f"Silent files found: {len(silent_rows)}")
    print(f"Read failures: {len(failed_paths)}")

    delete_statuses = {
        row.path: ("pending" if args.delete else "not-requested")
        for row in silent_rows
    }
    delete_errors: dict[Path, str] = {}

    if silent_rows:
        print("\nSilent files:\n")
        for row in silent_rows:
            print(
                f"  {display_relative(root, row.path)} | duration={row.duration_s:.2f}s | "
                f"peak={format_dbfs(row.peak_dbfs)} | rms={format_dbfs(row.rms_dbfs)} | "
                f"delete={delete_statuses[row.path]}"
            )

    if failed_paths:
        print("\nFailed to read:\n")
        for failed_path, reason in failed_paths:
            print(f"  {display_relative(root, failed_path)} | {reason}")

    if not args.delete or not silent_rows:
        if not args.delete:
            print("\nDeletion disabled. Re-run with --delete to remove the silent files.")
        if report_path is not None:
            write_report(
                report_path,
                repo_root,
                silent_rows,
                delete_requested=args.delete,
                delete_statuses=delete_statuses,
                delete_errors=delete_errors,
            )
            print(f"\nReport written to: {report_path}")
        return

    deleted = 0
    delete_failures: list[tuple[Path, str]] = []

    for row in silent_rows:
        try:
            row.path.unlink()
            deleted += 1
            delete_statuses[row.path] = "deleted"
        except Exception as exc:
            delete_statuses[row.path] = "failed"
            delete_errors[row.path] = str(exc)
            delete_failures.append((row.path, str(exc)))

    print(f"\nDeleted files: {deleted}")
    print("\nDeletion log:\n")
    for row in silent_rows:
        line = f"  {display_relative(root, row.path)} | delete={delete_statuses[row.path]}"
        if row.path in delete_errors:
            line = f"{line} | error={delete_errors[row.path]}"
        print(line)

    if delete_failures:
        print(f"Delete failures: {len(delete_failures)}")
        for failed_path, reason in delete_failures:
            print(f"  {display_relative(root, failed_path)} | {reason}")

    if report_path is not None:
        write_report(
            report_path,
            repo_root,
            silent_rows,
            delete_requested=args.delete,
            delete_statuses=delete_statuses,
            delete_errors=delete_errors,
        )
        print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
