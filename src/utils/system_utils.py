import hashlib
import os
import yaml
from pathlib import Path

from src.preprocessing.feature_modes import feature_mode_to_features, manifest_suffix_for_feature

def get_repo_root() -> Path:
    """Finds the repository root by walking up from this file's location."""
    return Path(__file__).resolve().parents[2]

def resolve_path(path_like: str | Path, root: Path) -> Path:
    """Resolves a path safely, checking absolute, cwd, and root."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (root / p).resolve()

def load_yaml(path: Path) -> dict:
    """Loads a YAML file into a dictionary safely."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def relative_to_root(path: Path, root: Path) -> str:
    """Returns a POSIX string path relative to the repository root."""
    return Path(os.path.relpath(path.resolve(), root)).as_posix()

def ensure_directory_exists(path: Path) -> Path:
    """Create the directory tree if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_path_hash(path_str: str) -> str:
    """Generate a short MD5 hash for a file path to prevent filename collisions in cache."""
    return hashlib.md5(path_str.encode("utf-8")).hexdigest()[:10]

def resolve_run_dir(root: Path, output_dir: str | None, exp_name: str) -> Path:
    if output_dir:
        return resolve_path(output_dir, root)
    return root / "src" / "models" / "saved_weights" / exp_name

