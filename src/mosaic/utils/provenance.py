"""Provenance capture: code version, hardware, library versions.

Every experiment record embeds this so a result can be traced to the exact
software and machine that produced it. Values are read, never hard-coded.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from mosaic.schema.ids import stable_hash

#: Packages whose versions materially change numeric results.
TRACKED_PACKAGES = (
    "polars",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "networkx",
    "lightgbm",
    "xgboost",
    "duckdb",
    "pyarrow",
)


def _run_git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@lru_cache(maxsize=1)
def git_commit(repo_root: Path | None = None) -> str:
    """Short commit sha + dirty flag, or ``"unknown"`` outside a git checkout."""
    root = repo_root or Path(__file__).resolve().parents[3]
    sha = _run_git("rev-parse", "--short", "HEAD", cwd=root)
    if not sha:
        return "unknown"
    status = _run_git("status", "--porcelain", cwd=root)
    return f"{sha}{'-dirty' if status else ''}"


def code_version() -> str:
    """Version of the ``mosaic`` package itself."""
    try:
        return md.version("mosaic")
    except md.PackageNotFoundError:  # editable/src layout without install
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from mosaic import __version__

            return __version__
        except Exception:  # pragma: no cover
            return "0.0.0+unknown"


@lru_cache(maxsize=1)
def hardware_info() -> dict[str, Any]:
    """Machine description recorded with every scaling experiment."""
    import os

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": _physical_cores(),
        "ram_gb": _total_ram_gb(),
    }
    try:
        import psutil  # type: ignore[import-not-found]

        freq = psutil.cpu_freq()
        if freq is not None:
            info["cpu_max_mhz"] = round(freq.max, 1)
    except Exception:
        pass
    return info


def _physical_cores() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _total_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        try:  # POSIX fallback
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1)
        except (AttributeError, ValueError, OSError):
            return None


def library_versions() -> dict[str, str]:
    """Installed versions of the packages that can move a metric."""
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "absent"
    return out


def environment_fingerprint() -> str:
    """Hash of the parts of the environment that can change results."""
    return stable_hash(
        {
            "code": code_version(),
            "git": git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "libraries": library_versions(),
        },
        length=20,
    )
