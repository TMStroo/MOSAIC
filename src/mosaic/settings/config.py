"""Typed settings, loaded from environment with a checked-in default.

Precedence: explicit constructor > environment (``.env`` then OS) > defaults.
All paths are resolved against the repository root so a fresh clone works with
zero configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_data_root() -> Path:
    return REPO_ROOT / "data"


class Settings(BaseSettings):
    """Runtime configuration for every MOSAIC subsystem."""

    model_config = SettingsConfigDict(
        env_prefix="MOSAIC_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- paths ---------------------------------------------------------------
    data_root: Path = Field(default_factory=_default_data_root)
    cache_root: Path = Field(default=REPO_ROOT / "var" / "cache")
    results_root: Path = Field(default=REPO_ROOT / "results")
    reports_root: Path = Field(default=REPO_ROOT / "reports" / "generated")
    db_url: str = Field(default="")

    # --- runtime -------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    seed: int = 20260901
    n_jobs: int = -1

    # --- api -----------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- research defaults ---------------------------------------------------
    default_fpr_target: float = 0.01
    default_seeds: list[int] = Field(default_factory=lambda: [11, 23, 37, 59, 71])

    @field_validator("data_root", "cache_root", "results_root", "reports_root", mode="before")
    @classmethod
    def _abs(cls, value: Any) -> Any:
        if value in (None, ""):
            return value
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path)

    @property
    def db(self) -> str:
        """Resolved SQLAlchemy URL; SQLite by default so nothing external is needed."""
        if self.db_url:
            return self.db_url
        return f"sqlite:///{(REPO_ROOT / 'var' / 'mosaic.sqlite').as_posix()}"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def manifests_dir(self) -> Path:
        return self.data_root / "manifests"

    @property
    def synthetic_dir(self) -> Path:
        return self.data_root / "synthetic"

    @property
    def experiments_dir(self) -> Path:
        return self.results_root / "experiments"

    @property
    def figures_dir(self) -> Path:
        return self.results_root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.results_root / "tables"

    def ensure_dirs(self) -> None:
        for path in (
            self.cache_root,
            self.results_root,
            self.reports_root,
            self.experiments_dir,
            self.figures_dir,
            self.tables_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.manifests_dir,
            self.synthetic_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def n_workers(self) -> int:
        return os.cpu_count() or 1 if self.n_jobs == -1 else max(1, self.n_jobs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings (override in tests with ``get_settings.cache_clear()``)."""
    return Settings()
