"""Where the data and artifact trees live.

The research repository resolved every path relative to its own checkout, so
scripts carried ``Path(__file__).resolve().parents[1]`` or, in the chain jobs, an
absolute ``/vf/users/...`` prefix. Neither survives being packaged.

Here the code and the data are separate concerns. Modules ask for
:func:`data_root` and the operator points it wherever the inputs actually are::

    export DEMAP_DATA_ROOT=/path/to/demap-data

Every documented workflow also takes explicit ``--`` paths, so the environment
variable is a convenience for the common case, never the only way in.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["data_root", "artifact_root", "repo_root", "resolve_data_path", "ensure_dir"]

ENV_VAR = "DEMAP_DATA_ROOT"
ARTIFACT_ENV_VAR = "DEMAP_ARTIFACT_ROOT"


def repo_root() -> Path:
    """Root of this source tree (the directory holding ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Root of the data and artifact trees.

    Defaults to the current working directory so a user who runs from inside a
    prepared data directory needs no configuration at all.
    """
    return Path(os.environ.get(ENV_VAR, ".")).resolve()


def artifact_root() -> Path:
    """Root of the frozen experiment artifacts.

    ``DEMAP_ARTIFACT_ROOT`` if set, otherwise :func:`data_root`. The two are
    documented as separate knobs because a reader may hold the prepared data
    tree without the multi-gigabyte artifact tree, or the reverse; in the
    research checkout they happen to be the same directory.
    """
    raw = os.environ.get(ARTIFACT_ENV_VAR)
    return Path(raw).resolve() if raw else data_root()


def resolve_data_path(path_str: str | os.PathLike) -> Path:
    """Absolute paths pass through; relative ones resolve against :func:`data_root`."""
    p = Path(path_str)
    return p if p.is_absolute() else data_root() / p


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
