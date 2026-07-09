"""Packaging metadata checks."""

from __future__ import annotations

import re
from pathlib import Path


def test_declared_license_file_exists() -> None:
    """The file referenced by project.license must be present."""

    project_root = Path(__file__).resolve().parents[1]
    pyproject = project_root / "pyproject.toml"
    match = re.search(
        r'^license\s*=\s*\{\s*file\s*=\s*"([^"]+)"\s*\}',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )

    assert match is not None
    assert (project_root / match.group(1)).is_file()
