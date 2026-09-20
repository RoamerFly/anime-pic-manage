"""Reproducible JSON/CSV result exports; never mutate source image files."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _materialize(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(result) for result in results]


def _flatten(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_json(results: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write UTF-8 JSON atomically and return the destination path."""

    path = Path(output_path).expanduser()
    materialized = _materialize(results)

    def write(destination: Path) -> None:
        destination.write_text(
            json.dumps(materialized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    _atomic_write(path, write)
    return path


def export_csv(results: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write a UTF-8 BOM CSV for Excel/Windows compatibility."""

    path = Path(output_path).expanduser()
    materialized = _materialize(results)
    columns: list[str] = []
    for row in materialized:
        for key in row:
            if key not in columns:
                columns.append(key)

    def write(destination: Path) -> None:
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(
                {key: _flatten(row.get(key)) for key in columns} for row in materialized
            )

    _atomic_write(path, write)
    return path


def export_results(
    results: Iterable[Mapping[str, Any]], output_path: str | Path, *, format: str | None = None
) -> Path:
    """Export results as JSON or CSV based on explicit format or suffix."""

    path = Path(output_path).expanduser()
    export_format = (format or path.suffix.lstrip(".")).casefold()
    if export_format == "json":
        return export_json(results, path)
    if export_format == "csv":
        return export_csv(results, path)
    raise ValueError("导出格式必须为 json 或 csv")
