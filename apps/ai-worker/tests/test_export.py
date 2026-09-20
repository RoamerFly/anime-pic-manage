from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_worker.export import export_results


def test_json_and_csv_exports_keep_unicode(tmp_path: Path) -> None:
    rows = [{"character_tag": "character_alpha", "display_name": "角色甲", "top": {"score": 0.9}}]
    json_path = export_results(rows, tmp_path / "结果.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["display_name"] == "角色甲"
    csv_path = export_results(rows, tmp_path / "结果.csv")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        parsed = list(csv.DictReader(stream))
    assert parsed[0]["display_name"] == "角色甲"
    assert "score" in parsed[0]["top"]
