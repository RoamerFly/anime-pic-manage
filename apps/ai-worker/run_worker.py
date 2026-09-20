"""Development launcher that works without installing the package first."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def run() -> int:
    from ai_worker.worker import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
