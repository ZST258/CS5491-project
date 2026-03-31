from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_summary_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" in payload and isinstance(payload["summary"], dict):
        return payload
    return {"summary": payload, "metadata": {}, "run_name": Path(path).stem.replace("_summary", "")}


def summary_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("summary", payload)
