"""Append-only local journal for Pre-Trade Check records (PR: manual persistence).

Stores one JSON object per line (JSON Lines) at ``data/pre_trade_journal.jsonl``.
These are the user's private real trade records — the path is gitignored and must
never be committed. This module is the data-collection foundation only; post-trade
review/analysis is a later PR. Standard library only, no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

JOURNAL_PATH = Path(__file__).resolve().parents[1] / "data" / "pre_trade_journal.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_record(
    *,
    decision: dict,
    evaluation: dict,
    market_context: dict | None = None,
    journal_path: Path | str = JOURNAL_PATH,
    saved_at: str | None = None,
) -> dict:
    """Append one record (decision + evaluation + market_context + saved_at).

    ``market_context`` is the optional manual snapshot of what the user observed
    in the reference sources; None when nothing was recorded. Creates the
    directory/file on first save. Returns the written record.
    """
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "saved_at": saved_at or _utc_now_iso(),
        "decision": decision,
        "evaluation": evaluation,
        "market_context": market_context,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_records(journal_path: Path | str = JOURNAL_PATH) -> list[dict]:
    """Read all journal records. Missing file -> []. Malformed lines are skipped."""
    path = Path(journal_path)
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue  # skip a corrupt line rather than crash
    return records
