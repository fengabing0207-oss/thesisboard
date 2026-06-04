import json

from src.journal import append_record, load_records


def _record_inputs():
    decision = {"ticker": "NVDA", "planned_action": "buy", "status": "pending"}
    evaluation = {"verdict": "wait", "active_risk_flags": ["no_invalidation_rule"], "reasons": ["define exit"]}
    return decision, evaluation


def test_append_writes_one_valid_json_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    decision, evaluation = _record_inputs()
    append_record(decision=decision, evaluation=evaluation, journal_path=path, saved_at="2026-06-04T00:00:00+00:00")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["decision"]["ticker"] == "NVDA"
    assert parsed["evaluation"]["verdict"] == "wait"
    assert parsed["saved_at"] == "2026-06-04T00:00:00+00:00"


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "journal.jsonl"
    decision, evaluation = _record_inputs()
    written = append_record(decision=decision, evaluation=evaluation, journal_path=path, saved_at="2026-06-04T12:00:00+00:00")

    records = load_records(path)
    assert len(records) == 1
    assert records[0] == written
    assert records[0]["decision"] == decision
    assert records[0]["evaluation"] == evaluation
    assert records[0]["saved_at"] == "2026-06-04T12:00:00+00:00"


def test_append_is_append_only(tmp_path):
    path = tmp_path / "journal.jsonl"
    decision, evaluation = _record_inputs()
    append_record(decision=decision, evaluation=evaluation, journal_path=path)
    append_record(decision={**decision, "ticker": "MSFT"}, evaluation=evaluation, journal_path=path)

    records = load_records(path)
    assert [r["decision"]["ticker"] for r in records] == ["NVDA", "MSFT"]


def test_append_auto_fills_saved_at(tmp_path):
    path = tmp_path / "journal.jsonl"
    decision, evaluation = _record_inputs()
    written = append_record(decision=decision, evaluation=evaluation, journal_path=path)
    assert written["saved_at"]  # non-empty ISO timestamp


def test_missing_file_returns_empty(tmp_path):
    assert load_records(tmp_path / "does_not_exist.jsonl") == []


def test_empty_file_returns_empty(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_records(path) == []


def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "journal.jsonl"
    good = json.dumps({"saved_at": "t", "decision": {"ticker": "AAPL"}, "evaluation": {"verdict": "proceed"}})
    path.write_text(good + "\n" + "{not valid json" + "\n" + "\n" + good + "\n", encoding="utf-8")

    records = load_records(path)
    assert len(records) == 2  # the two good lines; malformed and blank skipped
    assert all(r["decision"]["ticker"] == "AAPL" for r in records)


def test_creates_directory_on_first_save(tmp_path):
    path = tmp_path / "nested" / "dir" / "journal.jsonl"
    decision, evaluation = _record_inputs()
    append_record(decision=decision, evaluation=evaluation, journal_path=path)
    assert path.exists()
    assert len(load_records(path)) == 1


def test_market_context_round_trips_with_notes(tmp_path):
    path = tmp_path / "journal.jsonl"
    decision, evaluation = _record_inputs()
    context = {
        "market_regime_note": "VIX elevated, risk-off",
        "sector_theme_note": "",
        "ticker_context_note": "extended above 50dma",
        "event_risk_note": "earnings in 3 days",
        "options_flow_note": "",
        "source_checked_at": "2026-06-04T00:00:00+00:00",
    }
    append_record(decision=decision, evaluation=evaluation, market_context=context, journal_path=path)

    records = load_records(path)
    assert records[0]["market_context"] == context
    assert records[0]["market_context"]["market_regime_note"] == "VIX elevated, risk-off"


def test_market_context_defaults_to_none(tmp_path):
    path = tmp_path / "journal.jsonl"
    decision, evaluation = _record_inputs()
    append_record(decision=decision, evaluation=evaluation, journal_path=path)
    assert load_records(path)[0]["market_context"] is None


def test_empty_market_context_notes_still_save(tmp_path):
    path = tmp_path / "journal.jsonl"
    decision, evaluation = _record_inputs()
    context = {field: "" for field in (
        "market_regime_note",
        "sector_theme_note",
        "ticker_context_note",
        "event_risk_note",
        "options_flow_note",
    )}
    context["source_checked_at"] = "2026-06-04T00:00:00+00:00"
    append_record(decision=decision, evaluation=evaluation, market_context=context, journal_path=path)

    records = load_records(path)
    assert records[0]["market_context"] == context  # empty strings preserved, no crash
