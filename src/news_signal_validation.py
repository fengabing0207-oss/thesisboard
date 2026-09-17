"""Explicit point-in-time and chronological audits for News Signal Lab."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


FEATURE_AUDIT_VERSION = "feature-availability.v1"
PREDICTION_AUDIT_VERSION = "walk-forward-leakage.v1"


def audit_feature_availability(dataset: pd.DataFrame) -> dict:
    """Check that every feature snapshot existed by its declared as-of time."""

    required = {
        "ticker",
        "signal_session",
        "signal_close_at",
        "as_of_timestamp",
        "first_seen_at_max",
        "data_vintage",
        "availability_rule_version",
        "target_definition",
        "label_available_at",
        "usable_for_model",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        return _missing_audit(FEATURE_AUDIT_VERSION, missing)

    frame = dataset.copy()
    if frame.empty:
        return _empty_audit(FEATURE_AUDIT_VERSION, "no_rows")
    for column in (
        "signal_close_at",
        "as_of_timestamp",
        "first_seen_at_max",
        "label_available_at",
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)

    usable = frame["usable_for_model"].fillna(False).astype(bool)
    checks = {
        "missing_as_of_timestamp": frame["as_of_timestamp"].isna(),
        "missing_data_vintage": frame["data_vintage"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(""),
        "missing_availability_rule_version": frame["availability_rule_version"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(""),
        "missing_target_definition": frame["target_definition"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(""),
        "feature_observed_at_or_after_as_of": (
            frame["first_seen_at_max"].isna()
            | (frame["first_seen_at_max"] >= frame["as_of_timestamp"])
        ),
        "as_of_differs_from_signal_close": (
            frame["as_of_timestamp"].isna()
            | frame["signal_close_at"].isna()
            | (frame["as_of_timestamp"] != frame["signal_close_at"])
        ),
        "usable_label_not_strictly_forward": usable
        & (
            frame["label_available_at"].isna()
            | (frame["label_available_at"] <= frame["as_of_timestamp"])
        ),
    }
    return _finish_audit(frame, checks, audit_version=FEATURE_AUDIT_VERSION)


def audit_prediction_availability(
    model_predictions: Mapping[str, pd.DataFrame],
) -> dict:
    """Prove that OOS predictions use only prior labels and common rows."""

    if not model_predictions:
        return _empty_audit(PREDICTION_AUDIT_VERSION, "no_model_predictions")
    required = {
        "ticker",
        "signal_session",
        "signal_close_at",
        "as_of_timestamp",
        "first_seen_at_max",
        "data_vintage",
        "availability_rule_version",
        "target_definition",
        "model_name",
        "model_version",
        "horizon_end",
        "label_available_at",
        "target_abnormal_return",
        "train_session_start",
        "train_session_end",
        "train_label_cutoff",
        "test_close_at",
    }
    all_checks: list[dict] = []
    examples: list[dict] = []
    total_rows = 0
    key_sets: dict[str, set[tuple]] = {}
    normalized_frames: dict[str, pd.DataFrame] = {}
    missing_by_model: dict[str, list[str]] = {}

    for name, source in model_predictions.items():
        missing = sorted(required - set(source.columns))
        if missing:
            missing_by_model[str(name)] = missing
            continue
        frame = source.copy()
        total_rows += len(frame)
        frame["signal_session"] = pd.to_datetime(
            frame["signal_session"], errors="coerce"
        ).dt.normalize()
        frame["train_session_start"] = pd.to_datetime(
            frame["train_session_start"], errors="coerce"
        ).dt.normalize()
        frame["train_session_end"] = pd.to_datetime(
            frame["train_session_end"], errors="coerce"
        ).dt.normalize()
        frame["horizon_end"] = pd.to_datetime(
            frame["horizon_end"], errors="coerce"
        ).dt.normalize()
        for column in (
            "signal_close_at",
            "as_of_timestamp",
            "first_seen_at_max",
            "label_available_at",
            "train_label_cutoff",
            "test_close_at",
        ):
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        key_sets[str(name)] = set(
            frame[["ticker", "signal_session"]].itertuples(index=False, name=None)
        )
        normalized_frames[str(name)] = frame
        checks = {
            "duplicate_ticker_session": frame.duplicated(
                ["ticker", "signal_session"], keep=False
            ),
            "missing_data_vintage": frame["data_vintage"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(""),
            "missing_model_version": frame["model_version"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(""),
            "model_name_mismatch": frame["model_name"]
            .fillna("")
            .astype(str)
            .ne(str(name)),
            "missing_availability_rule_version": frame["availability_rule_version"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(""),
            "missing_target_definition": frame["target_definition"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(""),
            "missing_horizon_end": frame["horizon_end"].isna(),
            "missing_target_abnormal_return": frame["target_abnormal_return"].isna(),
            "feature_observed_at_or_after_as_of": (
                frame["first_seen_at_max"].isna()
                | (frame["first_seen_at_max"] >= frame["as_of_timestamp"])
            ),
            "as_of_differs_from_signal_close": (
                frame["as_of_timestamp"].isna()
                | frame["signal_close_at"].isna()
                | (frame["as_of_timestamp"] != frame["signal_close_at"])
            ),
            "test_close_differs_from_as_of": (
                frame["test_close_at"].isna()
                | frame["as_of_timestamp"].isna()
                | (frame["test_close_at"] != frame["as_of_timestamp"])
            ),
            "training_label_cutoff_not_before_test": (
                frame["train_label_cutoff"].isna()
                | frame["test_close_at"].isna()
                | (frame["train_label_cutoff"] >= frame["test_close_at"])
            ),
            "training_window_not_strictly_prior": (
                frame["train_session_start"].isna()
                | frame["train_session_end"].isna()
                | (frame["train_session_start"] > frame["train_session_end"])
                | (frame["train_session_end"] >= frame["signal_session"])
            ),
            "target_label_not_strictly_forward": (
                frame["label_available_at"].isna()
                | frame["test_close_at"].isna()
                | (frame["label_available_at"] <= frame["test_close_at"])
            ),
        }
        for check, mask in checks.items():
            count = int(mask.fillna(True).sum())
            all_checks.append({"model": str(name), "check": check, "violation_count": count})
            if count:
                sample = frame.loc[mask.fillna(True), ["ticker", "signal_session"]].head(3)
                examples.extend(
                    {
                        "model": str(name),
                        "check": check,
                        "ticker": row["ticker"],
                        "signal_session": row["signal_session"],
                    }
                    for row in sample.to_dict("records")
                )

    if missing_by_model:
        return {
            "status": "missing_required_columns",
            "audit_version": PREDICTION_AUDIT_VERSION,
            "row_count": int(total_rows),
            "model_count": int(len(model_predictions)),
            "missing_by_model": missing_by_model,
            "checks": pd.DataFrame(all_checks),
            "violation_examples": pd.DataFrame(examples),
        }

    if key_sets:
        reference_name = next(iter(key_sets))
        reference = key_sets[reference_name]
        reference_contracts = _prediction_contracts(normalized_frames[reference_name])
        for name, keys in key_sets.items():
            mismatch = len(reference.symmetric_difference(keys))
            all_checks.append(
                {"model": name, "check": "common_oos_keys", "violation_count": int(mismatch)}
            )
            contracts = _prediction_contracts(normalized_frames[name])
            contract_mismatch = sum(
                contracts.get(key) != reference_contracts.get(key)
                for key in reference | keys
            )
            all_checks.append(
                {
                    "model": name,
                    "check": "common_oos_data_contract",
                    "violation_count": int(contract_mismatch),
                }
            )

    check_table = pd.DataFrame(all_checks)
    violation_count = int(check_table["violation_count"].sum()) if not check_table.empty else 0
    return {
        "status": "ok" if violation_count == 0 else "failed",
        "audit_version": PREDICTION_AUDIT_VERSION,
        "row_count": int(total_rows),
        "model_count": int(len(model_predictions)),
        "violation_count": violation_count,
        "checks": check_table,
        "violation_examples": pd.DataFrame(examples),
    }


def _finish_audit(
    frame: pd.DataFrame,
    checks: dict[str, pd.Series],
    *,
    audit_version: str,
) -> dict:
    rows = []
    examples = []
    for check, mask in checks.items():
        mask = mask.fillna(True)
        count = int(mask.sum())
        rows.append({"check": check, "violation_count": count})
        if count:
            sample = frame.loc[mask, ["ticker", "signal_session"]].head(3)
            examples.extend(
                {"check": check, **row} for row in sample.to_dict("records")
            )
    violation_count = int(sum(row["violation_count"] for row in rows))
    return {
        "status": "ok" if violation_count == 0 else "failed",
        "audit_version": audit_version,
        "row_count": int(len(frame)),
        "violation_count": violation_count,
        "checks": pd.DataFrame(rows),
        "violation_examples": pd.DataFrame(examples),
    }


def _prediction_contracts(frame: pd.DataFrame) -> dict[tuple, tuple]:
    result = {}
    for row in frame.itertuples(index=False):
        key = (row.ticker, row.signal_session)
        result[key] = (
            row.data_vintage,
            row.as_of_timestamp,
            row.label_available_at,
            row.horizon_end,
            row.availability_rule_version,
            row.target_definition,
            float(row.target_abnormal_return),
        )
    return result


def _missing_audit(audit_version: str, missing: list[str]) -> dict:
    return {
        "status": "missing_required_columns",
        "audit_version": audit_version,
        "row_count": 0,
        "violation_count": None,
        "missing_columns": missing,
        "checks": pd.DataFrame(),
        "violation_examples": pd.DataFrame(),
    }


def _empty_audit(audit_version: str, status: str) -> dict:
    return {
        "status": status,
        "audit_version": audit_version,
        "row_count": 0,
        "violation_count": 0,
        "checks": pd.DataFrame(),
        "violation_examples": pd.DataFrame(),
    }
