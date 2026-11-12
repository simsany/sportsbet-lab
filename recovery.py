from __future__ import annotations

import logging
import time
from typing import Any

from web.postgres_store import load_records, initialize, save_record

log = logging.getLogger(__name__)

# Exact feature names exposed by the original ProphitBet StatisticsEngine.
# The fallback keeps recovery safe if an import fails during an early boot.
_FALLBACK_STATS = {
    "HW", "HL", "HGF", "HGA", "HGD", "HWGD", "HLGD", "HW%", "HL%",
    "AW", "AL", "AGF", "AGA", "AGD", "AWGD", "ALGD", "AW%", "AL%",
    "HAGF", "HAGA", "HAGD", "HAWGD", "HALGD", "HAGFT", "HAGAT",
    "HWT", "HLT", "HGFT", "HGAT", "HWT%", "HLT%",
    "AWT", "ALT", "AGFT", "AGAT", "AWT%", "ALT%",
}
_DEFAULT_STATS = ["HGF", "HGA", "AGF", "AGA"]


def _supported_stats() -> set[str]:
    try:
        from src.core.preprocessing.statistics import StatisticsEngine

        values = StatisticsEngine().stats_columns
        if callable(values):
            values = values()
        supported = {str(value).strip() for value in values}
        if supported:
            return supported
    except Exception:
        log.exception("Could not read StatisticsEngine feature names; using verified fallback")
    return set(_FALLBACK_STATS)


def _as_stat_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Also handle old recipes that accidentally stored a comma-separated string.
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_legacy_league_recipe(record: dict) -> dict:
    """Validate and persist a league recipe before any restore/API operation."""
    payload = dict(record.get("payload") or {})
    if not payload:
        raise ValueError("Recovery recipe has an empty payload")

    supported = _supported_stats()
    requested = _as_stat_list(payload.get("stats_columns"))
    valid = list(dict.fromkeys(name for name in requested if name in supported))
    invalid = [name for name in requested if name not in supported]

    # Preserve every valid selected feature. If the recipe only contains raw
    # match columns (e.g. HS/AS/HST/AST), use valid original-engine features.
    normalized = valid if valid else list(_DEFAULT_STATS)
    if not set(normalized).issubset(supported):
        raise RuntimeError("Internal recovery fallback contains unsupported statistics")

    if normalized != requested:
        record_id = record.get("record_id")
        if not record_id:
            raise ValueError("Recovery league recipe is missing record_id")
        payload["stats_columns"] = normalized
        # Persist first. If the process crashes later, the next boot reads the
        # repaired recipe instead of retrying the same invalid HS/AS values.
        save_record("league", record_id, payload)
        log.warning(
            "Repaired persisted league recipe %s: invalid stats=%s; using=%s",
            record_id, invalid, normalized,
        )
    else:
        payload["stats_columns"] = normalized

    return payload


def _post_with_retry(client, path: str, payload: dict, expected: tuple[int, ...]):
    last = None
    for attempt in range(1, 4):
        last = client.post(path, json=payload)
        if last.status_code in expected:
            return last
        if last.status_code < 500:
            break
        time.sleep(attempt * 5)
    detail = last.get_json(silent=True) if last is not None else "no response"
    raise RuntimeError(
        f"{path} recovery failed ({last.status_code if last else '?'}): {detail}"
    )


def _restore_league(client, record: dict) -> None:
    from src.core.database.league import LeagueDatabase

    # IMPORTANT: normalize and persist BEFORE checking local storage. This also
    # repairs old Postgres recipes on boots where a local league already exists.
    payload = _normalize_legacy_league_recipe(record)
    league_id = payload.get("league_id") or record.get("record_id")
    if not league_id:
        raise ValueError("League recovery recipe is missing league_id")

    db = LeagueDatabase()
    if db.league_exists(league_id):
        try:
            db.load_league(league_id)
            return
        except Exception:
            log.warning("League %s has a stale local index; rebuilding it", league_id)
            db.delete_league(league_id)

    _post_with_retry(client, "/api/leagues", payload, (201, 409))

    db = LeagueDatabase()
    if not db.league_exists(league_id):
        raise RuntimeError("League API succeeded but league index is missing")
    db.load_league(league_id)


def _restore_model(client, record: dict) -> None:
    from src.core.database.model import ModelDatabase

    league_id = record["parent_id"]
    payload = dict(record["payload"])
    model_id = payload["model_id"]
    db = ModelDatabase(league_id)

    if db.model_exists(model_id):
        try:
            db.load_model(model_id)
            return
        except Exception:
            log.warning("Model %s/%s has a stale checkpoint; retraining", league_id, model_id)
            db.delete_model(model_id)

    response = _post_with_retry(
        client, f"/api/leagues/{league_id}/train", payload, (200, 409)
    )
    db = ModelDatabase(league_id)
    if not db.model_exists(model_id):
        raise RuntimeError(
            f"Training endpoint returned {response.status_code} but model is absent"
        )
    db.load_model(model_id)


def recover_if_needed():
    from web.app import app

    initialize()
    leagues = load_records("league")
    models = load_records("model")
    client = app.test_client()
    failures = []

    for record in leagues:
        try:
            _restore_league(client, record)
        except Exception as exc:
            failures.append(f"league {record.get('record_id', '?')}: {exc}")
            log.exception("Could not restore league %s", record.get("record_id", "?"))

    for record in models:
        league_id = record.get("parent_id")
        if any(f"league {league_id}:" in failure for failure in failures):
            continue
        try:
            _restore_model(client, record)
        except Exception as exc:
            failures.append(f"model {record.get('record_id', '?')}: {exc}")
            log.exception("Could not restore model %s", record.get("record_id", "?"))

    if failures:
        raise RuntimeError("Automatic recovery incomplete: " + "; ".join(failures))

    log.info("Recovery complete: %d league and %d model recipes checked", len(leagues), len(models))
