from __future__ import annotations

import logging
import time

from web.postgres_store import load_records, initialize

log = logging.getLogger(__name__)


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

    league_id = record["record_id"]
    db = LeagueDatabase()
    if db.league_exists(league_id):
        try:
            db.load_league(league_id)
            return
        except Exception:
            # Index entry survived but the Feather dataset did not.
            log.warning("League %s has a stale local index; rebuilding it", league_id)
            db.delete_league(league_id)
    _post_with_retry(client, "/api/leagues", record["payload"], (201, 409))

    # Verify that the actual dataset can be loaded after the API reports success.
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
            # A stale model index/checkpoint must not be mistaken for a restored model.
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

    # Restore all datasets first; models depend on their corresponding league.
    for record in leagues:
        try:
            _restore_league(client, record)
        except Exception as exc:
            failures.append(f"league {record['record_id']}: {exc}")
            log.exception("Could not restore league %s", record["record_id"])

    for record in models:
        league_id = record["parent_id"]
        if any(f"league {league_id}:" in failure for failure in failures):
            continue
        try:
            _restore_model(client, record)
        except Exception as exc:
            failures.append(f"model {record['record_id']}: {exc}")
            log.exception("Could not restore model %s", record["record_id"])

    if failures:
        raise RuntimeError("Automatic recovery incomplete: " + "; ".join(failures))

    log.info("Recovery complete: %d league and %d model recipes checked",
             len(leagues), len(models))
