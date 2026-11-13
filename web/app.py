from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.core.database.league import LeagueDatabase
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.models.classifiers import LogisticRegressor, RandomForest
from src.core.models.model import ModelConfig
from src.core.models.trainer import Trainer
from src.core.preprocessing.filters import filter_league_odds
from src.core.preprocessing.statistics import StatisticsEngine
from src.core.preprocessing.utils.targets import TargetType
from src.core.preprocessing.utils.inputs import construct_inputs_by_match
from web.postgres_store import save_record, load_records, initialize as initialize_postgres

ROOT = Path("/opt/prophitbet")
STORAGE = ROOT / "storage" / "leagues"
_storage_lock = threading.RLock()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False


@contextmanager
def persistent_storage():
    """Disposable local cache; the checked-in recovery manifest rebuilds it."""
    with _storage_lock:
        STORAGE.mkdir(parents=True, exist_ok=True)
        yield


def _league_db():
    return LeagueDatabase()


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value


def _available_leagues():
    db = _league_db()
    return [
        {"country": x.country, "name": x.name, "min_year": x.min_year,
         "category": x.category, "url": x.url, "fixture_url": x.fixture_url}
        for x in db.leagues
    ]


@app.get("/")
def index():
    with persistent_storage():
        db = _league_db()
        leagues = []
        for league_id in db.get_league_ids():
            cfg = db.index[league_id]
            try:
                rows = len(db.load_league(league_id))
            except Exception:
                rows = 0
            leagues.append({"id": league_id, "country": cfg.league.country,
                            "name": cfg.league.name, "rows": rows})
        return render_template("index.html", leagues=leagues,
                               catalogue=_available_leagues(),
                               persistent=False)


@app.get("/health")
def health():
    try:
        initialize_postgres()
        postgres = "connected"
    except Exception as exc:
        return jsonify({"status": "error", "postgres": str(exc)}), 503
    return jsonify({"status": "ok", "local_storage": "disposable-cache",
                    "recovery_metadata": postgres,
                    "original_commit": "49fa5ff14783af5dbbac7919ec743ab41628fcac"})


@app.get("/api/catalogue")
def catalogue():
    return jsonify(_available_leagues())


@app.get("/api/leagues")
def leagues():
    with persistent_storage():
        db = _league_db()
        result = []
        for league_id in db.get_league_ids():
            cfg = db.index[league_id]
            result.append({"id": league_id, "country": cfg.league.country,
                           "name": cfg.league.name, "start_year": cfg.start_year,
                           "match_history_window": cfg.match_history_window})
        return jsonify(result)


@app.post("/api/leagues")
def create_league():
    data = request.get_json(force=True)
    required = ["country", "name", "start_year", "stats_columns"]
    missing = [x for x in required if not data.get(x)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    # Validate and normalize before saving a recovery recipe or downloading.
    # HS/AS/HST/AST are raw match columns, not StatisticsEngine feature keys.
    raw_stats = data.get("stats_columns")
    if isinstance(raw_stats, str):
        requested_stats = [part.strip() for part in raw_stats.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw_stats, (list, tuple)):
        requested_stats = [str(part).strip() for part in raw_stats if str(part).strip()]
    else:
        return jsonify({"error": "stats_columns must be a list of ProphitBet feature names."}), 400

    if not requested_stats:
        return jsonify({"error": "Select at least one supported ProphitBet feature."}), 400

    # Preserve order while removing duplicates.
    requested_stats = list(dict.fromkeys(requested_stats))
    supported_stats = {str(name).strip() for name in StatisticsEngine().stats_columns}
    invalid_stats = [name for name in requested_stats if name not in supported_stats]
    if invalid_stats:
        return jsonify({
            "error": (
                f"Unsupported statistics: {', '.join(invalid_stats)}. "
                "HS, AS, HST and AST are raw match columns, not model features. "
                f"Supported values: {', '.join(sorted(supported_stats))}"
            ),
            "invalid_stats": invalid_stats,
            "supported_stats": sorted(supported_stats),
        }), 400

    data["stats_columns"] = requested_stats

    with persistent_storage():
        db = _league_db()
        selected = next((x for x in db.leagues
                         if x.country == data["country"] and x.name == data["name"]), None)
        if selected is None:
            return jsonify({"error": "League not found in original ProphitBet catalogue"}), 404
        league_id = data.get("league_id") or f'{selected.country}-{selected.name}'
        if db.league_exists(league_id):
            return jsonify({"error": "League id already exists"}), 409
        cfg = LeagueConfig(
            league=selected, league_id=league_id,
            start_year=int(data["start_year"]),
            stats_columns=list(data["stats_columns"]),
            match_history_window=int(data.get("match_history_window", 3)),
            goal_diff_margin=int(data.get("goal_diff_margin", 2)),
            aggregation=data.get("aggregation", "sum"),
        )
        recovery_config = {
            "country": selected.country,
            "name": selected.name,
            "league_id": league_id,
            "start_year": int(data["start_year"]),
            "stats_columns": list(data["stats_columns"]),
            "match_history_window": int(data.get("match_history_window", 3)),
            "goal_diff_margin": int(data.get("goal_diff_margin", 2)),
            "aggregation": data.get("aggregation", "sum"),
        }
        save_record("league", league_id, recovery_config)
        frame = db.download_league(cfg)
        if frame is None or frame.empty:
            return jsonify({"error": "Original downloader returned no historical matches"}), 422
        frame = filter_league_odds(df=frame, odd_1_mask_range=None,
                                   odd_x_mask_range=None, odd_2_mask_range=None,
                                   validate_dataframe=False)
        if frame is None or frame.empty:
            return jsonify({"error": "No matches remain after original odds filtering"}), 422
        db.save_league(frame, cfg)
        return jsonify({"id": league_id, "matches": len(frame),
                        "recovery_config_saved": True}), 201


@app.get("/api/leagues/<league_id>/teams")
def teams(league_id):
    with persistent_storage():
        db = _league_db()
        if not db.league_exists(league_id):
            return jsonify({"error": "Unknown league"}), 404
        frame = db.load_league(league_id)
        values = sorted(set(frame["Home"].dropna().astype(str)) |
                        set(frame["Away"].dropna().astype(str)))
        return jsonify(values)


@app.get("/api/leagues/<league_id>/models")
def models(league_id):
    with persistent_storage():
        db = ModelDatabase(league_id)
        result = []
        for model_id, cfg in db.index.get(league_id, {}).items():
            result.append({"id": model_id, "target": str(cfg.classifier_config.get("target_type")),
                           "train_samples": cfg.num_train_samples,
                           "eval_samples": cfg.num_eval_samples})
        return jsonify(result)


@app.post("/api/leagues/<league_id>/train")
def train(league_id):
    data = request.get_json(force=True)
    model_id = data.get("model_id", "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    with persistent_storage():
        ldb = _league_db()
        if not ldb.league_exists(league_id):
            return jsonify({"error": "Unknown league"}), 404
        frame = ldb.load_league(league_id).dropna().reset_index(drop=True)
        if len(frame) < 100:
            return jsonify({"error": f"Not enough complete historical rows: {len(frame)}"}), 422
        target = TargetType(data.get("target", "Result"))
        classifier_name = data.get("classifier", "LogisticRegressor")
        if classifier_name == "LogisticRegressor":
            classifier = LogisticRegressor(target_type=target, penalty=data.get("penalty"))
        elif classifier_name == "RandomForest":
            classifier = RandomForest(target_type=target,
                                      n_estimators=int(data.get("n_estimators", 200)),
                                      max_depth=data.get("max_depth"))
        else:
            return jsonify({"error": "Supported web training classifiers: LogisticRegressor, RandomForest"}), 400
        recovery_config = dict(data)
        recovery_config.update({"model_id": model_id, "target": target.value,
                                "classifier": classifier_name})
        recovery_config.pop("league_id", None)
        save_record("model", f"{league_id}:{model_id}", recovery_config, parent_id=league_id)

        trainer = Trainer()
        classifier, report = trainer.train(classifier, frame,
                                           test_size=float(data.get("test_size", 0.2)),
                                           seed=int(data.get("seed", 0)))
        report_records = json.loads(report.to_json(orient="records", date_format="iso"))
        test_size = float(data.get("test_size", 0.2))
        cfg = ModelConfig(
            league_id=league_id, model_id=model_id,
            classifier_config=classifier.model_config,
            num_train_samples=len(frame) - int(len(frame) * test_size),
            num_eval_samples=int(len(frame) * test_size),
            train_report=report,
        )
        ModelDatabase(league_id).save_model(cfg, classifier)
        return jsonify({"model_id": model_id, "target": target.value,
                        "rows": len(frame), "report": report_records,
                        "recovery_config_saved": True})


@app.post("/api/leagues/<league_id>/predict")
def predict(league_id):
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    home, away = data.get("home"), data.get("away")
    try:
        odds = [float(data[x]) for x in ("odd_1", "odd_x", "odd_2")]
        if not all(np.isfinite(x) and x > 1 for x in odds):
            raise ValueError()
    except Exception:
        return jsonify({"error": "Odds must be valid decimal odds greater than 1"}), 400
    with persistent_storage():
        ldb = _league_db()
        if not ldb.league_exists(league_id):
            return jsonify({"error": "Unknown league"}), 404
        mdb = ModelDatabase(league_id)
        if not mdb.model_exists(model_id):
            return jsonify({"error": "Unknown model"}), 404
        history = ldb.load_league(league_id)
        match = pd.Series([home, away, *odds], index=["Home", "Away", "1", "X", "2"])
        row = construct_inputs_by_match(league_df=history, match_df=match)
        if row is None:
            return jsonify({"error": "No historical match found for this home/away pairing"}), 422
        latest = history.iloc[0]
        row = row.copy()
        row["Date"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        row["Season"] = latest["Season"]
        row["Week"] = int(pd.to_numeric(history["Week"], errors="coerce").max()) + 1
        p = [1/x for x in odds]
        row["1X"] = max(round(1/(p[0]+p[1]), 2), 1.0)
        row["12"] = max(round(1/(p[0]+p[2]), 2), 1.0)
        row["X2"] = max(round(1/(p[1]+p[2]), 2), 1.0)
        row["HG"], row["AG"] = 0, 0
        row["Result"], row["Result-U/O"] = "D", "U"
        prediction_frame = row.to_frame().T.reset_index(drop=True)
        model, cfg = mdb.load_model(model_id)
        probs = np.asarray(model.predict_proba(prediction_frame), dtype=float)[0]
        target = cfg.classifier_config["target_type"]
        target = target if isinstance(target, TargetType) else TargetType(target)
        labels = list(target.get_mapper().keys())
        if len(probs) != len(labels):
            return jsonify({"error": "Model probability dimensions do not match target labels"}), 500
        return jsonify({"model_id": model_id, "target": target.value,
                        "prediction": labels[int(np.argmax(probs))],
                        "probabilities": {label: float(prob) for label, prob in zip(labels, probs)}})


@app.errorhandler(Exception)
def unhandled(error):
    # Do not convert normal HTTP errors (e.g. Flask routing 404) into 500 responses.
    if isinstance(error, HTTPException):
        return jsonify({"error": error.description}), error.code
    app.logger.exception("Unhandled web request error")
    return jsonify({"error": str(error)}), 500
