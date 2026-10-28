
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
RUN_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}

SPORTS = {
    "football": {"label": "Football / soccer", "league_options": [
        "England", "Scotland", "Germany", "Italy", "Spain", "France", "Netherlands",
        "Belgium", "Portugal", "Turkey", "Greece", "Argentina", "Austria", "Brazil",
        "China", "Denmark", "Finland", "Ireland", "Japan", "Mexico", "Norway",
        "Poland", "Romania", "Russia", "Sweden", "Switzerland", "USA"
    ]},
    "nba": {"label": "NBA", "league_options": ["NBA"]},
    "euroleague": {"label": "EuroLeague", "league_options": ["Euroleague"]},
}
OUTCOME_MARKETS = ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]


def _plain(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _sportsbet_sources(sport, leagues, years, source_mode):
    from sportsbet.dataloaders import DataLoader
    from sportsbet.sources import (
        FootballDataOdds, FootballDataStats, NBAStats, EuroLeagueStats, OddsApi
    )

    use_odds_api = source_mode == "odds_api"
    if use_odds_api and not os.environ.get("ODDS_API_KEY"):
        raise ValueError(
            "Odds API was selected, but ODDS_API_KEY is not configured in "
            "Heroku Config Vars. No picks were generated."
        )

    if sport == "football":
        stats = FootballDataStats()
        odds = (
            OddsApi(key_env="ODDS_API_KEY", markets=["h2h", "totals"], regions=["eu"])
            if use_odds_api else FootballDataOdds()
        )
        param_grid = {"league": leagues, "division": [1], "year": years}
        default_odds_type = "odds_api" if use_odds_api else "market_maximum"
    elif sport == "nba":
        stats = NBAStats()
        if not use_odds_api:
            raise ValueError(
                "NBA predictions require a real odds feed. Configure ODDS_API_KEY "
                "in Heroku and select Odds API."
            )
        odds = OddsApi(key_env="ODDS_API_KEY", markets=["h2h"], regions=["eu"])
        param_grid = {"league": ["NBA"], "division": [1], "year": years}
        default_odds_type = "odds_api"
    elif sport == "euroleague":
        stats = EuroLeagueStats()
        if not use_odds_api:
            raise ValueError(
                "EuroLeague predictions require a real odds feed. Configure "
                "ODDS_API_KEY in Heroku and select Odds API."
            )
        odds = OddsApi(key_env="ODDS_API_KEY", markets=["h2h"], regions=["eu"])
        param_grid = {"league": ["Euroleague"], "division": [1], "year": years}
        default_odds_type = "odds_api"
    else:
        raise ValueError("Unsupported sport selected.")

    loader = DataLoader(param_grid=param_grid, stats=stats, odds=odds)
    return loader, default_odds_type


def _model_pipeline(X):
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical = list(X.select_dtypes(
        include=["object", "category", "string"]
    ).columns)
    numeric = list(X.select_dtypes(
        include=[np.number, "bool"]
    ).columns)

    if not categorical and not numeric:
        raise ValueError("The selected source produced no usable model features.")

    transformers = []
    if categorical:
        transformers.append((
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical,
        ))
    if numeric:
        transformers.append((SimpleImputer(strategy="median"), numeric))

    preprocessing = make_column_transformer(
        *transformers, remainder="drop", sparse_threshold=0.3
    )
    return make_pipeline(
        preprocessing,
        MultiOutputClassifier(
            LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=7,
                class_weight="balanced",
            )
        ),
    )


def _choose_markets(Y, O):
    from sportsbet.evaluation import derive_market_base, find_latest_odds_column

    y_bases = list(dict.fromkeys(
        derive_market_base(str(c)) for c in Y.columns
    ))
    o_bases = []
    for col in O.columns:
        parts = str(col).split("__")
        if len(parts) == 4:
            o_bases.append(parts[1])

    common = [m for m in OUTCOME_MARKETS if m in y_bases and m in o_bases]
    usable = []
    for market in common:
        odds_col = find_latest_odds_column(list(O.columns), market)
        if odds_col is not None:
            vals = pd.to_numeric(O[odds_col], errors="coerce")
            if vals.gt(1.0).any():
                usable.append(market)

    if not usable:
        raise ValueError(
            "The chosen data source returned no overlapping outcome labels "
            "and valid odds for the selected markets."
        )
    return usable


def _backtest_result(bettor, X, Y, O, folds):
    from sklearn.model_selection import TimeSeriesSplit
    from sportsbet.evaluation import backtest

    n_splits = max(2, min(5, int(folds)))
    if len(X) < max(40, n_splits + 5):
        raise ValueError(
            f"Not enough historical rows ({len(X)}) for a "
            f"{n_splits}-fold walk-forward backtest."
        )

    result = backtest(
        bettor, X, Y, O,
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=1,
        verbose=0,
    )
    if result is None or result.empty:
        raise ValueError("The library returned no backtest metrics.")

    frame = result.reset_index()
    return {
        "rows": int(len(X)),
        "columns": [str(c) for c in frame.columns],
        "folds": [
            {str(k): _plain(v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ],
        "note": (
            "Walk-forward historical evaluation from sports-betting. "
            "Results are historical and do not guarantee future performance."
        ),
    }


def _run(payload, only_backtest=False):
    sport = str(payload.get("sport", "football")).lower()
    if sport not in SPORTS:
        raise ValueError("Choose football, NBA or EuroLeague.")

    source_mode = str(payload.get("source_mode", "free")).lower()
    years = sorted({
        int(y) for y in (payload.get("years") or [2023, 2024, 2025, 2026])
        if 2000 <= int(y) <= 2035
    })
    if not years:
        raise ValueError("Select at least one valid season year.")
    if len(years) > 6:
        raise ValueError("Choose at most six season years per run.")

    leagues = payload.get("leagues") or (
        ["England"] if sport == "football"
        else [SPORTS[sport]["league_options"][0]]
    )
    if not isinstance(leagues, list):
        leagues = [leagues]
    leagues = [
        str(v) for v in leagues if str(v) in SPORTS[sport]["league_options"]
    ]
    if not leagues:
        raise ValueError("Select at least one supported league for this sport.")

    from sportsbet.evaluation import ClassifierBettor

    loader, odds_type = _sportsbet_sources(
        sport, leagues, years, source_mode
    )
    X_train, Y_train, O_train = loader.extract_train_data(
        odds_type=odds_type,
        input_event_status="preplay",
    )

    if X_train.empty:
        raise ValueError("The selected data source returned no historical training rows.")
    if not (len(X_train) == len(Y_train) == len(O_train)):
        raise ValueError(
            "The data loader returned misaligned training data; refusing "
            "to train on mismatched rows."
        )

    if not only_backtest:
        X_fix, _, O_fix = loader.extract_fixtures_data()
        if X_fix.empty:
            raise ValueError(
                "The selected data source returned no upcoming fixtures "
                "for these leagues."
            )
        if len(X_fix) != len(O_fix):
            raise ValueError(
                "The data loader returned misaligned fixture and odds rows."
            )

    markets = _choose_markets(Y_train, O_train)
    classifier = _model_pipeline(X_train)
    bettor = ClassifierBettor(
        classifier=classifier,
        betting_markets=markets,
        init_cash=10000.0,
        stake=50.0,
    )

    bt = _backtest_result(
        bettor, X_train, Y_train, O_train,
        payload.get("backtest_folds", 3),
    )

    if only_backtest:
        return {
            "sport": sport,
            "data_source": _source_label(sport, source_mode),
            "backtest": bt["folds"],
            "backtest_columns": bt["columns"],
            "rows": bt["rows"],
            "note": bt["note"],
        }

    bettor.fit(X_train, Y_train, O_train)
    probabilities = np.asarray(bettor.predict_proba(X_fix), dtype=float)
    bets = np.asarray(bettor.bet(X_fix, O_fix), dtype=bool)

    if (
        probabilities.ndim != 2
        or probabilities.shape != bets.shape
        or probabilities.shape[1] != len(markets)
    ):
        raise ValueError(
            f"Unexpected model output shapes: probabilities={probabilities.shape}, "
            f"bets={bets.shape}. No picks were emitted."
        )

    from sportsbet.evaluation import find_latest_odds_column
    odds_columns = {
        m: find_latest_odds_column(list(O_fix.columns), m, provider=odds_type)
        for m in markets
    }

    min_probability = float(payload.get("min_probability", 0.60))
    min_confidence = float(payload.get("min_confidence", 0.0)) / 100.0
    min_edge = float(payload.get("min_edge", 2.0)) / 100.0
    min_ev = float(payload.get("min_ev", 0.0)) / 100.0
    min_odds = float(payload.get("min_odds", 1.30))
    max_odds = float(payload.get("max_odds", 8.0))
    max_picks = max(1, min(50, int(payload.get("max_picks", 20))))

    groups = []
    one_x_two = [
        m for m in ["home_win", "draw", "away_win"] if m in markets
    ]
    if len(one_x_two) > 1:
        groups.append(one_x_two)

    pair = [m for m in ["over_2.5", "under_2.5"] if m in markets]
    if len(pair) > 1:
        groups.append(pair)

    for market in markets:
        if not any(market in group for group in groups):
            groups.append([market])

    picks = []
    for i in range(len(X_fix)):
        row = X_fix.iloc[i]
        for j, market in enumerate(markets):
            if not bets[i, j]:
                continue

            p = float(probabilities[i, j])
            odds_col = odds_columns.get(market)
            if not odds_col:
                continue
            try:
                odds = float(O_fix.iloc[i][odds_col])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(odds) or odds <= 1:
                continue

            group = next((g for g in groups if market in g), [market])
            group_probs = sorted(
                [float(probabilities[i, markets.index(m)]) for m in group],
                reverse=True,
            )
            confidence = (
                group_probs[0] - group_probs[1]
                if len(group_probs) > 1 else group_probs[0]
            )
            implied = 1.0 / odds
            edge = p - implied
            ev = p * odds - 1.0

            if (
                p < min_probability
                or confidence < min_confidence
                or edge < min_edge
                or ev < min_ev
                or odds < min_odds
                or odds > max_odds
            ):
                continue

            market_label = {
                "home_win": "Home win",
                "draw": "Draw",
                "away_win": "Away win",
            }.get(market, market.replace("_", " ").title())

            picks.append({
                "date": _plain(X_fix.index[i]),
                "league": _plain(row.get("league")),
                "home_team": _plain(row.get("home_team")),
                "away_team": _plain(row.get("away_team")),
                "market": market_label,
                "market_key": market,
                "probability": round(p, 4),
                "confidence": round(confidence, 4),
                "odds": round(odds, 3),
                "implied_probability": round(implied, 4),
                "edge": round(edge, 4),
                "ev": round(ev, 4),
                "model": "sports-betting ClassifierBettor + LogisticRegression",
                "odds_source": odds_type,
            })

    picks.sort(
        key=lambda p: (p["probability"], p["edge"], p["ev"]),
        reverse=True,
    )
    unique = {}
    for pick in picks:
        key = (pick["date"], pick["home_team"], pick["away_team"])
        if key not in unique:
            unique[key] = pick
    picks = list(unique.values())[:max_picks]

    return {
        "sport": sport,
        "data_source": _source_label(sport, source_mode),
        "training_rows": int(len(X_train)),
        "fixture_rows": int(len(X_fix)),
        "picks": picks,
        "backtest": bt["folds"],
        "backtest_columns": bt["columns"],
        "filtered_count": len(picks),
        "notice": (
            "Only real model outputs and real odds from the selected provider "
            "are displayed. Historical closing prices can differ from current prices."
        ),
    }


def _source_label(sport, source_mode):
    if source_mode == "odds_api":
        return f"{SPORTS[sport]['label']} + The Odds API (real odds; API key required)"
    if sport == "football":
        return "Football-Data.co.uk statistics + market-maximum odds (free)"
    return f"{SPORTS[sport]['label']} statistics (free) — odds required for betting scans"


@app.get("/")
def index():
    return render_template(
        "index.html",
        sports=SPORTS,
        odds_api_configured=bool(os.environ.get("ODDS_API_KEY")),
    )


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "library": "georgedouzas/sports-betting",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _start_job(payload, only_backtest=False):
    if not RUN_LOCK.acquire(blocking=False):
        return None, (
            jsonify({
                "error": "Another model run is already in progress. Wait for it to finish."
            }),
            429,
        )

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "result": None,
            "error": None,
        }

    def worker():
        try:
            result = _run(payload, only_backtest=only_backtest)
            with JOBS_LOCK:
                JOBS[job_id].update(
                    status="complete",
                    result=result,
                )
        except Exception as exc:
            app.logger.exception("Background model job failed")
            with JOBS_LOCK:
                JOBS[job_id].update(
                    status="error",
                    error={
                        "error": str(exc),
                        "detail": "Check Heroku logs for the full traceback.",
                    },
                )
        finally:
            RUN_LOCK.release()

    try:
        threading.Thread(
            target=worker,
            name=f"sportsbet-{job_id[:8]}",
            daemon=True,
        ).start()
    except Exception:
        RUN_LOCK.release()
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        raise

    return job_id, None


@app.post("/api/scan")
def api_scan():
    job_id, error_response = _start_job(
        request.get_json(silent=True) or {}
    )
    if error_response:
        return error_response
    return jsonify({"job_id": job_id, "status": "running"}), 202


@app.post("/api/backtest")
def api_backtest():
    job_id, error_response = _start_job(
        request.get_json(silent=True) or {},
        only_backtest=True,
    )
    if error_response:
        return error_response
    return jsonify({"job_id": job_id, "status": "running"}), 202


@app.get("/api/jobs/<job_id>")
def api_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({
                "error": "Job not found. Please start a new run."
            }), 404

        response = {
            "job_id": job_id,
            "status": job["status"],
        }
        if job["status"] == "complete":
            response["result"] = job["result"]
        elif job["status"] == "error":
            response.update(job["error"] or {
                "error": "Unknown background job failure."
            })
        return jsonify(response)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
