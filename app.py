from __future__ import annotations

import os
import threading
import traceback
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
        "England", "Scotland", "Germany", "Italy", "Spain", "France",
        "Netherlands", "Belgium", "Portugal", "Turkey", "Greece", "Argentina",
        "Austria", "Brazil", "China", "Denmark", "Finland", "Ireland", "Japan",
        "Mexico", "Norway", "Poland", "Romania", "Russia", "Sweden",
        "Switzerland", "USA"
    ]},
    "nba": {"label": "NBA", "league_options": ["NBA"]},
    "euroleague": {"label": "EuroLeague", "league_options": ["Euroleague"]},
}
OUTCOME_MARKETS = ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]


class ModelRunError(Exception):
    def __init__(self, info):
        self.info = info
        super().__init__(info.get("error", "Model run failed."))


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


def _diag(stage, message, **details):
    item = {"stage": stage, "message": message}
    item.update({k: _plain(v) for k, v in details.items()})
    app.logger.info("DIAGNOSTIC %s", item)
    return item


def _sportsbet_sources(sport, leagues, years, source_mode):
    from sportsbet.dataloaders import DataLoader
    from sportsbet.sources import (
        FootballDataOdds, FootballDataStats, NBAStats, EuroLeagueStats, OddsApi
    )

    use_api = source_mode == "odds_api"
    if use_api and not os.environ.get("ODDS_API_KEY"):
        raise ValueError(
            "Odds API selected, but ODDS_API_KEY is missing from Heroku Config Vars."
        )

    if sport == "football":
        stats = FootballDataStats()
        odds = OddsApi(
            key_env="ODDS_API_KEY", markets=["h2h", "totals"], regions=["eu"]
        ) if use_api else FootballDataOdds()
        grid = {"league": leagues, "division": [1], "year": years}
        odds_type = "odds_api" if use_api else "market_maximum"
    elif sport == "nba":
        if not use_api:
            raise ValueError("NBA scans require Odds API. Select Odds API mode.")
        stats = NBAStats()
        odds = OddsApi(key_env="ODDS_API_KEY", markets=["h2h"], regions=["eu"])
        grid = {"league": ["NBA"], "division": [1], "year": years}
        odds_type = "odds_api"
    elif sport == "euroleague":
        if not use_api:
            raise ValueError("EuroLeague scans require Odds API. Select Odds API mode.")
        stats = EuroLeagueStats()
        odds = OddsApi(key_env="ODDS_API_KEY", markets=["h2h"], regions=["eu"])
        grid = {"league": ["Euroleague"], "division": [1], "year": years}
        odds_type = "odds_api"
    else:
        raise ValueError("Unsupported sport.")
    return DataLoader(param_grid=grid, stats=stats, odds=odds), odds_type


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
    numeric = list(X.select_dtypes(include=[np.number, "bool"]).columns)
    if not categorical and not numeric:
        raise ValueError("No usable model features were returned by the data source.")

    transformers = []
    if categorical:
        transformers.append((
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ), categorical
        ))
    if numeric:
        transformers.append((SimpleImputer(strategy="median"), numeric))
    prep = make_column_transformer(*transformers, remainder="drop")
    return make_pipeline(
        prep,
        MultiOutputClassifier(LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=7,
            class_weight="balanced"
        ))
    )


def _choose_markets(Y, O):
    from sportsbet.evaluation import derive_market_base, find_latest_odds_column
    y_bases = {derive_market_base(str(c)) for c in Y.columns}
    o_bases = {str(c).split("__")[1] for c in O.columns if len(str(c).split("__")) == 4}
    common = [m for m in OUTCOME_MARKETS if m in y_bases and m in o_bases]
    usable = []
    for market in common:
        col = find_latest_odds_column(list(O.columns), market)
        if col and pd.to_numeric(O[col], errors="coerce").gt(1).any():
            usable.append(market)
    if not usable:
        raise ValueError("No common outcome labels and valid odds were found.")
    return usable


def _backtest_result(bettor, X, Y, O, folds):
    from sklearn.model_selection import TimeSeriesSplit
    from sportsbet.evaluation import backtest
    n = max(2, min(5, int(folds)))
    if len(X) < max(40, n + 5):
        raise ValueError(f"Not enough historical rows ({len(X)}) for {n} backtest folds.")
    result = backtest(
        bettor, X, Y, O, cv=TimeSeriesSplit(n_splits=n), n_jobs=1, verbose=0
    )
    if result is None or result.empty:
        raise ValueError("The library returned no backtest metrics.")
    frame = result.reset_index()
    return {
        "rows": int(len(X)),
        "columns": [str(c) for c in frame.columns],
        "folds": [{str(k): _plain(v) for k, v in row.items()}
                  for row in frame.to_dict(orient="records")],
        "note": "Walk-forward historical evaluation; past results do not guarantee future performance."
    }


def _source_label(sport, source_mode):
    if source_mode == "odds_api":
        return f"{SPORTS[sport]['label']} + The Odds API"
    if sport == "football":
        return "Football-Data.co.uk statistics + market-maximum odds (free)"
    return f"{SPORTS[sport]['label']} statistics"


def _run(payload, only_backtest=False):
    stage = "validate_request"
    diagnostics = []
    try:
        sport = str(payload.get("sport", "football")).lower()
        if sport not in SPORTS:
            raise ValueError("Choose football, NBA or EuroLeague.")
        source_mode = str(payload.get("source_mode", "free")).lower()
        years = sorted({int(y) for y in (payload.get("years") or [2023, 2024, 2025, 2026])
                        if 2000 <= int(y) <= 2035})
        if not years or len(years) > 6:
            raise ValueError("Select between one and six valid season years.")
        leagues = payload.get("leagues") or (
            ["England"] if sport == "football" else [SPORTS[sport]["league_options"][0]]
        )
        if not isinstance(leagues, list):
            leagues = [leagues]
        leagues = [str(x) for x in leagues if str(x) in SPORTS[sport]["league_options"]]
        if not leagues:
            raise ValueError("Select at least one supported league.")

        diagnostics.append(_diag(
            stage, "Request validated.", sport=sport, source_mode=source_mode,
            leagues=", ".join(leagues), years=", ".join(map(str, years))
        ))

        stage = "initialize_data_sources"
        from sportsbet.evaluation import ClassifierBettor
        loader, odds_type = _sportsbet_sources(sport, leagues, years, source_mode)
        diagnostics.append(_diag(
            stage, "Data source initialized.", source=_source_label(sport, source_mode),
            odds_type=odds_type
        ))

        stage = "load_training_data"
        X, Y, O = loader.extract_train_data(
            odds_type=odds_type, input_event_status="preplay"
        )
        diagnostics.append(_diag(
            stage, "Training data loaded.", features=len(X), labels=len(Y), odds=len(O),
            feature_columns=len(X.columns), label_columns=len(Y.columns),
            odds_columns=len(O.columns)
        ))
        if X.empty:
            raise ValueError("The provider returned zero historical training rows.")
        if not (len(X) == len(Y) == len(O)):
            raise ValueError(f"Training rows are misaligned: X={len(X)}, Y={len(Y)}, O={len(O)}.")

        if not only_backtest:
            stage = "load_fixtures"
            Xf, _, Of = loader.extract_fixtures_data()
            diagnostics.append(_diag(
                stage, "Fixture extraction completed.", fixture_rows=len(Xf),
                odds_rows=len(Of), feature_columns=list(map(str, Xf.columns)),
                odds_columns=list(map(str, Of.columns)),
                sample=Xf.head(5).to_dict(orient="records") if not Xf.empty else []
            ))
            if Xf.empty:
                raise ValueError(
                    "The fixture provider returned zero upcoming matches. "
                    "Historical training data was loaded, but no fixtures were returned."
                )
            if len(Xf) != len(Of):
                raise ValueError(f"Fixture/odds rows are misaligned: X={len(Xf)}, O={len(Of)}.")

        stage = "select_markets"
        markets = _choose_markets(Y, O)
        diagnostics.append(_diag(stage, "Markets selected.", markets=markets))

        stage = "build_model"
        bettor = ClassifierBettor(
            classifier=_model_pipeline(X), betting_markets=markets,
            init_cash=10000.0, stake=50.0
        )
        diagnostics.append(_diag(stage, "Model pipeline created."))

        stage = "backtest"
        bt = _backtest_result(bettor, X, Y, O, payload.get("backtest_folds", 3))
        diagnostics.append(_diag(stage, "Backtest completed.", rows=bt["rows"],
                                 folds=len(bt["folds"])))
        if only_backtest:
            return {
                "sport": sport, "data_source": _source_label(sport, source_mode),
                "backtest": bt["folds"], "backtest_columns": bt["columns"],
                "rows": bt["rows"], "note": bt["note"], "diagnostics": diagnostics
            }

        stage = "fit_model"
        bettor.fit(X, Y, O)
        diagnostics.append(_diag(stage, "Model fit completed."))

        stage = "predict_fixtures"
        probabilities = np.asarray(bettor.predict_proba(Xf), dtype=float)
        bets = np.asarray(bettor.bet(Xf, Of), dtype=bool)
        if probabilities.ndim != 2 or probabilities.shape != bets.shape or probabilities.shape[1] != len(markets):
            raise ValueError(f"Unexpected prediction shapes: probabilities={probabilities.shape}, bets={bets.shape}.")
        diagnostics.append(_diag(stage, "Predictions generated.",
                                 probability_shape=probabilities.shape, bet_shape=bets.shape))

        from sportsbet.evaluation import find_latest_odds_column
        odds_cols = {m: find_latest_odds_column(list(Of.columns), m, provider=odds_type)
                     for m in markets}
        min_p = float(payload.get("min_probability", 0.60))
        min_conf = float(payload.get("min_confidence", 0)) / 100
        min_edge = float(payload.get("min_edge", 2)) / 100
        min_ev = float(payload.get("min_ev", 0)) / 100
        min_odds = float(payload.get("min_odds", 1.30))
        max_odds = float(payload.get("max_odds", 8))
        max_picks = max(1, min(50, int(payload.get("max_picks", 20))))

        groups = []
        hda = [m for m in ["home_win", "draw", "away_win"] if m in markets]
        totals = [m for m in ["over_2.5", "under_2.5"] if m in markets]
        if len(hda) > 1:
            groups.append(hda)
        if len(totals) > 1:
            groups.append(totals)
        for m in markets:
            if not any(m in g for g in groups):
                groups.append([m])

        stage = "filter_selections"
        picks = []
        for i in range(len(Xf)):
            for j, market in enumerate(markets):
                if not bets[i, j]:
                    continue
                p = float(probabilities[i, j])
                col = odds_cols.get(market)
                if not col:
                    continue
                try:
                    odds = float(Of.iloc[i][col])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(odds) or odds <= 1:
                    continue
                group = next(g for g in groups if market in g)
                gp = sorted(float(probabilities[i, markets.index(m)]) for m in group)
                confidence = gp[-1] - gp[-2] if len(gp) > 1 else gp[-1]
                implied = 1 / odds
                edge, ev = p - implied, p * odds - 1
                if p < min_p or confidence < min_conf or edge < min_edge or ev < min_ev or odds < min_odds or odds > max_odds:
                    continue
                label = {"home_win": "Home win", "draw": "Draw", "away_win": "Away win"}.get(
                    market, market.replace("_", " ").title()
                )
                row = Xf.iloc[i]
                picks.append({
                    "date": _plain(Xf.index[i]), "league": _plain(row.get("league")),
                    "home_team": _plain(row.get("home_team")), "away_team": _plain(row.get("away_team")),
                    "market": label, "market_key": market, "probability": round(p, 4),
                    "confidence": round(confidence, 4), "odds": round(odds, 3),
                    "implied_probability": round(implied, 4), "edge": round(edge, 4),
                    "ev": round(ev, 4),
                    "model": "sports-betting ClassifierBettor + LogisticRegression",
                    "odds_source": odds_type
                })

        picks.sort(key=lambda p: (p["probability"], p["edge"], p["ev"]), reverse=True)
        unique = {}
        for pick in picks:
            key = (pick["date"], pick["home_team"], pick["away_team"])
            unique.setdefault(key, pick)
        picks = list(unique.values())[:max_picks]
        diagnostics.append(_diag(stage, "Selection filtering completed.",
                                 qualified_picks=len(picks), fixtures=len(Xf)))
        return {
            "sport": sport, "data_source": _source_label(sport, source_mode),
            "training_rows": int(len(X)), "fixture_rows": int(len(Xf)),
            "picks": picks, "backtest": bt["folds"],
            "backtest_columns": bt["columns"], "filtered_count": len(picks),
            "diagnostics": diagnostics,
            "notice": "Only real model outputs and provider odds are displayed."
        }

    except Exception as exc:
        info = {
            "stage": stage, "exception_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "diagnostics": diagnostics
        }
        app.logger.exception("Model run failed at stage %s", stage)
        raise ModelRunError(info) from exc


@app.get("/")
def index():
    return render_template(
        "index.html", sports=SPORTS,
        odds_api_configured=bool(os.environ.get("ODDS_API_KEY"))
    )
    @app.get("/api/source-test")
def source_test():
    import requests
    import pandas as pd
    from io import BytesIO

    url = "https://www.football-data.co.uk/fixtures.csv"

    result = {
        "url": url,
        "http_status": None,
        "content_type": None,
        "file_size": 0,
        "columns": [],
        "sample": [],
        "date_formats": {},
        "future_matches": {},
        "error": None,
    }

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "SportsBetLab/1.0"}
        )

        result["http_status"] = response.status_code
        result["content_type"] = response.headers.get("Content-Type")
        result["file_size"] = len(response.content)

        response.raise_for_status()

        df = pd.read_csv(
            BytesIO(response.content),
            encoding="ISO-8859-1"
        )

        result["columns"] = df.columns.tolist()

        date_col = next(
            (c for c in df.columns if c.strip().lower() == "date"),
            None
        )

        if date_col is None:
            raise ValueError(
                f"Nem található Date oszlop. Kapott oszlopok: {df.columns.tolist()}"
            )

        result["sample"] = (
            df.head(5)
            .fillna("")
            .to_dict(orient="records")
        )

        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            parsed = pd.to_datetime(
                df[date_col],
                format=fmt,
                errors="coerce"
            )

            valid = int(parsed.notna().sum())
            future = int(
                (parsed.dropna() >= pd.Timestamp.today().normalize()).sum()
            )

            result["date_formats"][fmt] = {
                "valid_rows": valid,
                "invalid_rows": int(len(df) - valid),
                "future_rows": future,
            }

        result["future_matches"] = {
            fmt: values["future_rows"]
            for fmt, values in result["date_formats"].items()
        }

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


@app.get("/health")
def health():
    return jsonify({
        "ok": True, "library": "georgedouzas/sports-betting",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


def _start_job(payload, only_backtest=False):
    if not RUN_LOCK.acquire(blocking=False):
        return None, (jsonify({
            "error": "Another model run is already in progress. Wait for it to finish."
        }), 429)

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "result": None, "error": None}

    def worker():
        try:
            result = _run(payload, only_backtest=only_backtest)
            with JOBS_LOCK:
                JOBS[job_id].update(status="complete", result=result)
        except Exception as exc:
            app.logger.exception("Background model job failed")
            if isinstance(exc, ModelRunError):
                info = exc.info
                err = {
                    "error": info["error"],
                    "detail": f"Stage: {info['stage']} | Exception: {info['exception_type']}",
                    "stage": info["stage"], "exception_type": info["exception_type"],
                    "diagnostics": info["diagnostics"],
                    "traceback": info["traceback"]
                }
            else:
                err = {
                    "error": str(exc), "detail": type(exc).__name__,
                    "traceback": traceback.format_exc()
                }
            with JOBS_LOCK:
                JOBS[job_id].update(status="error", error=err)
        finally:
            RUN_LOCK.release()

    try:
        threading.Thread(target=worker, name=f"sportsbet-{job_id[:8]}", daemon=True).start()
    except Exception:
        RUN_LOCK.release()
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        raise
    return job_id, None


@app.post("/api/scan")
def api_scan():
    job_id, error_response = _start_job(request.get_json(silent=True) or {})
    if error_response:
        return error_response
    return jsonify({"job_id": job_id, "status": "running"}), 202


@app.post("/api/backtest")
def api_backtest():
    job_id, error_response = _start_job(
        request.get_json(silent=True) or {}, only_backtest=True
    )
    if error_response:
        return error_response
    return jsonify({"job_id": job_id, "status": "running"}), 202


@app.get("/api/jobs/<job_id>")
def api_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found. Please start a new run."}), 404
        response = {"job_id": job_id, "status": job["status"]}
        if job["status"] == "complete":
            response["result"] = job["result"]
        elif job["status"] == "error":
            response.update(job["error"] or {"error": "Unknown background job failure."})
        return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
