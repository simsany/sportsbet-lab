from __future__ import annotations

import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
RUN_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}

SPORTS = {
    "football": {
        "label": "Football / soccer",
        "league_options": [
            "England", "Scotland", "Germany", "Italy", "Spain", "France",
            "Netherlands", "Belgium", "Portugal", "Turkey", "Greece",
            "Argentina", "Austria", "Brazil", "China", "Denmark", "Finland",
            "Ireland", "Japan", "Mexico", "Norway", "Poland", "Romania",
            "Russia", "Sweden", "Switzerland", "USA"
        ]
    },
    "nba": {"label": "NBA", "league_options": ["NBA"]},
    "euroleague": {"label": "EuroLeague", "league_options": ["Euroleague"]},
}

OUTCOME_MARKETS = [
    "home_win", "draw", "away_win", "over_2.5", "under_2.5"
]


class ModelRunError(Exception):
    def __init__(self, info):
        self.info = info
        super().__init__(info.get("error", "Model run failed."))


def _plain(value):
    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value if isinstance(
        value, (str, int, float, bool)
    ) else str(value)


def _diag(stage, message, **details):
    item = {
        "stage": stage,
        "message": message
    }

    item.update({
        key: _plain(value)
        for key, value in details.items()
    })

    app.logger.info("DIAGNOSTIC %s", item)
    return item


def _sportsbet_sources(sport, leagues, years, source_mode):
    from sportsbet.dataloaders import DataLoader
    from sportsbet.sources import (
        FootballDataOdds,
        FootballDataStats,
        NBAStats,
        EuroLeagueStats,
        OddsApi
    )

    use_api = source_mode == "odds_api"

    if use_api and not os.environ.get("ODDS_API_KEY"):
        raise ValueError(
            "Odds API selected, but ODDS_API_KEY is missing "
            "from Heroku Config Vars."
        )

    if sport == "football":
        stats = FootballDataStats()

        odds = (
            OddsApi(
                key_env="ODDS_API_KEY",
                markets=["h2h", "totals"],
                regions=["eu"]
            )
            if use_api
            else FootballDataOdds()
        )

        grid = {
            "league": leagues,
            "division": [1],
            "year": years
        }

        odds_type = (
            "odds_api" if use_api else "market_maximum"
        )

    elif sport == "nba":
        if not use_api:
            raise ValueError(
                "NBA scans require Odds API. Select Odds API mode."
            )

        stats = NBAStats()

        odds = OddsApi(
            key_env="ODDS_API_KEY",
            markets=["h2h"],
            regions=["eu"]
        )

        grid = {
            "league": ["NBA"],
            "division": [1],
            "year": years
        }

        odds_type = "odds_api"

    elif sport == "euroleague":
        if not use_api:
            raise ValueError(
                "EuroLeague scans require Odds API. "
                "Select Odds API mode."
            )

        stats = EuroLeagueStats()

        odds = OddsApi(
            key_env="ODDS_API_KEY",
            markets=["h2h"],
            regions=["eu"]
        )

        grid = {
            "league": ["Euroleague"],
            "division": [1],
            "year": years
        }

        odds_type = "odds_api"

    else:
        raise ValueError("Unsupported sport.")

    return DataLoader(
        param_grid=grid,
        stats=stats,
        odds=odds
    ), odds_type


def _model_pipeline(X):
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical = list(
        X.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    )

    numeric = list(
        X.select_dtypes(
            include=[np.number, "bool"]
        ).columns
    )

    if not categorical and not numeric:
        raise ValueError(
            "No usable model features were returned by the data source."
        )

    transformers = []

    if categorical:
        transformers.append((
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore")
            ),
            categorical
        ))

    if numeric:
        transformers.append((
            SimpleImputer(strategy="median"),
            numeric
        ))

    prep = make_column_transformer(
        *transformers,
        remainder="drop"
    )

    return make_pipeline(
        prep,
        MultiOutputClassifier(
            LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=7,
                class_weight="balanced"
            )
        )
    )


def _choose_markets(Y, O):
    from sportsbet.evaluation import (
        derive_market_base,
        find_latest_odds_column
    )

    y_bases = {
        derive_market_base(str(column))
        for column in Y.columns
    }

    o_bases = {
        str(column).split("__")[1]
        for column in O.columns
        if len(str(column).split("__")) == 4
    }

    common = [
        market
        for market in OUTCOME_MARKETS
        if market in y_bases and market in o_bases
    ]

    usable = []

    for market in common:
        column = find_latest_odds_column(
            list(O.columns),
            market
        )

        if column and pd.to_numeric(
            O[column],
            errors="coerce"
        ).gt(1).any():
            usable.append(market)

    if not usable:
        raise ValueError(
            "No common outcome labels and valid odds were found."
        )

    return usable


def _backtest_result(bettor, X, Y, O, folds):
    from sklearn.model_selection import TimeSeriesSplit
    from sportsbet.evaluation import backtest

    n = max(2, min(5, int(folds)))

    if len(X) < max(40, n + 5):
        raise ValueError(
            f"Not enough historical rows ({len(X)}) "
            f"for {n} backtest folds."
        )

    result = backtest(
        bettor,
        X,
        Y,
        O,
        cv=TimeSeriesSplit(n_splits=n),
        n_jobs=1,
        verbose=0
    )

    if result is None or result.empty:
        raise ValueError(
            "The library returned no backtest metrics."
        )

    frame = result.reset_index()

    return {
        "rows": int(len(X)),
        "columns": [
            str(column)
            for column in frame.columns
        ],
        "folds": [
            {
                str(key): _plain(value)
                for key, value in row.items()
            }
            for row in frame.to_dict(orient="records")
        ],
        "note": (
            "Walk-forward historical evaluation; "
            "past results do not guarantee future performance."
        )
    }


def _source_label(sport, source_mode):
    if source_mode == "odds_api":
        return (
            f"{SPORTS[sport]['label']} + The Odds API"
        )

    if sport == "football":
        return (
            "Football-Data.co.uk statistics + "
            "market-maximum odds (free)"
        )

    return f"{SPORTS[sport]['label']} statistics"


def _run(payload, only_backtest=False):
    stage = "validate_request"
    diagnostics = []

    try:
        sport = str(
            payload.get("sport", "football")
        ).lower()

        if sport not in SPORTS:
            raise ValueError(
                "Choose football, NBA or EuroLeague."
            )

        source_mode = str(
            payload.get("source_mode", "free")
        ).lower()

        years = sorted({
            int(year)
            for year in (
                payload.get("years")
                or [2023, 2024, 2025, 2026]
            )
            if 2000 <= int(year) <= 2035
        })

        if not years or len(years) > 6:
            raise ValueError(
                "Select between one and six valid season years."
            )

        leagues = payload.get("leagues") or (
            ["England"]
            if sport == "football"
            else [SPORTS[sport]["league_options"][0]]
        )

        if not isinstance(leagues, list):
            leagues = [leagues]

        leagues = [
            str(item)
            for item in leagues
            if str(item) in SPORTS[sport]["league_options"]
        ]

        if not leagues:
            raise ValueError(
                "Select at least one supported league."
            )

        diagnostics.append(
            _diag(
                stage,
                "Request validated.",
                sport=sport,
                source_mode=source_mode,
                leagues=", ".join(leagues),
                years=", ".join(map(str, years))
            )
        )

        stage = "initialize_data_sources"

        from sportsbet.evaluation import ClassifierBettor

        loader, odds_type = _sportsbet_sources(
            sport,
            leagues,
            years,
            source_mode
        )

        diagnostics.append(
            _diag(
                stage,
                "Data source initialized.",
                source=_source_label(sport, source_mode),
                odds_type=odds_type
            )
        )

        stage = "load_training_data"

        X, Y, O = loader.extract_train_data(
            odds_type=odds_type,
            input_event_status="preplay"
        )

        diagnostics.append(
            _diag(
                stage,
                "Training data loaded.",
                features=len(X),
                labels=len(Y),
                odds=len(O),
                feature_columns=len(X.columns),
                label_columns=len(Y.columns),
                odds_columns=len(O.columns)
            )
        )

        if X.empty:
            raise ValueError(
                "The provider returned zero historical training rows."
            )

        if not (len(X) == len(Y) == len(O)):
            raise ValueError(
                f"Training rows are misaligned: "
                f"X={len(X)}, Y={len(Y)}, O={len(O)}."
            )

        if not only_backtest:
            stage = "load_fixtures"

            Xf, _, Of = loader.extract_fixtures_data()

            diagnostics.append(
                _diag(
                    stage,
                    "Fixture extraction completed.",
                    fixture_rows=len(Xf),
                    odds_rows=len(Of),
                    feature_columns=list(
                        map(str, Xf.columns)
                    ),
                    odds_columns=list(
                        map(str, Of.columns)
                    ),
                    sample=(
                        Xf.head(5).to_dict(orient="records")
                        if not Xf.empty
                        else []
                    )
                )
            )

            if Xf.empty:
                raise ValueError(
                    "The fixture provider returned zero upcoming matches. "
                    "Historical training data was loaded, but no fixtures "
                    "were returned."
                )

            if len(Xf) != len(Of):
                raise ValueError(
                    f"Fixture/odds rows are misaligned: "
                    f"X={len(Xf)}, O={len(Of)}."
                )

        stage = "select_markets"

        markets = _choose_markets(Y, O)

        diagnostics.append(
            _diag(
                stage,
                "Markets selected.",
                markets=markets
            )
        )

        stage = "build_model"

        bettor = ClassifierBettor(
            classifier=_model_pipeline(X),
            betting_markets=markets,
            init_cash=10000.0,
            stake=50.0
        )

        diagnostics.append(
            _diag(
                stage,
                "Model pipeline created."
            )
        )

        stage = "backtest"

        bt = _backtest_result(
            bettor,
            X,
            Y,
            O,
            payload.get("backtest_folds", 3)
        )

        diagnostics.append(
            _diag(
                stage,
                "Backtest completed.",
                rows=bt["rows"],
                folds=len(bt["folds"])
            )
        )

        if only_backtest:
            return {
                "sport": sport,
                "data_source": _source_label(
                    sport,
                    source_mode
                ),
                "backtest": bt["folds"],
                "backtest_columns": bt["columns"],
                "rows": bt["rows"],
                "note": bt["note"],
                "diagnostics": diagnostics
            }

        stage = "fit_model"

        bettor.fit(X, Y, O)

        diagnostics.append(
            _diag(
                stage,
                "Model fit completed."
            )
        )

        stage = "predict_fixtures"

        probabilities = np.asarray(
            bettor.predict_proba(Xf),
            dtype=float
        )

        bets = np.asarray(
            bettor.bet(Xf, Of),
            dtype=bool
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape != bets.shape
            or probabilities.shape[1] != len(markets)
        ):
            raise ValueError(
                f"Unexpected prediction shapes: "
                f"probabilities={probabilities.shape}, "
                f"bets={bets.shape}."
            )

        diagnostics.append(
            _diag(
                stage,
                "Predictions generated.",
                probability_shape=probabilities.shape,
                bet_shape=bets.shape
            )
        )

        from sportsbet.evaluation import find_latest_odds_column

        odds_cols = {
            market: find_latest_odds_column(
                list(Of.columns),
                market,
                provider=odds_type
            )
            for market in markets
        }

        min_p = float(
            payload.get("min_probability", 0.60)
        )

        min_conf = float(
            payload.get("min_confidence", 0)
        ) / 100

        min_edge = float(
            payload.get("min_edge", 2)
        ) / 100

        min_ev = float(
            payload.get("min_ev", 0)
        ) / 100

        min_odds = float(
            payload.get("min_odds", 1.30)
        )

        max_odds = float(
            payload.get("max_odds", 8)
        )

        max_picks = max(
            1,
            min(50, int(payload.get("max_picks", 20)))
        )

        groups = []

        hda = [
            market
            for market in ["home_win", "draw", "away_win"]
            if market in markets
        ]

        totals = [
            market
            for market in ["over_2.5", "under_2.5"]
            if market in markets
        ]

        if len(hda) > 1:
            groups.append(hda)

        if len(totals) > 1:
            groups.append(totals)

        for market in markets:
            if not any(
                market in group
                for group in groups
            ):
                groups.append([market])

        stage = "filter_selections"

        picks = []

        for i in range(len(Xf)):
            for j, market in enumerate(markets):
                if not bets[i, j]:
                    continue

                p = float(probabilities[i, j])

                column = odds_cols.get(market)

                if not column:
                    continue

                try:
                    odds = float(Of.iloc[i][column])
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(odds) or odds <= 1:
                    continue

                group = next(
                    group
                    for group in groups
                    if market in group
                )

                gp = sorted(
                    float(
                        probabilities[
                            i,
                            markets.index(item)
                        ]
                    )
                    for item in group
                )

                confidence = (
                    gp[-1] - gp[-2]
                    if len(gp) > 1
                    else gp[-1]
                )

                implied = 1 / odds
                edge = p - implied
                ev = p * odds - 1

                if (
                    p < min_p
                    or confidence < min_conf
                    or edge < min_edge
                    or ev < min_ev
                    or odds < min_odds
                    or odds > max_odds
                ):
                    continue

                label = {
                    "home_win": "Home win",
                    "draw": "Draw",
                    "away_win": "Away win"
                }.get(
                    market,
                    market.replace("_", " ").title()
                )

                row = Xf.iloc[i]

                picks.append({
                    "date": _plain(Xf.index[i]),
                    "league": _plain(row.get("league")),
                    "home_team": _plain(row.get("home_team")),
                    "away_team": _plain(row.get("away_team")),
                    "market": label,
                    "market_key": market,
                    "probability": round(p, 4),
                    "confidence": round(confidence, 4),
                    "odds": round(odds, 3),
                    "implied_probability": round(implied, 4),
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                    "model": (
                        "sports-betting ClassifierBettor + "
                        "LogisticRegression"
                    ),
                    "odds_source": odds_type
                })

        picks.sort(
            key=lambda item: (
                item["probability"],
                item["edge"],
                item["ev"]
            ),
            reverse=True
        )

        unique = {}

        for pick in picks:
            key = (
                pick["date"],
                pick["home_team"],
                pick["away_team"]
            )

            unique.setdefault(key, pick)

        picks = list(unique.values())[:max_picks]

        diagnostics.append(
            _diag(
                stage,
                "Selection filtering completed.",
                qualified_picks=len(picks),
                fixtures=len(Xf)
            )
        )

        return {
            "sport": sport,
            "data_source": _source_label(
                sport,
                source_mode
            ),
            "training_rows": int(len(X)),
            "fixture_rows": int(len(Xf)),
            "picks": picks,
            "backtest": bt["folds"],
            "backtest_columns": bt["columns"],
            "filtered_count": len(picks),
            "diagnostics": diagnostics,
            "noti
