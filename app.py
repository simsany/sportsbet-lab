
import os, time, json, threading, subprocess, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request
from model_engine import predict as model_predict, model_status, train_all

app = Flask(__name__)
BASE_URL = "https://api.the-odds-api.com/v4"
API_KEY = os.getenv("ODDS_API_KEY", "").strip()
CONFIG_PATH = Path(os.getenv("DATA_DIR", ".")) / "scanner_config.json"
CACHE_TTL = int(os.getenv("ODDS_CACHE_TTL", "90"))
_lock = threading.Lock()
_cache = {}
_training_lock = threading.Lock()
_training_state = {"status": "idle", "results": []}

SPORT_GROUPS = {
    "football": ["soccer"],
    "tennis": ["tennis"],
    "basketball": ["basketball"],
    "baseball": ["baseball"],
    "nfl": ["americanfootball_nfl"],
}
SPORT_LABELS = {
    "football": "Football",
    "tennis": "Tennis",
    "basketball": "Basketball",
    "baseball": "Baseball",
    "nfl": "NFL",
}
FIXED_MODELS = [
    {"id": "elo", "name": "Elo", "description": "Rating-alapú erőbecslés"},
    {"id": "poisson", "name": "Poisson / Dixon–Coles", "description": "Gól- és ponteloszlás-alapú modell"},
    {"id": "xgboost", "name": "XGBoost", "description": "Felügyelt gradient boosting modell"},
    {"id": "catboost", "name": "CatBoost", "description": "Felügyelt boosting kategorikus változókkal"},
    {"id": "lightgbm", "name": "LightGBM", "description": "Felügyelt, hatékony gradient boosting"},
]
DEFAULT_CONFIG = {
    "sports": ["football", "tennis", "basketball", "baseball", "nfl"],
    "sport_keys": [],
    "min_odds": 1.70,
    "sort_by": "probability",
    "max_picks": 50,
    "regions": ["eu"],
    "markets": ["h2h"],
    "include_inplay": False,
}


def load_config():
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}

    out = {**DEFAULT_CONFIG, **cfg}
    out["sports"] = [s for s in out.get("sports", []) if s in SPORT_LABELS]
    out["sport_keys"] = [
        str(k) for k in out.get("sport_keys", []) if isinstance(k, str)
    ]
    out["markets"] = [
        m for m in out.get("markets", [])
        if m in {"h2h", "spreads", "totals"}
    ] or ["h2h"]
    out["regions"] = [
        r for r in out.get("regions", [])
        if r in {"eu", "uk", "us", "us2", "au"}
    ] or ["eu"]

    try:
        out["min_odds"] = max(1.01, min(100, float(out["min_odds"])))
    except Exception:
        out["min_odds"] = 1.70

    try:
        out["max_picks"] = max(3, min(1000, int(out["max_picks"])))
    except Exception:
        out["max_picks"] = 50

    if out["sort_by"] not in {
        "probability", "ev", "odds", "agreement", "start_time"
    }:
        out["sort_by"] = "probability"

    return out


def save_config(cfg):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def odds_request(path, params=None):
    if not API_KEY:
        raise RuntimeError(
            "Hiányzik az ODDS_API_KEY környezeti változó a Heroku Config Vars beállításai közül."
        )

    params = dict(params or {})
    params["apiKey"] = API_KEY
    url = BASE_URL + path
    now = time.time()

    cache_key = (
        url,
        tuple(sorted(
            (k, str(v)) for k, v in params.items() if k != "apiKey"
        ))
    )

    with _lock:
        cached = _cache.get(cache_key)
        if cached and now - cached[0] < CACHE_TTL:
            return cached[1], cached[2], True

    try:
        resp = requests.get(url, params=params, timeout=25)
    except requests.RequestException as e:
        raise RuntimeError(f"Odds API hálózati hiba: {e}")

    if resp.status_code == 401:
        raise RuntimeError(
            "Odds API 401: az ODDS_API_KEY érvénytelen vagy nem ehhez az API-hoz tartozik."
        )

    if resp.status_code == 429:
        raise RuntimeError(
            "Odds API 429: elérted a kérési korlátot. Ellenőrizd a kvótát."
        )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Odds API HTTP {resp.status_code}: {resp.text[:500]}"
        )

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError("Az Odds API nem JSON választ adott.")

    quota = {
        k: resp.headers.get(k)
        for k in [
            "x-requests-remaining",
            "x-requests-used",
            "x-requests-last"
        ]
    }

    with _lock:
        _cache[cache_key] = (now, data, quota)

    return data, quota, False


def get_sports_catalog():
    data, quota, cached = odds_request("/sports", {})
    return data, quota, cached


def classify_sport(sport_obj):
    key = (sport_obj.get("key") or "").lower()
    group = (sport_obj.get("group") or "").lower()

    if key.startswith("soccer_") or "soccer" in group:
        return "football"
    if key.startswith("tennis_") or "tennis" in group:
        return "tennis"
    if key.startswith("basketball_") or "basketball" in group:
        return "basketball"
    if key.startswith("baseball_") or "baseball" in group:
        return "baseball"
    if key == "americanfootball_nfl":
        return "nfl"

    return None


def decimal_price(price, fmt="decimal"):
    try:
        p = float(price)

        if fmt == "decimal":
            return p
        if p > 0:
            return 1 + (p / 100)

        return 1 + (100 / abs(p))
    except Exception:
        return None


def build_market_picks(event, selected_markets, min_odds):
    rows = []
    books = event.get("bookmakers") or []

    for market_key in selected_markets:
        by_selection = defaultdict(list)

        for book in books:
            market = next(
                (
                    m for m in book.get("markets", [])
                    if m.get("key") == market_key
                ),
                None
            )

            if not market:
                continue

            outcomes = market.get("outcomes") or []
            implied = []

            for outcome in outcomes:
                price = decimal_price(outcome.get("price"))

                if price and price > 1:
                    implied.append((outcome, price))

            total = sum(1 / p for _, p in implied)

            if total <= 0:
                continue

            for outcome, price in implied:
                name = outcome.get("name", "")
                point = outcome.get("point")
                selection = (
                    f"{name}" + (f" ({point})" if point is not None else "")
                )

                by_selection[selection].append({
                    "bookmaker": (
                        book.get("title")
                        or book.get("key")
                        or "Unknown"
                    ),
                    "price": price,
                    "fair_prob": (1 / price) / total,
                    "last_update": market.get("last_update")
                })

        for selection, offers in by_selection.items():
            best = max(offers, key=lambda x: x["price"])
            avg_fair = sum(
                x["fair_prob"] for x in offers
            ) / len(offers)

            market_ev = (avg_fair * best["price"]) - 1

            if best["price"] < min_odds:
                continue

            model_result = None
            model_error = None

            sport_name = classify_sport({
                "key": event.get("sport_key", ""),
                "group": event.get("sport_title", "")
            })

            # Jelenleg csak a h2h / Moneyline modell-előrejelzés van bekötve.
            # Spread és Totals esetén nem szabad hamis modellbecslést mutatni.
            if market_key == "h2h" and sport_name:
                try:
                    model_result = model_predict(
                        sport_name,
                        str(event.get("home_team", "")),
                        str(event.get("away_team", "")),
                        event.get("commence_time")
                    )
                except Exception as exc:
                    model_error = (
                        f"{type(exc).__name__}: {exc}"
                    )[:300]

            model_probability = None
            model_ev = None
            model_agreement = None
            uncertainty = None
            models_used = 0

            if model_result:
                outcome_name = (
                    str(selection).split(" (")[0].strip().lower()
                )
                home_name = (
                    str(event.get("home_team", "")).strip().lower()
                )
                away_name = (
                    str(event.get("away_team", "")).strip().lower()
                )

                if outcome_name == home_name:
                    outcome_index = 0
                elif sport_name == "football" and outcome_name in {
                    "draw", "tie"
                }:
                    outcome_index = 1
                elif sport_name == "football" and outcome_name == away_name:
                    outcome_index = 2
                elif sport_name != "football" and outcome_name == away_name:
                    outcome_index = 1
                else:
                    outcome_index = None

                probabilities = model_result.get("probabilities", [])

                if (
                    outcome_index is not None
                    and outcome_index < len(probabilities)
                ):
                    model_probability = (
                        probabilities[outcome_index] * 100
                    )

                    participating = model_result.get(
                        "model_probabilities", {}
                    )
                    models_used = len(participating)

                    votes = sum(
                        1
                        for probs in participating.values()
                        if int(max(
                            range(len(probs)),
                            key=lambda i: probs[i]
                        )) == outcome_index
                    )

                    model_agreement = (
                        f"{votes}/{models_used}" if models_used else None
                    )

                    model_ev = (
                        probabilities[outcome_index] * best["price"] - 1
                    ) * 100

                    uncertainty = model_result.get("uncertainty")

            rows.append({
                "event_id": event.get("id"),
                "sport_key": event.get("sport_key"),
                "sport_title": event.get("sport_title"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
                "market": market_key,
                "selection": selection,
                "best_odds": round(best["price"], 3),
                "best_bookmaker": best["bookmaker"],
                "bookmaker_count": len(offers),
                "market_probability": round(avg_fair * 100, 2),
                "implied_probability": round(
                    (1 / best["price"]) * 100, 2
                ),
                "market_price_vs_consensus_pct": round(
                    market_ev * 100, 2
                ),
                "model_probability": (
                    round(model_probability, 2)
                    if model_probability is not None else None
                ),
                "model_agreement": model_agreement,
                "model_ev": (
                    round(model_ev, 2)
                    if model_ev is not None else None
                ),
                "uncertainty": (
                    round(uncertainty, 2)
                    if uncertainty is not None else None
                ),
                "training_rows": (
                    model_result.get("training_rows")
                    if model_result else None
                ),
                "models_used": models_used,
                "model_error": model_error,
                "data_source": "The Odds API",
                "last_update": best.get("last_update"),
                "probability_type": (
                    "model ensemble"
                    if model_probability is not None
                    else "odds-derived, de-vigged market consensus"
                )
            })

    return rows


def fetch_scan(cfg):
    if not API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY nincs beállítva a Heroku Config Vars-ban."
        )

    catalog, catalog_quota, catalog_cached = get_sports_catalog()

    active = [
        s for s in catalog
        if s.get("active")
        and classify_sport(s) in cfg["sports"]
        and not s.get("has_outrights")
    ]

    grouped = defaultdict(list)

    for sport in active:
        grouped[classify_sport(sport)].append(sport)

    picks = []
    errors = []
    quotas = [catalog_quota]
    found = {s: 0 for s in cfg["sports"]}

    for sport in cfg["sports"]:
        candidates = grouped.get(sport, [])

        if cfg.get("sport_keys"):
            candidates = [
                sp for sp in candidates
                if sp.get("key") in cfg["sport_keys"]
            ]

        if not candidates:
            errors.append(
                f"{SPORT_LABELS[sport]}: az API kulcsod csomagjában "
                "nincs aktív sport/versenysorozat."
            )
            continue

        for sp in candidates:
            try:
                data, quota, cached = odds_request(
                    f"/sports/{sp['key']}/odds",
                    {
                        "regions": ",".join(cfg["regions"]),
                        "markets": ",".join(cfg["markets"]),
                        "oddsFormat": "decimal",
                        "dateFormat": "iso"
                    }
                )

                quotas.append(quota)
                found[sport] += len(data)

                for event in data:
                    commence = event.get("commence_time")

                    if not cfg["include_inplay"] and commence:
                        try:
                            if datetime.fromisoformat(
                                commence.replace("Z", "+00:00")
                            ) < datetime.now(timezone.utc):
                                continue
                        except Exception:
                            pass

                    picks.extend(
                        build_market_picks(
                            event,
                            cfg["markets"],
                            cfg["min_odds"]
                        )
                    )

            except Exception as exc:
                errors.append(
                    f"{sp.get('title', sp['key'])}: {exc}"
                )

    dedup = {}

    for pick in picks:
        dedup[
            (
                pick["event_id"],
                pick["market"],
                pick["selection"]
            )
        ] = pick

    picks = list(dedup.values())

    if cfg["sort_by"] == "probability":
        picks.sort(
            key=lambda p: (
                p["model_probability"] is not None,
                p["model_probability"]
                if p["model_probability"] is not None
                else p["market_probability"],
                p["best_odds"]
            ),
            reverse=True
        )

    elif cfg["sort_by"] == "ev":
        picks.sort(
            key=lambda p: (
                p["model_ev"] is not None,
                p["model_ev"]
                if p["model_ev"] is not None
                else -9999
            ),
            reverse=True
        )

    elif cfg["sort_by"] == "odds":
        picks.sort(
            key=lambda p: p["best_odds"],
            reverse=True
        )

    elif cfg["sort_by"] == "agreement":
        picks.sort(
            key=lambda p: (
                p["model_agreement"] is not None,
                p["model_agreement"] or "",
                p["bookmaker_count"]
            ),
            reverse=True
        )

    elif cfg["sort_by"] == "start_time":
        picks.sort(
            key=lambda p: p.get("commence_time") or "9999"
        )

    return {
        "status": "ok" if picks else "no_picks",
        "picks": picks[:cfg["max_picks"]],
        "total_found": len(picks),
        "events_by_sport": found,
        "errors": errors,
        "quota": quotas,
        "models": model_status(),
        "notice": (
            "A modellmezők csak ténylegesen betöltött és az adott "
            "eseményre előrejelzést adó modellekből készülnek. "
            "Az egyezés nevezője a ténylegesen részt vevő modellek száma. "
            "Jelenleg a modell-előrejelzés a győztes (h2h / Moneyline) "
            "piacra van bekötve; Spread és Totals esetén a modellmezők "
            "üresek maradnak. A piaci valószínűség nem modellbecslés."
        )
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        config=load_config(),
        models=FIXED_MODELS
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "odds_api_key_configured": bool(API_KEY),
        "training": _training_state["status"]
    })


@app.get("/api/models")
def api_models():
    return jsonify({
        "sports": model_status(),
        "training": _training_state
    })


def _run_training():
    global _training_state

    with _training_lock:
        _training_state = {
            "status": "running",
            "results": []
        }

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).parent
                        / "scripts"
                        / "prepare_training_data.py"
                    )
                ],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=int(
                    os.getenv("TRAINING_TIMEOUT_SECONDS", "2400")
                )
            )

            results = train_all()
            trained = sum(
                1 for item in results
                if item.get("status") == "trained"
            )

            status = (
                "completed"
                if proc.returncode == 0 and trained == len(results)
                else (
                    "completed_with_warnings"
                    if trained
                    else "failed"
                )
            )

            _training_state = {
                "status": status,
                "trained_sports": trained,
                "results": results,
                "download_exit_code": proc.returncode,
                "log": (
                    proc.stdout + "\n" + proc.stderr
                )[-12000:]
            }

            if proc.returncode != 0 and not trained:
                _training_state["error"] = (
                    "Az adatletöltés nem adott használható tanítóadatot. "
                    "Nézd meg a logot."
                )

        except Exception as exc:
            _training_state = {
                "status": "failed",
                "error": str(exc),
                "results": []
            }


@app.post("/api/train")
def api_train():
    global _training_state

    if _training_state.get("status") == "running":
        return jsonify({
            "ok": True,
            "status": "running"
        }), 202

    if (
        os.getenv("TRAINING_ADMIN_TOKEN")
        and request.headers.get("X-Training-Token")
        != os.getenv("TRAINING_ADMIN_TOKEN")
    ):
        return jsonify({"error": "Unauthorized"}), 401

    threading.Thread(
        target=_run_training,
        daemon=True
    ).start()

    return jsonify({
        "ok": True,
        "status": "started"
    }), 202


@app.get("/api/config")
def api_config():
    return jsonify(load_config())


@app.put("/api/config")
def update_config():
    incoming = request.get_json(silent=True) or {}
    cfg = load_config()

    if "sports" in incoming:
        cfg["sports"] = [
            s for s in incoming["sports"]
            if s in SPORT_LABELS
        ]

    if "sport_keys" in incoming:
        cfg["sport_keys"] = [
            str(k) for k in incoming["sport_keys"]
            if isinstance(k, str)
        ][:200]

    if "min_odds" in incoming:
        try:
            x = float(incoming["min_odds"])
            if not 1.01 <= x <= 100:
                raise ValueError()
            cfg["min_odds"] = x
        except Exception:
            return jsonify({
                "error": "Minimum odds: 1.01–100."
            }), 400

    if "max_picks" in incoming:
        try:
            x = int(incoming["max_picks"])
            if not 3 <= x <= 1000:
                raise ValueError()
            cfg["max_picks"] = x
        except Exception:
            return jsonify({
                "error": "Tippek száma: 3–1000."
            }), 400

    if "sort_by" in incoming:
        if incoming["sort_by"] not in {
            "probability",
            "ev",
            "odds",
            "agreement",
            "start_time"
        }:
            return jsonify({
                "error": "Érvénytelen rendezés."
            }), 400

        cfg["sort_by"] = incoming["sort_by"]

    if "markets" in incoming:
        cfg["markets"] = [
            m for m in incoming["markets"]
            if m in {"h2h", "spreads", "totals"}
        ]

        if not cfg["markets"]:
            return jsonify({
                "error": "Legalább egy piacot válassz."
            }), 400

    if "regions" in incoming:
        cfg["regions"] = [
            r for r in incoming["regions"]
            if r in {"eu", "uk", "us", "us2", "au"}
        ]

        if not cfg["regions"]:
            return jsonify({
                "error": "Legalább egy odds régiót válassz."
            }), 400

    cfg["include_inplay"] = bool(
        incoming.get(
            "include_inplay",
            cfg["include_inplay"]
        )
    )

    save_config(cfg)

    return jsonify({
        "ok": True,
        "config": cfg
    })


@app.get("/api/sports")
def api_sports():
    try:
        catalog, quota, cached = get_sports_catalog()
        available = []

        for sport in catalog:
            sport_name = classify_sport(sport)

            if sport_name:
                available.append({
                    "key": sport["key"],
                    "title": sport.get("title"),
                    "sport": sport_name,
                    "active": sport.get("active", False)
                })

        return jsonify({
            "available": available,
            "quota": quota,
            "cached": cached
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@app.post("/api/scan")
def api_scan():
    cfg = load_config()

    try:
        return jsonify(fetch_scan(cfg))
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "picks": []
        }), 502


if (
    os.getenv("AUTO_TRAIN", "1").lower()
    in {"1", "true", "yes"}
    and not any(
        x.get("status") == "trained"
        for x in model_status()
    )
):
    threading.Thread(
        target=_run_training,
        daemon=True
    ).start()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000"))
    )
