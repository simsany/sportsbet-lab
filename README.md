# SportsBet Lab

A Heroku-ready Flask interface around the real [`georgedouzas/sports-betting`](https://github.com/georgedouzas/sports-betting) library. It does not fabricate sample predictions and does not place bets.

## Supported sources and sports

The library's current public source catalogue includes:
- **Football/soccer:** `FootballDataStats` + `FootballDataOdds` for free historical statistics and closing odds. Free mode uses `market_maximum` odds. The free feed covers multiple leagues, with the UI allowing selection of supported league names.
- **NBA:** `NBAStats` (ESPN statistics) + optional `OddsApi`.
- **EuroLeague:** `EuroLeagueStats` + optional `OddsApi`.

`OddsApi` requires a provider API key and may incur costs/usage charges. Add it only if you choose to use that provider. In free mode, NBA and EuroLeague are stats-only and the app deliberately refuses to emit betting picks without real odds. The library's current stats source classes do not provide a complete native stats loader for every sport key exposed by the odds API; this app only enables end-to-end models for football, NBA and EuroLeague.

## Model behavior

- Builds `sportsbet.dataloaders.DataLoader` from real source classes.
- Fits `sportsbet.evaluation.ClassifierBettor` using a scikit-learn `MultiOutputClassifier(LogisticRegression(...))` pipeline.
- One-hot encodes categorical features and imputes missing values inside the fitted preprocessing pipeline.
- Runs the library's `sportsbet.evaluation.backtest` with `TimeSeriesSplit` before returning picks.
- Uses the library's real `predict_proba` and `bet` outputs. If output shapes or source data are incompatible, the request fails instead of guessing.
- Applies filters to probability, confidence margin, edge, EV, decimal odds range and result count; keeps at most one selection per match.
- Confidence is defined transparently as the probability margin between the top and second outcome within the same complementary market group. It is not a separately calibrated confidence score.
- Edge = model probability − (1 / decimal odds). Expected value per unit stake = model probability × decimal odds − 1.
- The app is a research interface only. It has no bookmaker login or automatic execution.

## Local run

Python 3.12 is targeted.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Heroku

1. Create an empty GitHub repository named `sportsbet-lab`.
2. Push these files to the repository.
3. Create a new Heroku app and connect the GitHub repository, or deploy with the Heroku CLI:
   ```bash
   heroku login
   heroku create YOUR-UNIQUE-APP-NAME
   git push heroku main
   heroku logs --tail
   ```
4. The `Procfile` starts Gunicorn. `/health` is a simple health endpoint.
5. Free football mode needs no API key. If you choose to use The Odds API for football, NBA or EuroLeague, add `ODDS_API_KEY` as a Heroku Config Var. Do not put keys in Git or the chat.

## Limits

- Source downloads and training can take several minutes. Start with one or two leagues and a few seasons.
- Heroku dyno storage is ephemeral; this app does not rely on persisted local model files and refits per run.
- A small dyno may run out of memory/time on large selections. A single model run is allowed at a time.
- Historical odds are closing snapshots; prices can differ from current bookmaker offers.
- Backtest results are historical only and do not guarantee future profit.
