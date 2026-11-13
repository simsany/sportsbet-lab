# Sports AI Scanner — GitHub → Heroku

Önálló, Flask-alapú több sportágas predikciós alkalmazás Football, Tennis, Basketball, Baseball és NFL támogatással. Az aktuális oddsokat a The Odds API v4 szolgáltatja. A történeti adatokat a szerver tölti le és a modelleket automatikusan tanítja.

## Funkciók

- Öt állandó modellcsalád: **Elo, sportághoz igazított score-distribution (Poisson/Dixon–Coles), XGBoost, CatBoost, LightGBM**. Nincs modellválasztó.
- Sportág- és liga-választás, minimum decimális odds, tippszám, rendezés, piac és odds-régió.
- Aktuális / közelgő mérkőzések és bookmaker oddsok a The Odds API-ból. Történelmi Odds API végpontot nem hív.
- Modellvalószínűség, 5 modellből számolt egyezés, modell-EV, implied probability és modell-szórás csak akkor, ha az adott sportág teljes modellcsomagja sikeresen betanult, és a kiválasztott piac H2H.
- Piaci odds-konszenzus külön mező; nem kerül modellvalószínűségként vagy modell-EV-ként feltüntetésre.
- 70% időrendi tanítóhalmaz, 10% kalibráció, 20% érintetlen, későbbi időszakú holdout. A holdouton Accuracy, Log Loss és Brier Score készül.
- Csapatnév-egyezés nélkül nem ad sportági modellpredikciót. Hiányzó adat, sikertelen letöltés vagy kevés tanítóminta esetén a modellmezők üresek maradnak.
- Nincs PostgreSQL és nincs S3.

## GitHub → Heroku beállítás

1. Csomagold ki ezt a ZIP-et, és **a `sports-ai-scanner-v3` mappa tartalmát** töltsd fel a GitHub repository gyökerébe.
2. A GitHub repositoryban: **Settings → Secrets and variables → Actions → New repository secret**:
   - `HEROKU_API_KEY`: Heroku account API key.
   - `HEROKU_APP_NAME`: a már létrehozott Heroku app pontos neve.
3. Heroku → **Settings → Config Vars**: legyen beállítva `ODDS_API_KEY` a The Odds API-kulcsoddal.
4. A Heroku app Python buildpacket használjon. A `Procfile` indítja a Gunicorn szervert.
5. Push a `main` branchre. A GitHub Actions először telepíti a tesztfüggőségeket, futtatja a szintaxis- és unit teszteket, és csak siker esetén deployol Herokura. Kézi indítás: Actions → Test and deploy Sports AI Scanner → Run workflow.

## Első indulás és adatbetanítás

`AUTO_TRAIN=1` az alapértelmezés. Az első web dyno indulásakor háttérfolyamatban lefut a `scripts/prepare_training_data.py`, utána a `model_engine.py` megpróbálja betanítani az öt sportágat. A folyamat a weboldal megnyitása közben is futhat.

Státuszok:
- `GET /health` — app és kulcs beállítottsága.
- `GET /api/models` — sportágankénti betanítási státusz és holdout metrikák.
- `POST /api/train` — adatfrissítés és újratanítás.
- `POST /api/scan` — valós oddsok lekérése és elemzés.

A manuális újratanítás védelmére állítsd be a `TRAINING_ADMIN_TOKEN` Config Var-t; utána a `POST /api/train` kéréshez `X-Training-Token` header kell.

## Adatforrások és normalizálás

| Sport | Forrás | Felhasználás |
|---|---|---|
| Football | football-data.co.uk | Befejezett mérkőzések, FT gólok és csapatnevek |
| Tennis | Hugging Face `minmaxxxer/atp-wta-matches-with-odds` | ATP/WTA főtáblás meccsek; győztes/vesztes és szettadatok, ha elérhetők |
| Basketball | Hugging Face `hamzas/nba-games` | NBA meccsek és pontszámok; csapatkódok feloldása |
| Baseball | MLB Stats API | Befejezett alapszakasz mérkőzések, futások és csapatnevek |
| NFL | nflverse schedules | Befejezett REG/POST mérkőzések és csapatkódok feloldása |

A források sémája és elérhetősége változhat. A letöltő ezt jelzi, nem gyárt helyettesítő sorokat. A Hugging Face adatkártyák és a források feltételei változhatnak; üzleti/kereskedelmi felhasználás előtt ellenőrizd a licencet. A letöltött nyers adatok és feldolgozott CSV-k a Heroku dyno ideiglenes fájlrendszerén vannak.

## Modell-módszertan és korlátok

- Elo és gördülő formajellemzők kizárólag az adott mérkőzés előtti eredményekből készülnek.
- Football: független Poisson-gólmodell Dixon–Coles alacsony gólszám-korrekcióval.
- Baseball: független Poisson-futásmodell.
- Basketball/NFL/Tennis: score-margin modell normális eloszlással; nem alkalmazunk nyers Poisson-eloszlást a nagy pontszámokra.
- Az XGBoost, CatBoost és LightGBM az időrendben korábbi adatokon tanulnak. Hőmérséklet-alapú kalibrációt csak a külön kalibrációs szakasz használ.
- A holdout metrikák nem garantálják a jövőbeli eredményt. A modell-EV a modellbecslés és a pillanatnyilag elérhető legjobb odds összevetése, nem biztos nyereség.
- Jelenleg csak H2H kimenetelhez számolunk saját modellvalószínűséget. Spread/total soroknál a modellmezők üresek maradnak.
- A csapatnevek eltérése miatt egyes eseményeknél nem lesz saját modellpredikció. Ez szándékos: nem használunk vakon 1500-as Elo-alapértéket ismeretlen csapatokra.

## Heroku fájlrendszer

A Heroku dyno fájlrendszere ideiglenes: újraindításkor a letöltött adatok és a modellek elveszhetnek, ezért az automatikus tanítás újra lefuthat. Több web dyno nem ajánlott ehhez a helyi fájlos/cache-es beállításhoz. Ha a dyno memória-korlátba fut, nagyobb memóriájú dyno szükséges. A modellek első tanítása több percig is tarthat.

## Lokális fejlesztés

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m compileall -q .
pytest -q
ODDS_API_KEY=your_key python app.py
```

## Főbb fájlok

- `app.py` — Flask API, Odds API, konfiguráció, scanner.
- `model_engine.py` — adatfeldolgozás utáni tanítás, kalibráció, holdout metrikák és predikció.
- `scripts/prepare_training_data.py` — publikus történeti adatok letöltése/normalizálása.
- `templates/`, `static/` — web UI.
- `.github/workflows/deploy-heroku.yml` — teszt + automatikus deploy.

## Pre-deploy validation

Run locally with Python 3.12:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
pytest -q
```

The test suite includes pre-match feature leakage checks, probability-shape checks, canonical CSV normalization, and a full synthetic training/prediction smoke test for all five sports. It does not certify betting profitability or substitute for a real-data backtest.
