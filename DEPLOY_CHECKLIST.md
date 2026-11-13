# GitHub → Heroku deploy checklist

1. Upload the *contents* of this folder to the root of the GitHub repository (not the enclosing folder).
2. Set GitHub Actions repository secrets:
   - `HEROKU_API_KEY`
   - `HEROKU_APP_NAME`
3. In Heroku Config Vars set `ODDS_API_KEY`.
4. Select the Heroku Python buildpack. This project uses the `Procfile`; do not run `heroku container:release` for this build.
5. Use a dyno with at least 1 GB RAM for the first training run. Keep one web dyno because model artifacts and cache are stored on its ephemeral filesystem.
6. Push to `main`. The GitHub workflow must pass dependency installation, Python compilation and tests before it pushes to Heroku.
7. Check `/health`, `/api/models`, then run a scan from the web UI.

A green deploy means the web process was released; it does not guarantee every external training-data provider is reachable. `/api/models` is the source of truth for per-sport training status. Missing data or failed validation must remain visible as an untrained sport, not as model picks.
