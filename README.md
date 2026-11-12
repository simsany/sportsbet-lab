# ProphitBet Web Port — PostgreSQL recovery, no S3/AWS

Uses the original ProphitBet backend pinned to commit
`49fa5ff14783af5dbbac7919ec743ab41628fcac`.

## Persistence model

Heroku Postgres stores only the rebuild recipes (league configuration and model
training configuration). The dyno filesystem stores the generated Feather datasets,
pickle indexes and trained classifier files as a disposable cache.

After a dyno filesystem reset, startup:
1. Reads league and model recipes from PostgreSQL.
2. Downloads and computes the original historical league dataset again.
3. Retrains the configured original ProphitBet classifiers.
4. Verifies the dataset and saved classifier can actually be loaded.
5. Starts Gunicorn only after recovery succeeds.

No S3, AWS credentials, or external file storage are used.

## Heroku setup

Attach Heroku Postgres to the app. Heroku injects `DATABASE_URL`; the app creates
`prophitbet_recovery` and its index automatically. Do not manually create tables.

Every successful league creation and model training saves its recipe to PostgreSQL.
Recipes are saved before the expensive download/training step, so an interrupted
operation can be retried after restart.

## Verified against the pinned upstream source

The web adapter uses the actual `LeagueConfig`, `Trainer.train(model, df, test_size,
seed)`, `ModelConfig`, `ModelDatabase(league_id)`, `save_model`, `load_model`,
`TargetType.RESULT`/`RESULT_UNDER_OVER`, and `construct_inputs_by_match(league_df,
match_df: Series)` interfaces from the pinned upstream commit. Runtime dependencies
include `optuna` and `optuna-dashboard`, required by the original `ModelDatabase`
import chain, and versions aligned with that commit's requirements.

## Important limits

This is automatic reconstruction, not a byte-for-byte backup. It depends on the
original historical data providers continuing to serve the required seasons and on
the pinned upstream code remaining compatible with them. The first boot after a reset
may take a long time. If a provider removes data or changes its format, recovery can
fail; in that case the dyno reports a startup error rather than pretending the models
are available.

The web UI is still a partial port (league download, training, and 1X2 prediction);
it does not yet expose every original desktop analysis/tuning/agent/export screen.

## GitHub Actions deployment

The workflow `.github/workflows/deploy-heroku.yml` deploys on pushes to `main` or
`master`, and can also be started manually from the GitHub Actions tab.

Add these repository secrets under **Settings → Secrets and variables → Actions**:
- `HEROKU_API_KEY`: Heroku account API key (keep it secret).
- `HEROKU_APP_NAME`: exact Heroku app name.

The Heroku Postgres add-on must already be attached to this same Heroku app so
`DATABASE_URL` is injected automatically. The workflow builds the Docker image,
pushes it to Heroku Container Registry, and releases the `web` process.
