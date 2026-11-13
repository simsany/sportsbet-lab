# Training data sources

The downloader normalizes completed fixtures into `data/processed/<sport>.csv` with the columns:

`date,home_team,away_team,home_score,away_score`

- Football: https://www.football-data.co.uk/ — results from selected European leagues/seasons.
- Tennis: https://huggingface.co/datasets/minmaxxxer/atp-wta-matches-with-odds — ATP/WTA match results; verify dataset card and terms before commercial use.
- Basketball: https://huggingface.co/datasets/hamzas/nba-games — NBA game history; verify current schema and license.
- Baseball: https://statsapi.mlb.com/api/v1/schedule — completed MLB regular-season schedule/results.
- NFL: https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv — schedules and final scores.

The downloader skips unavailable/invalid sources and prints warnings. It does not synthesize missing observations. Source schemas and terms may change; the application reports insufficient data rather than fabricating a model.
