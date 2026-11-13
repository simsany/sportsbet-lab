"""End-to-end smoke test for every sport model family using synthetic fixtures only."""
import numpy as np
import pandas as pd
import pytest
import model_engine as engine


def _dataset(sport, n=420, seed=41):
    rng = np.random.default_rng(seed + engine.SPORTS.index(sport))
    teams = [f'{sport}-team-{i:02d}' for i in range(24)]
    rows = []
    for i in range(n):
        home, away = rng.choice(teams, 2, replace=False)
        if sport == 'football':
            hs, aws = rng.poisson(1.55), rng.poisson(1.15)
        elif sport == 'tennis':
            # Mimic the normalizer's alternating winner/loser orientation.
            hs, aws = (2, int(rng.integers(0, 2))) if i % 2 == 0 else (int(rng.integers(0, 2)), 2)
        elif sport == 'basketball':
            hs, aws = rng.normal(112, 13), rng.normal(108, 13)
        elif sport == 'baseball':
            hs, aws = rng.poisson(4.6), rng.poisson(4.1)
        else:
            hs, aws = rng.normal(24, 8), rng.normal(21, 8)
        rows.append((pd.Timestamp('2018-01-01', tz='UTC') + pd.Timedelta(days=i), home, away, max(0, round(hs)), max(0, round(aws))))
    return pd.DataFrame(rows, columns=['date', 'home_team', 'away_team', 'home_score', 'away_score'])


@pytest.mark.parametrize('sport', engine.SPORTS)
def test_full_training_and_prediction_for_each_sport(tmp_path, monkeypatch, sport):
    monkeypatch.setattr(engine, 'DATA', tmp_path / 'processed')
    monkeypatch.setattr(engine, 'ART', tmp_path / 'models')
    monkeypatch.setattr(engine, 'MIN_ROWS', 350)
    monkeypatch.setattr(engine, '_ARTIFACT_CACHE', {})
    engine.DATA.mkdir(parents=True, exist_ok=True)
    df = _dataset(sport)
    df.to_csv(engine.DATA / f'{sport}.csv', index=False)

    result = engine.train_sport(sport)
    assert result['status'] == 'trained', result
    assert set(result['evaluation']) == {'elo', 'poisson', 'xgboost', 'catboost', 'lightgbm'}
    for metric in result['evaluation'].values():
        assert metric['holdout_rows'] >= 30
        assert np.isfinite(metric['log_loss'])
        assert np.isfinite(metric['brier_score'])

    prediction = engine.predict(sport, str(df.iloc[-1].home_team), str(df.iloc[-1].away_team))
    assert prediction is not None
    assert prediction['models_used'] == 5
    probs = np.asarray(prediction['probabilities'])
    assert np.isfinite(probs).all() and np.all(probs >= 0)
    assert abs(probs.sum() - 1.0) < 1e-7


def test_unknown_teams_do_not_get_fabricated_model_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'ART', tmp_path / 'models')
    monkeypatch.setattr(engine, '_ARTIFACT_CACHE', {})
    engine.ART.mkdir(parents=True)
    # No trained artifact: unknown teams must return no model result.
    assert engine.predict('football', 'Unknown FC', 'Unknown United') is None
