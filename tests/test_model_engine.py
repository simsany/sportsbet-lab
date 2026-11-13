import numpy as np
import pandas as pd
from model_engine import _features, _elo_probs, _poisson_grid, _softmax_temperature, _known_key


def sample_frame():
    return pd.DataFrame([
        {'date':'2024-01-01T00:00:00Z','home_team':'A','away_team':'B','home_score':2,'away_score':0},
        {'date':'2024-01-03T00:00:00Z','home_team':'B','away_team':'A','home_score':1,'away_score':1},
        {'date':'2024-01-05T00:00:00Z','home_team':'A','away_team':'C','home_score':3,'away_score':1},
    ])


def test_features_are_pre_match_and_finite():
    x, ratings, history, last = _features(sample_frame())
    assert x.shape == (3, 14)
    assert np.isfinite(x).all()
    # First event starts from neutral Elo and no historical form.
    assert x[0, 0] == 0
    assert x[0, 3] == 0
    # Second event sees the first event's result, not its own result.
    assert x[1, 9] == 1
    assert x[1, 10] == 1


def test_sport_probability_shapes_and_sums():
    football = _poisson_grid(1.5, 1.1, 'football', .03)
    baseball = _poisson_grid(4.5, 4.0, 'baseball')
    basketball = _poisson_grid(5.0, 0, 'basketball', margin_sd=12)
    assert football.shape == (3,)
    assert baseball.shape == (2,)
    assert basketball.shape == (2,)
    for p in (football, baseball, basketball):
        assert np.isfinite(p).all()
        assert np.all(p >= 0)
        assert abs(p.sum()-1) < 1e-8


def test_temperature_calibration_preserves_probability_distribution():
    p = np.array([[.7,.3],[.2,.8]])
    q = _softmax_temperature(p, 1.4)
    assert q.shape == p.shape
    assert np.allclose(q.sum(axis=1), 1)


def test_team_name_matching_is_exact_normalized_only():
    assert _known_key('Los Angeles Lakers', ['Los Angeles Lakers','Boston Celtics']) == 'Los Angeles Lakers'
    assert _known_key('los-angeles lakers', ['Los Angeles Lakers']) == 'Los Angeles Lakers'
    assert _known_key('Lakers', ['Los Angeles Lakers']) is None
