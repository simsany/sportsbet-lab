import os
import pytest

pytest.importorskip('flask')
os.environ['AUTO_TRAIN'] = '0'
import app as webapp


def test_health_and_frontend_routes(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, 'CONFIG_PATH', tmp_path / 'config.json')
    client = webapp.app.test_client()
    health = client.get('/health')
    assert health.status_code == 200
    assert health.get_json()['status'] == 'ok'
    assert client.get('/').status_code == 200
    assert client.get('/api/models').status_code == 200


def test_config_validation_and_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, 'CONFIG_PATH', tmp_path / 'config.json')
    client = webapp.app.test_client()
    assert client.get('/api/config').status_code == 200
    bad = client.put('/api/config', json={'min_odds': 0.2})
    assert bad.status_code == 400
    good = client.put('/api/config', json={'min_odds': 1.85, 'max_picks': 25})
    assert good.status_code == 200
    saved = client.get('/api/config').get_json()
    assert saved['min_odds'] == 1.85
    assert saved['max_picks'] == 25


def test_scan_fails_closed_without_odds_key(monkeypatch):
    monkeypatch.setattr(webapp, 'API_KEY', '')
    response = webapp.app.test_client().post('/api/scan')
    assert response.status_code == 502
    assert 'ODDS_API_KEY' in response.get_json()['error']
