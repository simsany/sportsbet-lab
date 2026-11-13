import pandas as pd
from scripts.prepare_training_data import write


def test_normalizer_writes_canonical_schema(tmp_path, monkeypatch):
    import scripts.prepare_training_data as prep
    monkeypatch.setattr(prep, 'OUT', tmp_path)
    source = pd.DataFrame({
        'Date': ['01/08/2024', '02/08/2024', 'bad-date'],
        'Home': ['Alpha', 'Beta', 'Gamma'],
        'Away': ['Beta', 'Alpha', 'Delta'],
        'HG': [2, 1, 0],
        'AG': [1, 1, 0],
    })
    assert prep.write(source, 'football', 'Date', 'Home', 'Away', 'HG', 'AG')
    result = pd.read_csv(tmp_path / 'football.csv')
    assert list(result.columns) == ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    assert len(result) == 2
    assert result.home_score.tolist() == [2, 1]
