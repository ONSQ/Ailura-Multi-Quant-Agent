from datetime import date

import pytest

from ailura.data.base import LookaheadError, PointInTimeProvider


def test_pit_clips_and_raises(provider):
    pit = PointInTimeProvider(provider, date(2024, 3, 1))
    df = pit.history("AAPL", 50)
    assert df.index.max().date() <= date(2024, 3, 1)
    assert len(df) == 50
    with pytest.raises(LookaheadError):
        pit.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 3, 2))
    with pytest.raises(LookaheadError):
        pit.get_news("AAPL", as_of=date(2024, 3, 5))
    assert all(n.published <= date(2024, 3, 1) for n in pit.get_news("AAPL", lookback_days=7))
    assert pit.get_fundamentals("AAPL").as_of <= date(2024, 3, 1)


def test_pit_lenient_clips(provider):
    pit = PointInTimeProvider(provider, date(2024, 3, 1), strict=False)
    df = pit.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 12, 31))
    assert df.index.max().date() <= date(2024, 3, 1)


def test_synthetic_is_deterministic():
    from ailura.data.synthetic import SyntheticProvider

    a = SyntheticProvider(seed=3).get_ohlcv("NVDA", date(2024, 1, 1), date(2024, 2, 1))
    b = SyntheticProvider(seed=3).get_ohlcv("NVDA", date(2024, 1, 1), date(2024, 2, 1))
    c = SyntheticProvider(seed=4).get_ohlcv("NVDA", date(2024, 1, 1), date(2024, 2, 1))
    assert a.equals(b) and not a.equals(c)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()


def test_csv_provider_roundtrip(tmp_path, provider):
    from ailura.data.csv_provider import CSVProvider

    df = provider.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 3, 1))
    df.rename(columns=str.title).to_csv(tmp_path / "AAPL.csv", index_label="Date")
    (tmp_path / "AAPL_news.jsonl").write_text('{"date": "2024-02-28", "headline": "x", "sentiment": 0.5}\n{"date": "2024-03-05", "headline": "future"}\n')
    csv = CSVProvider(tmp_path)
    out = csv.get_ohlcv("AAPL", date(2024, 2, 1), date(2024, 3, 1))
    assert out.index.min().date() >= date(2024, 2, 1) and "close" in out.columns
    news = csv.get_news("AAPL", date(2024, 3, 1), 7)
    assert [n.headline for n in news] == ["x"]
