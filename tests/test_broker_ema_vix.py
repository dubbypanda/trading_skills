# ABOUTME: Tests for the EMA9/EMA21 + VIX/VXN regime strategy.
# ABOUTME: Covers the vol gate, including refusing to trade without a vol reading.

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from trading_skills.broker import ema_vix
from trading_skills.broker.ema_vix import _vol_fallback, run_ema_vix_strategy

MODULE = "trading_skills.broker.ema_vix"


def _frame(closes):
    return pd.DataFrame({"Close": closes})


class TestVolFallback:
    """The yfinance fallback reports 'no reading' rather than inventing one."""

    def test_returns_prior_close(self):
        with patch(f"{MODULE}.yf.download", return_value=_frame([19.4, 21.7])):
            assert _vol_fallback("^VIX") == pytest.approx(21.7)

    def test_empty_series_returns_none(self):
        with patch(f"{MODULE}.yf.download", return_value=_frame([])):
            assert _vol_fallback("^VIX") is None

    def test_download_failure_returns_none(self):
        with patch(f"{MODULE}.yf.download", side_effect=RuntimeError("network down")):
            assert _vol_fallback("^VIX") is None

    def test_never_returns_a_value_below_the_cutoff_on_failure(self):
        """A fabricated default under the cutoff would silently pass the vol gate."""
        with patch(f"{MODULE}.yf.download", side_effect=RuntimeError("network down")):
            result = _vol_fallback("^VIX")
        assert result is None or result >= ema_vix.DEFAULT_THRESHOLD["VIX"]


def _bars(n=40):
    """A rising 30-min series, enough history for the EMA lookback."""
    start = datetime(2026, 9, 10, 13, 30, tzinfo=UTC)
    return [
        {
            "dt": start + timedelta(minutes=30 * i),
            "open": 100.0 + i,
            "close": 100.5 + i,
        }
        for i in range(n)
    ]


class TestVolGateFailsClosed:
    """No vol reading must block the trade, not wave it through."""

    BARS = _bars()

    def _run(self, *, intraday, prior, source="ib-live"):
        async def fake_fetch_bars(*args, **kwargs):
            return self.BARS, intraday, source

        with (
            patch(f"{MODULE}._fetch_bars", side_effect=fake_fetch_bars),
            patch(f"{MODULE}._vol_fallback", return_value=prior),
        ):
            return asyncio.run(run_ema_vix_strategy("SPX", budget=1000, port=7496))

    def test_missing_prior_day_reading_blocks_the_trade(self):
        result = self._run(intraday=15.0, prior=None)
        assert result["success"] is False
        assert result["signal"] == "VOL-UNAVAILABLE"
        assert result["spread_type"] is None

    def test_missing_intraday_reading_blocks_the_trade(self):
        result = self._run(intraday=None, prior=15.0, source="yfinance-fallback")
        assert result["success"] is False
        assert result["signal"] == "VOL-UNAVAILABLE"
        assert result["spread_type"] is None

    def test_both_readings_missing_blocks_the_trade(self):
        result = self._run(intraday=None, prior=None, source="unavailable")
        assert result["success"] is False
        assert result["signal"] == "VOL-UNAVAILABLE"

    def test_reason_names_the_missing_side(self):
        result = self._run(intraday=15.0, prior=None)
        assert "prior" in result["reason"].lower()

    def test_elevated_vol_still_reports_vix_skip(self):
        """The existing skip path must not be swallowed by the new guard."""
        result = self._run(intraday=25.0, prior=15.0)
        assert result["success"] is False
        assert result["signal"] == "VIX-SKIP"

    def test_readings_present_and_calm_passes_the_vol_gate(self):
        result = self._run(intraday=15.0, prior=16.0)
        assert result["signal"] not in ("VOL-UNAVAILABLE", "VIX-SKIP")
