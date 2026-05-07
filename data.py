"""
Módulo de Ingesta de Datos.

Descarga precios desde yfinance y calcula los indicadores técnicos
que alimentarán el resto del motor (RSI, MACD, Bandas de Bollinger,
Medias Móviles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# yfinance limita el histórico para datos intradía. Esta tabla evita
# pedir rangos imposibles y caer en respuestas vacías.
_MAX_PERIOD_BY_INTERVAL = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h": "730d",
    "1d": "5y",
}


@dataclass
class MarketData:
    ticker: str
    interval: str
    prices: pd.DataFrame  # OHLCV + indicadores
    last_price: float


def _safe_period(interval: str, requested: Optional[str]) -> str:
    """Devuelve el período más cercano que yfinance acepta para el intervalo."""
    cap = _MAX_PERIOD_BY_INTERVAL.get(interval, "60d")
    return requested or cap


def download_prices(
    ticker: str,
    interval: str = "5m",
    period: Optional[str] = None,
) -> pd.DataFrame:
    """Descarga OHLCV desde Yahoo Finance.

    El DataFrame resultante tiene columnas: Open, High, Low, Close, Volume.
    El índice es DatetimeIndex localizado en UTC y luego convertido a naive
    para simplificar las operaciones aguas abajo.
    """
    period = _safe_period(interval, period)
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        return df

    # yfinance puede devolver MultiIndex cuando se pide un único ticker;
    # lo aplanamos para trabajar siempre con columnas escalares.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Close"])

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI clásico de Wilder."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD con línea de señal e histograma."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Hist": hist}
    )


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger."""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    # %B: posición relativa dentro de las bandas (0 = banda baja, 1 = banda alta)
    pct_b = (series - lower) / (upper - lower)
    return pd.DataFrame(
        {"BB_Mid": mid, "BB_Upper": upper, "BB_Lower": lower, "BB_PctB": pct_b}
    )


def moving_averages(
    series: pd.Series, windows=(20, 50, 200)
) -> pd.DataFrame:
    """Medias móviles simples."""
    return pd.DataFrame({f"SMA_{w}": series.rolling(w).mean() for w in windows})


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade RSI, MACD, Bollinger, SMAs y log-retornos al DataFrame OHLCV."""
    if df.empty:
        return df

    out = df.copy()
    close = out["Close"]

    out["LogReturn"] = np.log(close / close.shift(1))
    out["RSI"] = rsi(close)
    out = out.join(macd(close))
    out = out.join(bollinger(close))
    out = out.join(moving_averages(close, windows=(20, 50)))
    # Cruce rápido vs lento como feature binaria útil para el ML
    out["MA_Cross"] = (out["SMA_20"] > out["SMA_50"]).astype(int)
    return out


def load_market_data(ticker: str, interval: str = "5m") -> MarketData:
    """Atajo: descarga + indicadores en un solo paso."""
    raw = download_prices(ticker, interval=interval)
    enriched = add_indicators(raw)
    last_price = float(enriched["Close"].iloc[-1]) if not enriched.empty else float("nan")
    return MarketData(
        ticker=ticker, interval=interval, prices=enriched, last_price=last_price
    )
