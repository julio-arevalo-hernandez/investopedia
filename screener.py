"""
Screener bursátil de dos etapas.

Etapa 1 — RÁPIDA. Bulk-download diario de cientos de tickers en una sola
llamada a yfinance y scoring por momentum / tendencia / RSI / vol /
liquidez. Filtra el universo a un top-N de candidatos.

Etapa 2 — LENTA. El pipeline completo (ARIMA + GARCH + ML + plan de trade)
se ejecuta sólo sobre los finalistas. Los mejores se rankean por Sharpe
esperado y se proyectan contra el objetivo diario del usuario.

El propósito es maximizar el ratio de ganancias diario explorando todo
el caladero de Yahoo en lugar de limitarse a la watchlist manual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from data import rsi as _rsi


@dataclass
class Stage1Score:
    ticker: str
    price: float
    ret_1d: float
    ret_5d: float
    ret_20d: float
    vol_20d_ann: float
    avg_dollar_vol: float
    rsi_14: float
    above_sma20: bool
    above_sma50: bool
    composite: float


def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - x.mean()) / sd


def stage1_screen(
    tickers: List[str],
    period: str = "3mo",
    min_dollar_volume: float = 5_000_000.0,
    min_price: float = 5.0,
) -> pd.DataFrame:
    """Cribado rápido sobre todo el universo.

    Score compuesto (todo entre [-∞, +∞], pero z-normalizado dentro del lote):
        +0.40 · z(retorno 5d)        ← momentum corto, lo más predictivo intradía
        +0.20 · z(retorno 20d)       ← momentum medio
        +0.20 · trend(SMA20/SMA50)   ← confirmación de tendencia
        +0.10 · rsi_score            ← penaliza tanto el sobre-comprado como el moribundo
        −0.10 · z(|vol−30%|)         ← preferimos volatilidad sana, no errática

    Filtros duros: precio > min_price y dollar-volume > min_dollar_volume
    para descartar nombres ilíquidos (donde el slippage destroza el edge).
    """
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True,
        auto_adjust=True,
    )

    # yf.download con un único ticker no devuelve MultiIndex.
    multi = isinstance(raw.columns, pd.MultiIndex) and len(tickers) > 1

    rows: List[Stage1Score] = []
    for t in tickers:
        try:
            sub = raw[t] if multi else raw
            sub = sub.dropna()
            if len(sub) < 30:
                continue

            close = sub["Close"]
            vol = sub["Volume"]
            last = float(close.iloc[-1])
            if last < min_price:
                continue

            avg_dv = float((close * vol).tail(20).mean())
            if not np.isfinite(avg_dv) or avg_dv < min_dollar_volume:
                continue

            ret_1d = float(close.pct_change(1).iloc[-1])
            ret_5d = float(close.pct_change(5).iloc[-1])
            ret_20d = float(close.pct_change(20).iloc[-1])

            sd = float(close.pct_change().tail(20).std(ddof=1))
            vol_20d_ann = sd * np.sqrt(252) if np.isfinite(sd) else 0.0

            sma20 = float(close.rolling(20).mean().iloc[-1])
            sma50 = (
                float(close.rolling(50).mean().iloc[-1])
                if len(close) >= 50 else sma20
            )
            rsi14 = float(_rsi(close).iloc[-1])

            rows.append(
                Stage1Score(
                    ticker=t, price=last,
                    ret_1d=ret_1d, ret_5d=ret_5d, ret_20d=ret_20d,
                    vol_20d_ann=vol_20d_ann, avg_dollar_vol=avg_dv,
                    rsi_14=rsi14,
                    above_sma20=last > sma20, above_sma50=last > sma50,
                    composite=0.0,
                )
            )
        except Exception:
            # Tickers retirados, fusionados o con datos rotos: los saltamos.
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in rows])

    # rsi_score: 1.0 cuando RSI=50 (momentum saludable), 0 en extremos.
    rsi_score = 1.0 - (df["rsi_14"].fillna(50) - 50).abs() / 50.0
    rsi_score = rsi_score.clip(lower=0.0, upper=1.0)

    trend = df["above_sma20"].astype(int) * 0.6 + df["above_sma50"].astype(int) * 0.4

    df["composite"] = (
        0.40 * _zscore(df["ret_5d"])
        + 0.20 * _zscore(df["ret_20d"])
        + 0.20 * trend
        + 0.10 * rsi_score
        - 0.10 * _zscore((df["vol_20d_ann"] - 0.30).abs())
    )

    return df.sort_values("composite", ascending=False).reset_index(drop=True)


def select_finalists(scored: pd.DataFrame, n: int = 15) -> List[str]:
    """Top N tickers por score compuesto."""
    if scored.empty:
        return []
    return scored.head(n)["ticker"].tolist()
