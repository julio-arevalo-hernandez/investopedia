"""
Módulo de Gestión de Portafolio y Riesgo.

- ``parametric_var``: Value at Risk paramétrico (gaussiano) sobre el capital.
- ``kelly_fraction``: fracción de capital sugerida por el Criterio de Kelly,
  con un *cap* prudencial (Kelly fraccional) para no sobre-apalancarse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------


@dataclass
class VarResult:
    confidence: float       # p.ej. 0.95
    horizon_bars: int       # horizonte temporal en barras
    var_pct: float          # pérdida potencial en % del capital
    var_money: float        # pérdida potencial en unidades monetarias


def parametric_var(
    log_returns: pd.Series,
    capital: float,
    confidence: float = 0.95,
    horizon_bars: int = 1,
) -> VarResult:
    """VaR paramétrico bajo hipótesis de normalidad.

    VaR = capital * (μ·h − z·σ·√h), expresado como pérdida positiva.
    """
    series = log_returns.dropna()
    if series.empty:
        return VarResult(confidence, horizon_bars, 0.0, 0.0)

    mu = float(series.mean())
    sigma = float(series.std(ddof=1))
    z = norm.ppf(confidence)

    # Pérdida potencial esperada en el peor (1-confidence)% del horizonte.
    loss_pct = -(mu * horizon_bars - z * sigma * np.sqrt(horizon_bars))
    loss_pct = max(loss_pct, 0.0)
    return VarResult(
        confidence=confidence,
        horizon_bars=horizon_bars,
        var_pct=loss_pct,
        var_money=loss_pct * capital,
    )


# ---------------------------------------------------------------------------
# Criterio de Kelly
# ---------------------------------------------------------------------------


@dataclass
class KellyResult:
    full_kelly: float       # f* sin cap
    fractional_kelly: float # cap aplicado, lo que se recomienda usar
    win_prob: float
    win_loss_ratio: float


def kelly_fraction(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    cap: float = 0.25,
    use_fractional: float = 0.5,
) -> KellyResult:
    """Calcula el porcentaje de capital a arriesgar.

    Fórmula clásica:    f* = p − (1 − p) / b   donde   b = avg_win / avg_loss.

    Aplicamos *Half-Kelly* por defecto (use_fractional=0.5) y un cap del 25 %
    porque Kelly puro asume probabilidades conocidas, lo cual nunca se cumple
    en mercados reales.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return KellyResult(0.0, 0.0, win_prob, 0.0)

    b = avg_win / avg_loss
    f_star = win_prob - (1 - win_prob) / b
    f_star = max(f_star, 0.0)         # nunca apostamos en contra
    capped = min(f_star * use_fractional, cap)
    return KellyResult(
        full_kelly=f_star,
        fractional_kelly=capped,
        win_prob=win_prob,
        win_loss_ratio=b,
    )


def estimate_win_loss(log_returns: pd.Series) -> tuple[float, float, float]:
    """Estima (p, avg_win, avg_loss) a partir del histórico reciente.

    Sirve para alimentar Kelly cuando no tenemos un track record propio.
    """
    series = log_returns.dropna()
    if series.empty:
        return 0.5, 0.0, 0.0

    wins = series[series > 0]
    losses = series[series < 0]
    if wins.empty or losses.empty:
        return 0.5, 0.0, 0.0

    p = len(wins) / (len(wins) + len(losses))
    avg_win = float(wins.mean())
    avg_loss = float(-losses.mean())
    return p, avg_win, avg_loss
