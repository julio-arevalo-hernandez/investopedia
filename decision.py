"""
Módulo de Toma de Decisiones.

Aquí se combinan los tres modelos (ARIMA, GARCH, ML) con un puñado de
indicadores técnicos para producir una única señal en formato semáforo
(BUY / HOLD / SELL) y un tamaño de posición.

PONDERACIONES (suma = 1.0):
    - 0.40  → Modelo de Machine Learning (probabilidad de subida)
    - 0.25  → Pronóstico ARIMA (% de cambio esperado)
    - 0.35  → Consenso técnico (RSI, MACD, posición vs Bandas de Bollinger,
              cruce de medias móviles)

GARCH no entra en el voto direccional sino que modula el TAMAÑO de la
posición vía Kelly: si la volatilidad anualizada prevista supera ciertos
umbrales, recortamos la fracción de capital a arriesgar.

UMBRALES DE DECISIÓN (sobre la puntuación combinada en [-1, +1]):
    score >  +0.30  →  BUY
    score <  -0.30  →  SELL
    en otro caso     →  HOLD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from models import ArimaResult, GarchResult, MlResult


WEIGHTS = {"ml": 0.40, "arima": 0.25, "technical": 0.35}
BUY_THRESHOLD = 0.30
SELL_THRESHOLD = -0.30


@dataclass
class DecisionSignal:
    action: str                       # "BUY" / "HOLD" / "SELL"
    score: float                      # puntuación combinada en [-1, +1]
    confidence: float                 # |score|, útil para el frontend
    components: Dict[str, float] = field(default_factory=dict)
    rationale: list = field(default_factory=list)
    suggested_fraction: float = 0.0   # fracción de capital sugerida
    suggested_dollars: float = 0.0    # importe en $ a desplegar


# ---------------------------------------------------------------------------
# Subseñales
# ---------------------------------------------------------------------------


def technical_score(latest: pd.Series) -> tuple[float, list]:
    """Devuelve una puntuación en [-1, 1] a partir de los últimos indicadores."""
    votes = []
    notes = []

    rsi = latest.get("RSI", np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            votes.append(1.0)
            notes.append(f"RSI {rsi:.1f} → sobreventa (alcista)")
        elif rsi > 70:
            votes.append(-1.0)
            notes.append(f"RSI {rsi:.1f} → sobrecompra (bajista)")
        else:
            votes.append((50 - rsi) / 50)  # leve sesgo a la media
            notes.append(f"RSI {rsi:.1f} → neutro")

    hist = latest.get("MACD_Hist", np.nan)
    if not np.isnan(hist):
        votes.append(np.tanh(hist * 5))   # comprime el histograma a [-1, 1]
        notes.append(f"MACD hist {hist:+.3f}")

    pct_b = latest.get("BB_PctB", np.nan)
    if not np.isnan(pct_b):
        if pct_b < 0.2:
            votes.append(0.7)
            notes.append("Precio cerca de la banda inferior → rebote probable")
        elif pct_b > 0.8:
            votes.append(-0.7)
            notes.append("Precio cerca de la banda superior → corrección probable")
        else:
            votes.append(0.0)

    sma_fast = latest.get("SMA_20", np.nan)
    sma_slow = latest.get("SMA_50", np.nan)
    if not (np.isnan(sma_fast) or np.isnan(sma_slow)):
        if sma_fast > sma_slow:
            votes.append(0.4)
            notes.append("SMA20 > SMA50 → tendencia corta alcista")
        else:
            votes.append(-0.4)
            notes.append("SMA20 < SMA50 → tendencia corta bajista")

    if not votes:
        return 0.0, ["Sin indicadores técnicos disponibles"]

    return float(np.clip(np.mean(votes), -1.0, 1.0)), notes


def _arima_score(arima: ArimaResult, horizon_days: int = 1) -> float:
    """Convierte el % esperado de ARIMA en una puntuación acotada.

    El cap se escala con el horizonte: 1 % por día (saturado), de modo que
    una previsión semanal del +5 % satura la puntuación, igual que una
    previsión diaria del +1 %.
    """
    if not arima.success:
        return 0.0
    cap = 0.01 * max(1, horizon_days)
    return float(np.clip(arima.pct_change / cap, -1.0, 1.0))


def _ml_score(ml: MlResult) -> float:
    """Centra la probabilidad alrededor de 0 y la escala."""
    if not ml.success:
        return 0.0
    return float(np.clip((ml.prob_up - 0.5) * 2, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Modulación de tamaño con GARCH
# ---------------------------------------------------------------------------


def _volatility_haircut(garch: GarchResult) -> float:
    """Multiplicador en (0, 1] que recorta el tamaño de la posición.

    - Vol anualizada < 30 %  → sin recorte (1.0)
    - Vol anualizada 30–60 % → recorte progresivo
    - Vol anualizada > 80 %  → recorte severo (0.3)
    """
    if not garch.success or np.isnan(garch.annualized):
        return 0.7  # falta de info ⇒ prudencia

    ann = garch.annualized
    if ann < 0.30:
        return 1.0
    if ann < 0.60:
        return 0.8
    if ann < 0.80:
        return 0.55
    return 0.3


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def build_signal(
    df: pd.DataFrame,
    arima: ArimaResult,
    garch: GarchResult,
    ml: MlResult,
    capital: float,
    kelly_fraction_value: float,
    horizon_days: int = 1,
) -> DecisionSignal:
    """Combina todo y devuelve la decisión final."""
    if df.empty:
        return DecisionSignal(
            action="HOLD", score=0.0, confidence=0.0,
            rationale=["Sin datos suficientes."],
        )

    latest = df.iloc[-1]
    tech, tech_notes = technical_score(latest)
    arima_s = _arima_score(arima, horizon_days=horizon_days)
    ml_s = _ml_score(ml)

    score = (
        WEIGHTS["ml"] * ml_s
        + WEIGHTS["arima"] * arima_s
        + WEIGHTS["technical"] * tech
    )
    score = float(np.clip(score, -1.0, 1.0))

    if score >= BUY_THRESHOLD:
        action = "BUY"
    elif score <= SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "HOLD"

    # Sólo desplegamos capital cuando hay convicción direccional clara.
    haircut = _volatility_haircut(garch)
    fraction = kelly_fraction_value * haircut if action == "BUY" else 0.0

    rationale = list(tech_notes)
    if arima.success:
        rationale.append(
            f"ARIMA prevé {arima.pct_change*100:+.2f}% (precio objetivo "
            f"{arima.forecast:.2f})."
        )
    if ml.success:
        rationale.append(
            f"ML estima P(subida) = {ml.prob_up:.2%} (CV acc {ml.cv_accuracy:.2%})."
        )
    if garch.success:
        rationale.append(
            f"GARCH: σ próx. bar = {garch.sigma*100:.2f}%, "
            f"vol anualizada ≈ {garch.annualized*100:.1f}% → haircut {haircut:.0%}."
        )

    return DecisionSignal(
        action=action,
        score=score,
        confidence=abs(score),
        components={
            "ml": ml_s,
            "arima": arima_s,
            "technical": tech,
        },
        rationale=rationale,
        suggested_fraction=fraction,
        suggested_dollars=fraction * capital,
    )
