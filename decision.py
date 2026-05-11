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
    score >  +BUY_THRESHOLD   →  BUY
    score <  -BUY_THRESHOLD   →  SELL
    en otro caso              →  HOLD

El umbral por defecto es 0.15 (modo equilibrado). Se puede modificar
desde la UI con el selector de sensibilidad:
    - Conservador: 0.30 — sólo BUY con consenso muy fuerte.
    - Equilibrado: 0.15 — recomendaciones moderadas (default).
    - Agresivo:    0.05 — cualquier sesgo positivo cuenta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from models import ArimaResult, GarchResult, MlResult


WEIGHTS = {"ml": 0.40, "arima": 0.25, "technical": 0.35}
BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15

SENSITIVITY_PRESETS: Dict[str, float] = {
    "Conservador": 0.30,
    "Equilibrado": 0.15,
    "Agresivo":    0.05,
}


@dataclass
class DecisionSignal:
    action: str                       # "BUY" / "HOLD" / "SELL"
    score: float                      # puntuación combinada en [-1, +1]
    confidence: float                 # |score|, útil para el frontend
    components: Dict[str, float] = field(default_factory=dict)
    rationale: list = field(default_factory=list)
    suggested_fraction: float = 0.0   # fracción de capital sugerida
    suggested_dollars: float = 0.0    # importe en $ a desplegar
    block_reason: str = ""            # por qué no es BUY (vacío si lo es)
    threshold: float = BUY_THRESHOLD  # umbral activo cuando se evaluó


# ---------------------------------------------------------------------------
# Subseñales
# ---------------------------------------------------------------------------


def technical_score(latest: pd.Series) -> Tuple[float, list]:
    """Devuelve una puntuación en [-1, 1] a partir de los últimos indicadores.

    Sin sesgo anti-momentum: RSI en zona 40-70 NO penaliza (es momentum
    saludable en una tendencia alcista). Sólo los extremos (RSI<30 ó >75)
    aportan votos fuertes. Los componentes que no tienen señal clara
    quedan fuera del promedio en lugar de votar 0 (lo cual sólo lo
    diluían).
    """
    votes = []
    notes = []

    close_now = float(latest.get("Close", np.nan))

    # ---- Tendencia (SMA20 vs SMA50) ----
    # Se mira primero porque modula otros votos.
    sma_fast = latest.get("SMA_20", np.nan)
    sma_slow = latest.get("SMA_50", np.nan)
    in_uptrend = False
    if not (np.isnan(sma_fast) or np.isnan(sma_slow)):
        if sma_fast > sma_slow:
            in_uptrend = True
            votes.append(0.6)
            notes.append("SMA20 > SMA50 → tendencia corta alcista")
        else:
            votes.append(-0.6)
            notes.append("SMA20 < SMA50 → tendencia corta bajista")
        # Bonus si el precio está por encima de SMA20 (momentum confirmado)
        if not np.isnan(close_now) and not np.isnan(sma_fast):
            if close_now > sma_fast:
                votes.append(0.3)
            else:
                votes.append(-0.3)

    # ---- RSI ----
    # Sólo los extremos cuentan; la zona media no penaliza el momentum sano.
    rsi = latest.get("RSI", np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            votes.append(1.0)
            notes.append(f"RSI {rsi:.1f} → sobreventa (alcista)")
        elif rsi > 75:
            votes.append(-1.0)
            notes.append(f"RSI {rsi:.1f} → sobrecompra extrema")
        elif rsi > 70:
            votes.append(-0.3)
            notes.append(f"RSI {rsi:.1f} → sobrecompra moderada")
        elif 45 <= rsi <= 65:
            # Zona de momentum saludable: pequeño voto positivo si hay tendencia.
            v = 0.3 if in_uptrend else 0.0
            votes.append(v)
            notes.append(f"RSI {rsi:.1f} → momentum saludable")
        else:
            # 30-45 zona de acumulación; suave bullish
            votes.append(0.1)
            notes.append(f"RSI {rsi:.1f} → acumulación")

    # ---- MACD ----
    # Normalizado por precio para no dar señales más fuertes en acciones caras.
    hist = latest.get("MACD_Hist", np.nan)
    if not np.isnan(hist) and not np.isnan(close_now) and close_now > 0:
        hist_norm = hist / close_now              # histograma como fracción del precio
        v = float(np.tanh(hist_norm * 500))       # saturación en ±0.2 % del precio
        votes.append(v)
        notes.append(f"MACD hist (norm) → voto {v:+.2f}")

    # ---- Bollinger %B ----
    pct_b = latest.get("BB_PctB", np.nan)
    if not np.isnan(pct_b):
        if pct_b < 0.15:
            votes.append(0.8)
            notes.append("Precio bajo BB inferior → rebote probable")
        elif pct_b > 0.85:
            votes.append(-0.8)
            notes.append("Precio sobre BB superior → corrección probable")
        elif 0.55 <= pct_b <= 0.75:
            # Subiendo en zona alta = momentum, no agotamiento
            votes.append(0.2 if in_uptrend else -0.1)
        # zona 0.15-0.55 no aporta voto (en lugar de votar 0 y diluir)

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
    buy_threshold: float = BUY_THRESHOLD,
    risk_per_trade_pct: float = 0.02,
) -> DecisionSignal:
    """Combina todo y devuelve la decisión final.

    El tamaño de la posición ya NO depende sólo de Kelly (que sobre series
    históricas tiende a 0 y bloquea todo). Se usa un riesgo fijo por
    operación (``risk_per_trade_pct``, por defecto 2 % del capital) como
    presupuesto base, modulado por:
      - GARCH haircut (vol alta → exposición menor)
      - Kelly como tope superior (si el edge histórico es fuerte, lo
        respetamos; si es 0, no bloquea, simplemente no añade convicción)
    """
    if df.empty:
        return DecisionSignal(
            action="HOLD", score=0.0, confidence=0.0,
            rationale=["Sin datos suficientes."],
            block_reason="Sin datos.",
            threshold=buy_threshold,
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
    sell_threshold = -abs(buy_threshold)

    if score >= buy_threshold:
        action = "BUY"
    elif score <= sell_threshold:
        action = "SELL"
    else:
        action = "HOLD"

    # ----- Dimensionado de la posición -----
    haircut = _volatility_haircut(garch)
    if action == "BUY":
        # Base: riesgo fijo por trade, escalado por convicción del score.
        # score=buy_threshold → conviction=0, score=1 → conviction=1
        denom = max(1.0 - buy_threshold, 1e-6)
        conviction = float(np.clip((score - buy_threshold) / denom, 0.0, 1.0))
        # Fracción base entre risk_per_trade y 4×risk_per_trade según convicción
        base_fraction = risk_per_trade_pct * (1.0 + 3.0 * conviction)
        # Kelly histórico como tope opcional: si supera la base, la elevamos
        # un poco; nunca la bloqueamos en cero.
        kelly_boost = max(kelly_fraction_value, base_fraction)
        fraction = min(kelly_boost, 0.25) * haircut  # cap 25 % por trade
    else:
        fraction = 0.0

    # ----- Razonamiento y diagnóstico -----
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
            f"GARCH: σ = {garch.sigma*100:.2f}%, "
            f"vol anualizada ≈ {garch.annualized*100:.1f}% → haircut {haircut:.0%}."
        )

    # Diagnóstico explícito: ¿por qué no es BUY?
    block = ""
    if action != "BUY":
        gap = buy_threshold - score
        contribs = {
            "ML":      WEIGHTS["ml"] * ml_s,
            "ARIMA":   WEIGHTS["arima"] * arima_s,
            "Técnico": WEIGHTS["technical"] * tech,
        }
        worst = min(contribs.items(), key=lambda kv: kv[1])
        if action == "SELL":
            block = f"Bajista: score {score:+.2f} ≤ -{abs(buy_threshold):.2f}."
        else:
            block = (
                f"Score {score:+.2f} a {gap:+.2f} del umbral {buy_threshold:+.2f}. "
                f"Componente más débil: {worst[0]} aporta {worst[1]:+.2f}."
            )

    return DecisionSignal(
        action=action,
        score=score,
        confidence=abs(score),
        components={"ml": ml_s, "arima": arima_s, "technical": tech},
        rationale=rationale,
        suggested_fraction=fraction,
        suggested_dollars=fraction * capital,
        block_reason=block,
        threshold=buy_threshold,
    )
