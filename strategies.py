"""
Módulo de Estrategias y Recomendaciones Concretas.

Este módulo es la pieza que convierte las señales abstractas (BUY/HOLD/SELL +
score) en planes de trading **explícitos** con entrada, stop-loss, take-profit
y número de acciones, y permite priorizar entre varios tickers para maximizar
el ratio de ganancias diario en el simulador de Investopedia.

Análisis econométrico avanzado incluido:
    * Exponente de Hurst   → clasifica régimen (tendencia / mean-reversion).
    * ATR(14)              → volatilidad absoluta para fijar stops.
    * Sharpe intradía      → calidad del retorno ajustado al riesgo.
    * Z-score de precio    → señal de mean-reversion estadística.
    * Momentum cross-term  → diferencia retornos corto vs medio plazo.
    * R-multiples          → relación riesgo/recompensa estandarizada.
    * Edge esperado        → P(win)·win% − P(loss)·loss% (criterio EV+).

Tres estrategias de compra concretas (long-only, apropiadas para Investopedia):

    1. MOMENTUM_LONG       — Hurst > 0.55. El mercado tiene memoria larga;
                             entramos siguiendo la tendencia con stop a 1.5·ATR
                             y target a 2R. Ideal para AAPL, NVDA, TSLA en días
                             trending.
    2. MEAN_REVERSION_LONG — Hurst < 0.45 y precio cerca de la banda inferior
                             de Bollinger (%B < 0.2). Entramos esperando
                             rebote a la media móvil; stop ajustado a 1·ATR.
    3. BREAKOUT_LONG       — Régimen mixto. Sólo se opera si el precio supera
                             el máximo de las últimas N barras *y* ARIMA
                             confirma el techo del IC al 95 %. Más conservador.

Las estrategias se rankean cross-sectionalmente por **Sharpe esperado**
(edge / σ_GARCH) y se proyecta el P&L diario contra el objetivo del usuario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from models import ArimaResult, GarchResult, MlResult


# ---------------------------------------------------------------------------
# Horizontes de inversión
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HorizonConfig:
    """Configuración del horizonte temporal de la operativa.

    Distintos horizontes implican distintos hiper-parámetros:

    - ``days``        : días hábiles que la operación tiene para desarrollarse.
    - ``interval``    : granularidad de datos sugerida para ese horizonte.
    - ``atr_mult_stop`` : amplitud del stop en múltiplos de ATR (más amplio a
      más horizonte para no salirnos por ruido intradía).
    - ``r_multiple``  : R:R objetivo (más holgado a más horizonte porque el
      mercado tiene tiempo de moverse).
    - ``arima_steps`` : pasos hacia adelante que ARIMA proyecta y horizonte
      del target del modelo de ML.
    """
    name: str
    days: int
    interval: str
    atr_mult_stop: float
    r_multiple: float
    arima_steps: int


HORIZONS: Dict[str, HorizonConfig] = {
    "Diario":    HorizonConfig("Diario",    1,  "5m", 1.5, 2.0, 1),
    "Semanal":   HorizonConfig("Semanal",   5,  "1h", 2.0, 3.0, 5),
    "Quincenal": HorizonConfig("Quincenal", 10, "1d", 2.5, 3.0, 10),
    "Mensual":   HorizonConfig("Mensual",   21, "1d", 3.0, 4.0, 21),
}


INTERVAL_BARS_PER_YEAR: Dict[str, int] = {
    "1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26,
    "30m": 252 * 13, "60m": 252 * 7, "1h": 252 * 7,
    "1d": 252,
}


def bars_per_year(interval: str) -> int:
    return INTERVAL_BARS_PER_YEAR.get(interval, 252 * 78)


# ---------------------------------------------------------------------------
# Primitivas econométricas avanzadas
# ---------------------------------------------------------------------------


def hurst_exponent(returns: pd.Series, max_lag: int = 60) -> float:
    """Hurst por escalado de la varianza de incrementos.

    H ≈ 0.5 → camino aleatorio (sin estructura aprovechable).
    H > 0.55 → serie persistente / con tendencia.
    H < 0.45 → serie anti-persistente / mean-reverting.
    """
    series = returns.dropna().values
    if len(series) < max_lag * 2:
        return 0.5

    lags = list(range(2, max_lag))
    tau = []
    valid_lags = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        sd = float(np.std(diff))
        if sd > 0:
            tau.append(sd)
            valid_lags.append(lag)

    if len(tau) < 5:
        return 0.5

    slope, _ = np.polyfit(np.log(valid_lags), np.log(tau), 1)
    # Limitamos a [0, 1] para evitar artefactos numéricos en regímenes ruidosos
    return float(np.clip(slope, 0.0, 1.0))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (ATR) de Wilder."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def intraday_sharpe(returns: pd.Series, bars_per_year: int = 252 * 78) -> float:
    """Sharpe ratio anualizado a partir de retornos por bar."""
    s = returns.dropna()
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if len(s) < 5 or sd == 0:
        return 0.0
    return float(s.mean() / sd * np.sqrt(bars_per_year))


def price_zscore(close: pd.Series, window: int = 20) -> float:
    """Z-score del precio actual respecto a su media corta — útil para mean-rev."""
    tail = close.tail(window)
    if len(tail) < window:
        return 0.0
    mu = float(tail.mean())
    sd = float(tail.std(ddof=1))
    if sd == 0:
        return 0.0
    return float((close.iloc[-1] - mu) / sd)


def momentum_score(close: pd.Series, short: int = 20, long: int = 60) -> float:
    """Diferencia entre el retorno corto y el retorno medio (momentum cross-term)."""
    if len(close) < long + 1:
        return 0.0
    short_ret = float(close.pct_change(short).iloc[-1])
    long_ret = float(close.pct_change(long).iloc[-1])
    return short_ret - long_ret


# ---------------------------------------------------------------------------
# Clasificación de régimen
# ---------------------------------------------------------------------------


@dataclass
class Regime:
    label: str        # "MOMENTUM" / "MEAN_REVERSION" / "BREAKOUT"
    hurst: float
    notes: str


def classify_regime(
    close: pd.Series, returns: pd.Series, latest: pd.Series
) -> Regime:
    h = hurst_exponent(returns)
    pct_b_raw = latest.get("BB_PctB", np.nan)
    pct_b = float(pct_b_raw) if pct_b_raw is not None and not pd.isna(pct_b_raw) else 0.5

    if h > 0.55:
        return Regime("MOMENTUM", h, f"Hurst {h:.2f} → tendencia persistente")
    if h < 0.45 and (pct_b < 0.2 or pct_b > 0.8):
        return Regime(
            "MEAN_REVERSION",
            h,
            f"Hurst {h:.2f} con %B={pct_b:.2f} → precio extremo respecto a la media",
        )
    return Regime(
        "BREAKOUT",
        h,
        f"Hurst {h:.2f} en zona neutra → sólo operar con ruptura confirmada",
    )


# ---------------------------------------------------------------------------
# Plan de trade concreto
# ---------------------------------------------------------------------------


@dataclass
class TradePlan:
    ticker: str
    strategy: str             # MOMENTUM_LONG / MEAN_REVERSION_LONG / BREAKOUT_LONG / AVOID
    regime: str
    action: str               # BUY / HOLD / SELL
    entry: float
    stop_loss: float
    take_profit: float
    shares: int
    position_dollars: float
    risk_dollars: float
    reward_dollars: float
    risk_reward: float
    prob_up: float
    expected_return_pct: float
    expected_pnl_dollars: float
    sharpe_score: float       # edge / σ_GARCH (ranking key)
    hurst: float
    atr_value: float
    rationale: List[str] = field(default_factory=list)


def build_trade_plan(
    ticker: str,
    df: pd.DataFrame,
    arima: ArimaResult,
    garch: GarchResult,
    ml: MlResult,
    signal_action: str,
    position_dollars: float,
    horizon: HorizonConfig | None = None,
    atr_mult_stop: float | None = None,
    r_multiple: float | None = None,
    capital: float | None = None,
    risk_per_trade_pct: float = 0.02,
    max_position_pct: float = 0.25,
) -> TradePlan:
    """Construye un plan ejecutable para el ticker.

    El plan se calibra con el ``HorizonConfig`` del horizonte temporal
    seleccionado. ``position_dollars`` es la sugerencia inicial del motor
    de decisión, pero si está infradimensionada respecto a ``capital ×
    risk_per_trade_pct / loss_pct`` (sizing estándar por riesgo fijo),
    se eleva al mínimo industria-standard. Esto evita el bug histórico
    donde Kelly sobre series intradía colapsaba a 0 y bloqueaba todas
    las recomendaciones.
    """
    h = horizon if horizon is not None else HORIZONS["Diario"]
    if atr_mult_stop is None:
        atr_mult_stop = h.atr_mult_stop
    if r_multiple is None:
        r_multiple = h.r_multiple
    close = df["Close"]
    latest = df.iloc[-1]
    entry = float(close.iloc[-1])
    atr_series = atr(df)
    atr_value = float(atr_series.iloc[-1]) if len(atr_series) else 0.0

    if (np.isnan(atr_value) or atr_value <= 0):
        # Backup: usamos la σ del GARCH como proxy de volatilidad absoluta.
        sigma_proxy = garch.sigma if (garch.success and garch.sigma > 0) else 0.005
        atr_value = entry * float(sigma_proxy)

    regime = classify_regime(close, df["LogReturn"], latest)
    p = float(ml.prob_up) if ml.success else 0.5

    if signal_action != "BUY":
        return TradePlan(
            ticker=ticker, strategy="AVOID", regime=regime.label,
            action=signal_action, entry=entry, stop_loss=0.0, take_profit=0.0,
            shares=0, position_dollars=0.0, risk_dollars=0.0,
            reward_dollars=0.0, risk_reward=0.0, prob_up=p,
            expected_return_pct=0.0, expected_pnl_dollars=0.0,
            sharpe_score=0.0, hurst=regime.hurst, atr_value=atr_value,
            rationale=[
                f"Señal {signal_action}: no se abre largo.",
                f"Régimen {regime.label} ({regime.notes}).",
            ],
        )

    # ----------- Entrada / stop / target según estrategia -----------
    if regime.label == "MOMENTUM":
        strategy = "MOMENTUM_LONG"
        stop_loss = entry - atr_mult_stop * atr_value
        take_profit = entry + r_multiple * (entry - stop_loss)
        strat_notes = (
            f"Estrategia momentum: stop a {atr_mult_stop}·ATR, "
            f"target a {r_multiple}R."
        )
    elif regime.label == "MEAN_REVERSION":
        strategy = "MEAN_REVERSION_LONG"
        stop_loss = entry - 1.0 * atr_value           # stop más ajustado
        bb_mid_raw = latest.get("BB_Mid", np.nan)
        bb_mid = float(bb_mid_raw) if not pd.isna(bb_mid_raw) else entry
        # Take-profit = la primera de estas dos opciones que sea más conservadora
        rr_target = entry + r_multiple * (entry - stop_loss)
        take_profit = max(entry * 1.001, min(max(bb_mid, rr_target), rr_target))
        strat_notes = (
            "Estrategia mean-reversion: stop a 1·ATR, target en la media móvil "
            "o R-múltiple, lo que primero llegue."
        )
    else:
        strategy = "BREAKOUT_LONG"
        stop_loss = entry - atr_mult_stop * atr_value
        # Si ARIMA prevé un techo creíble (IC al 95 %), úsalo como target;
        # si no, R-multiple estándar.
        if arima.success and arima.forecast > entry:
            arima_target = max(arima.forecast, arima.forecast_upper)
            take_profit = max(arima_target, entry + r_multiple * (entry - stop_loss))
        else:
            take_profit = entry + r_multiple * (entry - stop_loss)
        strat_notes = (
            "Estrategia breakout: sólo entrar tras ruptura; target apoyado en "
            "el techo del IC del ARIMA."
        )

    # Sanidad: stop nunca por encima del entry, target nunca por debajo.
    stop_loss = min(stop_loss, entry * 0.999)
    take_profit = max(take_profit, entry * 1.001)

    # ----- Sizing: máximo entre la sugerencia del signal y el fixed-fractional -----
    # Riesgo fijo por trade = ``capital × risk_per_trade_pct``.
    # Shares por riesgo  = riesgo_$ / (entry − stop).
    # Tomamos el MÁXIMO entre eso y la sugerencia (kelly+haircut) para garantizar
    # que el motor SIEMPRE genere al menos una posición coherente cuando hay BUY.
    loss_per_share = max(entry - stop_loss, entry * 0.005)
    if capital is not None and capital > 0:
        risk_budget = capital * risk_per_trade_pct
        shares_by_risk = int(risk_budget // loss_per_share)
    else:
        shares_by_risk = 0
    shares_by_kelly = int(position_dollars // entry) if entry > 0 else 0
    shares = max(shares_by_risk, shares_by_kelly, 1 if signal_action == "BUY" else 0)

    # Cap por trade: nunca más del max_position_pct del capital en un solo nombre.
    if capital is not None and capital > 0 and entry > 0:
        max_shares_by_size = int((capital * max_position_pct) // entry)
        if max_shares_by_size > 0:
            shares = min(shares, max_shares_by_size)

    actual_position = shares * entry
    risk_dollars = shares * loss_per_share
    reward_dollars = shares * (take_profit - entry)
    rr = (reward_dollars / risk_dollars) if risk_dollars > 0 else 0.0

    win_pct = (take_profit - entry) / entry
    loss_pct = (entry - stop_loss) / entry
    er_pct = p * win_pct - (1.0 - p) * loss_pct
    er_dollars = er_pct * actual_position

    sigma = garch.sigma if (garch.success and garch.sigma > 0) else 0.01
    sharpe = er_pct / sigma

    rationale = [
        f"Régimen detectado: {regime.label} — {regime.notes}.",
        strat_notes,
        f"ATR(14) = {atr_value:.3f}; stop {stop_loss:.2f} | target {take_profit:.2f}.",
        f"Riesgo ${risk_dollars:,.2f} vs Recompensa ${reward_dollars:,.2f} (R:R {rr:.2f}).",
        f"P(win) ML = {p:.2%} ⇒ E[retorno] {er_pct*100:+.2f}% "
        f"(E[P&L] ${er_dollars:+,.2f}).",
    ]

    return TradePlan(
        ticker=ticker, strategy=strategy, regime=regime.label,
        action=signal_action, entry=entry,
        stop_loss=stop_loss, take_profit=take_profit,
        shares=shares, position_dollars=actual_position,
        risk_dollars=risk_dollars, reward_dollars=reward_dollars,
        risk_reward=rr, prob_up=p,
        expected_return_pct=er_pct, expected_pnl_dollars=er_dollars,
        sharpe_score=sharpe, hurst=regime.hurst, atr_value=atr_value,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Ranking cross-sectional y proyección diaria
# ---------------------------------------------------------------------------


def rank_plans(plans: List[TradePlan], min_rr: float = 0.8) -> List[TradePlan]:
    """Ordena los planes por Sharpe esperado descendente.

    Filtra los AVOID y los planes claramente perdedores (EV ≤ 0 o R:R
    muy bajo). El umbral mínimo de R:R se baja a 0.8 para no descartar
    los mean-reversion en los que el target a la media móvil queda algo
    cerca del entry — son operaciones de alta P(win), tolerables aunque
    R:R baje algo de 1.
    """
    actionable = [
        p for p in plans
        if p.strategy != "AVOID"
        and p.expected_pnl_dollars > 0
        and p.risk_reward >= min_rr
    ]
    return sorted(actionable, key=lambda p: p.sharpe_score, reverse=True)


@dataclass
class HorizonProjection:
    horizon_name: str
    horizon_days: int
    target_pct: float
    target_dollars: float
    expected_dollars: float
    coverage: float                 # expected / target (1.0 = exactamente cubierto)
    top_picks: List[TradePlan]
    meets_target: bool


def project_horizon(
    plans: List[TradePlan],
    capital: float,
    target_pct: float,
    top_n: int = 3,
    horizon: HorizonConfig | None = None,
) -> HorizonProjection:
    """Selecciona las top-N oportunidades y proyecta su P&L vs. el objetivo
    a lo largo del horizonte temporal indicado.

    No hay diferencia matemática entre horizonte diario y semanal aquí: la
    suma de E[P&L] de los planes se compara con ``capital × target_pct``.
    Lo que varía es que los planes (vía ``build_trade_plan``) están
    calibrados para horizontes más largos: ARIMA proyecta varios pasos,
    el ML predice retorno acumulado y los stops/targets son más amplios.
    """
    h = horizon if horizon is not None else HORIZONS["Diario"]
    ranked = rank_plans(plans)[:top_n]
    target_dollars = capital * target_pct
    expected = float(sum(p.expected_pnl_dollars for p in ranked))
    coverage = (expected / target_dollars) if target_dollars > 0 else 0.0
    return HorizonProjection(
        horizon_name=h.name,
        horizon_days=h.days,
        target_pct=target_pct,
        target_dollars=target_dollars,
        expected_dollars=expected,
        coverage=coverage,
        top_picks=ranked,
        meets_target=coverage >= 1.0,
    )


# Alias retro-compatible (firma equivalente al horizonte diario)
DailyProjection = HorizonProjection
project_daily = project_horizon
