"""
Módulo de Análisis Econométrico y Predictivo.

Tres modelos independientes que producen señales que luego combina
``decision.py``:

- ``arima_forecast``  → tendencia esperada del precio en el próximo periodo.
- ``garch_volatility`` → desviación típica condicional (riesgo inminente).
- ``ml_direction``    → probabilidad de que el próximo retorno sea positivo,
  usando los indicadores técnicos como features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA

# Las series intradía generan muchos avisos de convergencia que ensucian la UI;
# los silenciamos porque ya gestionamos los fallos con try/except.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


@dataclass
class ArimaResult:
    forecast: float          # precio esperado en el siguiente bar
    forecast_lower: float    # límite inferior del IC al 95 %
    forecast_upper: float    # límite superior del IC al 95 %
    last_price: float
    pct_change: float        # cambio relativo previsto
    direction: int           # +1 alcista, -1 bajista, 0 neutro
    success: bool


def arima_forecast(
    close: pd.Series,
    order: tuple = (1, 1, 1),
    lookback: int = 400,
) -> ArimaResult:
    """Ajusta un ARIMA simple y devuelve la previsión del siguiente bar.

    Trabajamos con los últimos ``lookback`` puntos para que el ajuste sea
    rápido y refleje la dinámica reciente (lo relevante para day-trading).
    Además del punto central exponemos el intervalo de confianza al 95 %,
    indispensable para fijar stops y targets con base estadística.
    """
    last_price = float(close.iloc[-1])
    fail = ArimaResult(
        last_price, last_price, last_price, last_price, 0.0, 0, success=False
    )

    if len(close) < 50:
        return fail

    series = close.tail(lookback).astype(float)

    try:
        model = ARIMA(series, order=order)
        fit = model.fit()
        fc = fit.get_forecast(steps=1)
        forecast = float(fc.predicted_mean.iloc[0])
        ci = fc.conf_int(alpha=0.05)
        lower = float(ci.iloc[0, 0])
        upper = float(ci.iloc[0, 1])
    except Exception:
        return fail

    pct_change = (forecast - last_price) / last_price
    # Banda muerta del 0,05 % para no operar por ruido numérico
    if pct_change > 0.0005:
        direction = 1
    elif pct_change < -0.0005:
        direction = -1
    else:
        direction = 0

    return ArimaResult(
        forecast=forecast,
        forecast_lower=lower,
        forecast_upper=upper,
        last_price=last_price,
        pct_change=pct_change,
        direction=direction,
        success=True,
    )


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------


@dataclass
class GarchResult:
    sigma: float          # volatilidad condicional prevista (en unidades de retorno)
    annualized: float     # volatilidad anualizada aproximada
    success: bool


def garch_volatility(
    log_returns: pd.Series,
    lookback: int = 500,
    bars_per_year: int = 252 * 78,   # ~78 barras de 5m por sesión bursátil
) -> GarchResult:
    """Ajusta un GARCH(1,1) y devuelve la volatilidad prevista."""
    series = log_returns.dropna().tail(lookback)
    if len(series) < 50:
        return GarchResult(sigma=float("nan"), annualized=float("nan"), success=False)

    # ``arch`` espera retornos en %, no en unidades; reescalar mejora el
    # condicionamiento numérico del optimizador.
    try:
        scaled = series * 100
        am = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=1, reindex=False)
        sigma_pct = float(np.sqrt(forecast.variance.values[-1, 0]))
        sigma = sigma_pct / 100.0
    except Exception:
        return GarchResult(sigma=float("nan"), annualized=float("nan"), success=False)

    annualized = sigma * np.sqrt(bars_per_year)
    return GarchResult(sigma=sigma, annualized=annualized, success=True)


# ---------------------------------------------------------------------------
# Machine Learning (clasificación de la dirección)
# ---------------------------------------------------------------------------


# Features que el modelo usa para predecir la dirección del próximo retorno.
ML_FEATURES: List[str] = [
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_PctB",
    "SMA_20",
    "SMA_50",
    "MA_Cross",
    "LogReturn",
]


@dataclass
class MlResult:
    prob_up: float        # probabilidad estimada de retorno positivo
    direction: int        # +1 si prob_up > 0.55, -1 si < 0.45, 0 en zona gris
    cv_accuracy: float    # accuracy media en validación temporal
    success: bool


def _build_supervised(df: pd.DataFrame):
    """Convierte el DataFrame enriquecido en (X, y) listos para entrenar."""
    work = df.copy()
    # y_t = signo del retorno del siguiente bar
    work["Target"] = (work["LogReturn"].shift(-1) > 0).astype(int)
    work = work.dropna(subset=ML_FEATURES + ["Target"])
    X = work[ML_FEATURES]
    y = work["Target"]
    return X, y


def ml_direction(df: pd.DataFrame) -> MlResult:
    """Entrena un Gradient Boosting y predice la dirección del próximo bar.

    Se usa ``TimeSeriesSplit`` para no contaminar la validación con
    información del futuro.
    """
    if df.empty or len(df) < 200:
        return MlResult(0.5, 0, 0.0, success=False)

    X, y = _build_supervised(df)
    if len(X) < 150 or y.nunique() < 2:
        return MlResult(0.5, 0, 0.0, success=False)

    try:
        model = GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42
        )

        tscv = TimeSeriesSplit(n_splits=4)
        accs = []
        for train_idx, test_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            accs.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))
        cv_acc = float(np.mean(accs))

        # Reentrenamos con todo el histórico antes de predecir el siguiente bar
        model.fit(X, y)
        last_features = df[ML_FEATURES].dropna().iloc[[-1]]
        prob_up = float(model.predict_proba(last_features)[0, 1])
    except Exception:
        return MlResult(0.5, 0, 0.0, success=False)

    if prob_up > 0.55:
        direction = 1
    elif prob_up < 0.45:
        direction = -1
    else:
        direction = 0

    return MlResult(
        prob_up=prob_up, direction=direction, cv_accuracy=cv_acc, success=True
    )
