"""
Streamlit dashboard: Motor de Decisiones para Investopedia.

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data import load_market_data
from decision import WEIGHTS, build_signal
from models import arima_forecast, garch_volatility, ml_direction
from portfolio import estimate_win_loss, kelly_fraction, parametric_var


st.set_page_config(
    page_title="Investopedia Decision Engine",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Caché: el modelo se reentrena para cada (ticker, intervalo) pero sólo cada
# 5 minutos para no saturar yfinance ni gastar CPU innecesariamente.
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def analyze(ticker: str, interval: str, capital: float) -> Dict:
    market = load_market_data(ticker, interval=interval)
    df = market.prices

    if df.empty:
        return {"ticker": ticker, "interval": interval, "error": "Sin datos."}

    arima = arima_forecast(df["Close"])
    garch = garch_volatility(df["LogReturn"])
    ml = ml_direction(df)

    p, avg_w, avg_l = estimate_win_loss(df["LogReturn"])
    kelly = kelly_fraction(p, avg_w, avg_l)
    var = parametric_var(df["LogReturn"], capital=capital, confidence=0.95)

    signal = build_signal(
        df=df,
        arima=arima,
        garch=garch,
        ml=ml,
        capital=capital,
        kelly_fraction_value=kelly.fractional_kelly,
    )

    return {
        "ticker": ticker,
        "interval": interval,
        "market": market,
        "df": df,
        "arima": arima,
        "garch": garch,
        "ml": ml,
        "kelly": kelly,
        "var": var,
        "signal": signal,
        "fetched_at": datetime.utcnow(),
    }


# ---------------------------------------------------------------------------
# Sidebar: inputs del usuario
# ---------------------------------------------------------------------------


st.sidebar.title("⚙️  Configuración")

default_tickers = "AAPL, TSLA, SPY, MSFT, NVDA"
tickers_raw = st.sidebar.text_area(
    "Tickers (separados por coma)", value=default_tickers, height=80
)
interval = st.sidebar.selectbox(
    "Intervalo", options=["1m", "5m", "15m", "30m", "1h", "1d"], index=1
)
capital = st.sidebar.number_input(
    "Capital ficticio ($)", min_value=100.0, value=100_000.0, step=500.0
)

if st.sidebar.button("🔄 Refrescar análisis"):
    analyze.clear()

tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Cabecera y disclaimer
# ---------------------------------------------------------------------------


st.title("📈 Motor de Decisiones — Investopedia")
st.caption(
    "Combina ARIMA + GARCH + Gradient Boosting + indicadores técnicos para "
    "emitir una señal de **COMPRAR / MANTENER / VENDER** y un tamaño de "
    "posición por Kelly fraccional."
)

with st.expander("ℹ️  Cómo se construye la señal"):
    st.markdown(
        f"""
**Ponderación direccional**
- {WEIGHTS['ml']:.0%} → modelo de Machine Learning (Gradient Boosting sobre indicadores)
- {WEIGHTS['arima']:.0%} → pronóstico ARIMA del próximo bar
- {WEIGHTS['technical']:.0%} → consenso técnico (RSI, MACD, %B, cruce de medias)

**Umbrales:** `score ≥ +0.30` → BUY · `score ≤ −0.30` → SELL · resto → HOLD.

**GARCH** no vota la dirección: ajusta el TAMAÑO de la posición. Si la volatilidad
anualizada prevista es alta, recortamos la fracción de Kelly aplicando un *haircut*
(hasta multiplicar por 0.30 en regímenes muy volátiles). Sobre Kelly aplicamos
además *Half-Kelly* + cap del 25 % para evitar sobre-apalancarnos.
        """
    )


if not tickers:
    st.warning("Introduce al menos un ticker en la barra lateral.")
    st.stop()


# ---------------------------------------------------------------------------
# Tabla resumen (semáforo)
# ---------------------------------------------------------------------------


COLOR = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}

summary_rows = []
results: Dict[str, Dict] = {}

with st.spinner("Descargando datos y entrenando modelos..."):
    for tk in tickers:
        try:
            results[tk] = analyze(tk, interval, capital)
        except Exception as exc:  # noqa: BLE001
            results[tk] = {"ticker": tk, "error": str(exc)}

for tk, r in results.items():
    if "error" in r:
        summary_rows.append(
            {
                "Ticker": tk, "Precio": np.nan, "Señal": "—",
                "Score": np.nan, "Sugerido $": 0.0, "VaR 95% $": 0.0,
                "Estado": r["error"],
            }
        )
        continue
    sig = r["signal"]
    summary_rows.append(
        {
            "Ticker": tk,
            "Precio": r["market"].last_price,
            "Señal": f"{COLOR[sig.action]} {sig.action}",
            "Score": sig.score,
            "Sugerido $": sig.suggested_dollars,
            "VaR 95% $": r["var"].var_money,
            "Estado": "OK",
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
st.subheader("Panel de decisión")
st.dataframe(
    summary_df.style.format(
        {
            "Precio": "{:,.2f}",
            "Score": "{:+.2f}",
            "Sugerido $": "${:,.0f}",
            "VaR 95% $": "${:,.0f}",
        }
    ),
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Detalle por ticker
# ---------------------------------------------------------------------------


st.subheader("Análisis detallado")
selected = st.selectbox("Selecciona un ticker para ver su desglose", tickers)
r = results.get(selected, {})

if "error" in r:
    st.error(f"No se pudo analizar {selected}: {r['error']}")
    st.stop()

df = r["df"]
sig = r["signal"]
arima = r["arima"]
garch = r["garch"]
ml = r["ml"]
kelly = r["kelly"]
var = r["var"]


col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio actual", f"${r['market'].last_price:,.2f}")
col2.metric(
    "Señal",
    f"{COLOR[sig.action]} {sig.action}",
    delta=f"score {sig.score:+.2f}",
)
col3.metric(
    "Tamaño sugerido",
    f"${sig.suggested_dollars:,.0f}",
    delta=f"{sig.suggested_fraction:.1%} del capital",
)
col4.metric(
    "VaR 95% (1 bar)",
    f"-${var.var_money:,.0f}",
    delta=f"{var.var_pct:.2%}",
)


# Razonamiento
with st.expander("🧠  Razonamiento de la señal", expanded=True):
    st.write("**Aporte de cada componente al score:**")
    comp_df = pd.DataFrame(
        {
            "Componente": ["ML", "ARIMA", "Técnico"],
            "Peso": [WEIGHTS["ml"], WEIGHTS["arima"], WEIGHTS["technical"]],
            "Valor": [
                sig.components.get("ml", 0.0),
                sig.components.get("arima", 0.0),
                sig.components.get("technical", 0.0),
            ],
        }
    )
    comp_df["Aporte"] = comp_df["Peso"] * comp_df["Valor"]
    st.dataframe(
        comp_df.style.format(
            {"Peso": "{:.0%}", "Valor": "{:+.2f}", "Aporte": "{:+.2f}"}
        ),
        use_container_width=True,
    )
    for note in sig.rationale:
        st.write(f"- {note}")


# Métricas econométricas
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**ARIMA(1,1,1)**")
    if arima.success:
        st.write(f"Precio previsto: `${arima.forecast:,.2f}`")
        st.write(f"Cambio esperado: `{arima.pct_change*100:+.3f}%`")
    else:
        st.write("No converge con los datos disponibles.")
with m2:
    st.markdown("**GARCH(1,1)**")
    if garch.success:
        st.write(f"σ próx. bar: `{garch.sigma*100:.3f}%`")
        st.write(f"Vol. anualizada: `{garch.annualized*100:.1f}%`")
    else:
        st.write("No se pudo ajustar.")
with m3:
    st.markdown("**Gradient Boosting**")
    if ml.success:
        st.write(f"P(subida): `{ml.prob_up:.2%}`")
        st.write(f"Accuracy CV: `{ml.cv_accuracy:.2%}`")
    else:
        st.write("Datos insuficientes para entrenar.")


# Kelly
st.markdown("### Dimensionado de la posición (Kelly)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("P(ganar) hist.", f"{kelly.win_prob:.1%}")
k2.metric("Win/Loss ratio", f"{kelly.win_loss_ratio:.2f}")
k3.metric("Kelly puro", f"{kelly.full_kelly:.1%}")
k4.metric("Half-Kelly aplicado", f"{kelly.fractional_kelly:.1%}")


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------


st.markdown("### Gráficos interactivos")

tail = df.tail(300)
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.2, 0.25],
    vertical_spacing=0.04,
    subplot_titles=("Precio + Bandas de Bollinger + SMA", "RSI", "MACD"),
)

# Precio + Bollinger + SMAs
fig.add_trace(
    go.Candlestick(
        x=tail.index, open=tail["Open"], high=tail["High"],
        low=tail["Low"], close=tail["Close"], name="OHLC",
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=tail.index, y=tail["BB_Upper"], name="BB Upper",
               line=dict(color="lightgray", width=1)),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=tail.index, y=tail["BB_Lower"], name="BB Lower",
               line=dict(color="lightgray", width=1), fill="tonexty",
               fillcolor="rgba(200,200,200,0.15)"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=tail.index, y=tail["SMA_20"], name="SMA 20",
               line=dict(color="#1f77b4", width=1.2)),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=tail.index, y=tail["SMA_50"], name="SMA 50",
               line=dict(color="#ff7f0e", width=1.2)),
    row=1, col=1,
)

# Marcador para la previsión ARIMA
if arima.success:
    delta = tail.index[-1] - tail.index[-2]
    next_ts = tail.index[-1] + delta
    fig.add_trace(
        go.Scatter(
            x=[tail.index[-1], next_ts],
            y=[arima.last_price, arima.forecast],
            name="ARIMA forecast",
            mode="lines+markers",
            line=dict(color="magenta", width=2, dash="dash"),
        ),
        row=1, col=1,
    )

# RSI
fig.add_trace(
    go.Scatter(x=tail.index, y=tail["RSI"], name="RSI",
               line=dict(color="#9467bd")),
    row=2, col=1,
)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

# MACD
fig.add_trace(
    go.Bar(x=tail.index, y=tail["MACD_Hist"], name="MACD hist",
           marker_color="rgba(99,110,250,0.5)"),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=tail.index, y=tail["MACD"], name="MACD",
               line=dict(color="#2ca02c")),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=tail.index, y=tail["MACD_Signal"], name="Signal",
               line=dict(color="#d62728")),
    row=3, col=1,
)

fig.update_layout(
    height=750,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Datos refrescados: {r['fetched_at']:%Y-%m-%d %H:%M:%S} UTC · "
    "fuente: Yahoo Finance vía yfinance. Esta herramienta es educativa: "
    "no constituye asesoramiento financiero."
)
