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
from decision import SENSITIVITY_PRESETS, WEIGHTS, build_signal
from models import arima_forecast, garch_volatility, ml_direction
from portfolio import estimate_win_loss, kelly_fraction, parametric_var
from screener import select_finalists, stage1_screen
from strategies import (
    HORIZONS,
    bars_per_year,
    build_trade_plan,
    intraday_sharpe,
    momentum_score,
    price_zscore,
    project_horizon,
)
from universe import combine_universes, list_universes


st.set_page_config(
    page_title="Investopedia Decision Engine",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Caché: el modelo se reentrena para cada (ticker, intervalo) pero sólo cada
# 5 minutos para no saturar yfinance ni gastar CPU innecesariamente.
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stage1(tickers_key: str) -> pd.DataFrame:
    """Cachea la etapa 1 del screener una hora — cambia poco entre sesiones."""
    tickers = tickers_key.split(",") if tickers_key else []
    return stage1_screen(tickers)


@st.cache_data(ttl=300, show_spinner=False)
def analyze(
    ticker: str, interval: str, capital: float, horizon_name: str,
    buy_threshold: float, risk_per_trade_pct: float,
) -> Dict:
    horizon = HORIZONS[horizon_name]
    market = load_market_data(ticker, interval=interval)
    df = market.prices

    if df.empty:
        return {"ticker": ticker, "interval": interval, "error": "Sin datos."}

    arima = arima_forecast(df["Close"], steps=horizon.arima_steps)
    bpy = bars_per_year(interval)
    garch = garch_volatility(df["LogReturn"], bars_per_year=bpy)
    ml = ml_direction(df, steps=horizon.arima_steps)

    p, avg_w, avg_l = estimate_win_loss(df["LogReturn"])
    kelly = kelly_fraction(p, avg_w, avg_l)
    var = parametric_var(
        df["LogReturn"], capital=capital, confidence=0.95,
        horizon_bars=horizon.arima_steps,
    )

    signal = build_signal(
        df=df,
        arima=arima,
        garch=garch,
        ml=ml,
        capital=capital,
        kelly_fraction_value=kelly.fractional_kelly,
        horizon_days=horizon.days,
        buy_threshold=buy_threshold,
        risk_per_trade_pct=risk_per_trade_pct,
    )

    plan = build_trade_plan(
        ticker=ticker,
        df=df,
        arima=arima,
        garch=garch,
        ml=ml,
        signal_action=signal.action,
        position_dollars=signal.suggested_dollars,
        horizon=horizon,
        capital=capital,
        risk_per_trade_pct=risk_per_trade_pct,
    )

    sharpe = intraday_sharpe(df["LogReturn"], bars_per_year=bpy)
    z = price_zscore(df["Close"])
    mom = momentum_score(df["Close"])

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
        "plan": plan,
        "sharpe": sharpe,
        "zscore": z,
        "momentum": mom,
        "fetched_at": datetime.utcnow(),
    }


# ---------------------------------------------------------------------------
# Helper: render del bloque "Top Picks"
# ---------------------------------------------------------------------------


def render_top_picks(
    plans, capital: float, target_pct: float, top_n: int,
    horizon=None, title_prefix: str = "🚀 Top Picks",
):
    h = horizon if horizon is not None else HORIZONS["Diario"]
    projection = project_horizon(
        plans, capital, target_pct, int(top_n), horizon=h,
    )
    st.subheader(
        f"{title_prefix} — objetivo {target_pct*100:.1f}% "
        f"{h.name.lower()} ({h.days}d hábiles)"
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(f"Objetivo {h.name}", f"${projection.target_dollars:,.0f}",
              delta=f"{target_pct*100:.1f}% s/ capital")
    p2.metric("E[P&L] Top Picks", f"${projection.expected_dollars:,.0f}",
              delta=f"{projection.coverage*100:.0f}% del objetivo")
    p3.metric("¿Cubre objetivo?",
              "✅ SÍ" if projection.meets_target else "⚠️  NO",
              delta=f"{len(projection.top_picks)} oportunidades")
    p4.metric("Capital comprometido",
              f"${sum(p.position_dollars for p in projection.top_picks):,.0f}")

    if not projection.top_picks:
        st.warning(
            "⚠️  Ningún ticker cumple los filtros (BUY + EV positivo + R:R ≥ 1). "
            "Lo más rentable estadísticamente en este horizonte es **mantener efectivo** y "
            "esperar mejor set-up. Forzar entradas destruye el ratio de ganancias."
        )
        return projection

    if not projection.meets_target:
        st.warning(
            f"E[P&L] de los Top Picks = ${projection.expected_dollars:,.0f} no "
            f"alcanza el objetivo (${projection.target_dollars:,.0f}). "
            "Amplía la lista de tickers, baja el objetivo o acepta un día más "
            "conservador (sobre-apalancar destruye varianza-ajustada)."
        )

    picks_df = pd.DataFrame(
        [
            {
                "#": i + 1, "Ticker": p.ticker, "Estrategia": p.strategy,
                "Régimen (Hurst)": f"{p.regime} ({p.hurst:.2f})",
                "Acciones": p.shares, "Entrada": p.entry,
                "Stop": p.stop_loss, "Target": p.take_profit,
                "R:R": p.risk_reward, "P(win)": p.prob_up,
                "E[Ret %]": p.expected_return_pct,
                "E[P&L $]": p.expected_pnl_dollars,
                "Sharpe": p.sharpe_score,
            }
            for i, p in enumerate(projection.top_picks)
        ]
    ).set_index("#")

    st.dataframe(
        picks_df.style.format(
            {
                "Entrada": "${:,.2f}", "Stop": "${:,.2f}", "Target": "${:,.2f}",
                "R:R": "{:.2f}", "P(win)": "{:.1%}", "E[Ret %]": "{:+.2%}",
                "E[P&L $]": "${:+,.2f}", "Sharpe": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    for i, p in enumerate(projection.top_picks, 1):
        emoji = {"MOMENTUM_LONG": "📈", "MEAN_REVERSION_LONG": "↩️",
                 "BREAKOUT_LONG": "💥"}.get(p.strategy, "🎯")
        with st.expander(
            f"{emoji}  #{i}  {p.ticker} — {p.strategy}  "
            f"·  COMPRAR {p.shares} acc. @ ${p.entry:,.2f}  "
            f"·  E[P&L] ${p.expected_pnl_dollars:+,.2f}",
            expanded=(i == 1),
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"""
**Orden de COMPRA**
- Ticker: **{p.ticker}**
- Acciones: **{p.shares}**
- Entrada (market): **${p.entry:,.2f}**
- Capital comprometido: **${p.position_dollars:,.2f}**
"""
            )
            c2.markdown(
                f"""
**Gestión de riesgo**
- 🛑 Stop-loss: **${p.stop_loss:,.2f}**  ({(p.stop_loss/p.entry-1)*100:+.2f}%)
- 🎯 Take-profit: **${p.take_profit:,.2f}**  ({(p.take_profit/p.entry-1)*100:+.2f}%)
- R:R = **{p.risk_reward:.2f}** · Riesgo $ = **${p.risk_dollars:,.2f}**
"""
            )
            c3.markdown(
                f"""
**Edge estadístico**
- P(win) ML: **{p.prob_up:.1%}**
- E[retorno]: **{p.expected_return_pct*100:+.2f}%**
- E[P&L]: **${p.expected_pnl_dollars:+,.2f}**
- Sharpe esperado: **{p.sharpe_score:.2f}**
"""
            )
            st.markdown("**Razonamiento econométrico:**")
            for note in p.rationale:
                st.write(f"- {note}")

    return projection


# ---------------------------------------------------------------------------
# Sidebar: inputs del usuario
# ---------------------------------------------------------------------------


st.sidebar.title("⚙️  Configuración")

mode = st.sidebar.radio(
    "Modo",
    ["📊 Mi watchlist", "🔭 Cazador en Yahoo"],
    help=(
        "Watchlist: analiza sólo los tickers que tú especifiques.\n"
        "Cazador: barre cientos de acciones del S&P 500 / Nasdaq-100 / "
        "Dow 30 con un screener de dos etapas y entrega los mejores "
        "buys del horizonte."
    ),
)

horizon_name = st.sidebar.selectbox(
    "⏱️  Horizonte de inversión",
    options=list(HORIZONS.keys()),
    index=0,
    help=(
        "Diario:    1 día  · target moderado, ARIMA 1-step, ATR×1.5 / R:R 2.0\n"
        "Semanal:   5 días · target medio,    ARIMA 5-step, ATR×2.0 / R:R 3.0\n"
        "Quincenal: 10 días · ARIMA 10-step, ATR×2.5 / R:R 3.0\n"
        "Mensual:   21 días · ARIMA 21-step, ATR×3.0 / R:R 4.0"
    ),
)
horizon = HORIZONS[horizon_name]

default_tickers = "AAPL, TSLA, SPY, MSFT, NVDA"
tickers_raw = st.sidebar.text_area(
    "Tickers (separados por coma)", value=default_tickers, height=80
)

intervals = ["1m", "5m", "15m", "30m", "1h", "1d"]
default_interval_idx = (
    intervals.index(horizon.interval) if horizon.interval in intervals else 1
)
interval = st.sidebar.selectbox(
    "Intervalo de datos",
    options=intervals,
    index=default_interval_idx,
    help=f"Sugerido para {horizon.name}: {horizon.interval}",
    key=f"interval_{horizon.name}",
)
capital = st.sidebar.number_input(
    "Capital ficticio ($)", min_value=100.0, value=100_000.0, step=500.0
)

sensitivity = st.sidebar.selectbox(
    "🎚️  Sensibilidad de la señal",
    options=list(SENSITIVITY_PRESETS.keys()),
    index=1,  # Equilibrado por defecto
    help=(
        "Conservador (umbral 0.30): sólo BUY con consenso muy fuerte.\n"
        "Equilibrado (0.15): recomendaciones moderadas (DEFAULT).\n"
        "Agresivo (0.05): cualquier sesgo positivo cuenta — más operaciones, "
        "menos convicción por operación."
    ),
)
buy_threshold = SENSITIVITY_PRESETS[sensitivity]

risk_per_trade_pct = st.sidebar.slider(
    "⚠️  Riesgo por operación (% del capital)",
    min_value=0.5, max_value=5.0, value=2.0, step=0.25,
    help=(
        "Cuánto capital pones en juego por trade (la pérdida si toca stop). "
        "1-2 % es lo industria-standard; superior eleva el P&L pero también "
        "el drawdown."
    ),
) / 100.0

# El objetivo crece sub-linealmente con el horizonte (raíz cuadrada del tiempo,
# clásico en finanzas). Esto evita pedir 21 % al mensual (lo que forzaría
# estrategias muy agresivas).
default_target = float(np.sqrt(horizon.days) * 1.0)  # 1 % diario, ~2.2 % semanal
max_target = float(max(5.0, horizon.days * 2.0))
target_pct = st.sidebar.slider(
    f"🎯 Objetivo de retorno {horizon.name.lower()} (%)",
    min_value=0.1, max_value=max_target,
    value=min(default_target, max_target), step=0.1,
    key=f"target_{horizon.name}",
) / 100.0
daily_target_pct = target_pct  # alias mantenido por código heredado

top_n = st.sidebar.number_input(
    "Nº de Top Picks", min_value=1, max_value=10, value=3
)

if st.sidebar.button("🔄 Refrescar análisis"):
    analyze.clear()
    cached_stage1.clear()

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


COLOR = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}


# ---------------------------------------------------------------------------
# Modo "Cazador en Yahoo": screener de dos etapas
# ---------------------------------------------------------------------------


if mode == "🔭 Cazador en Yahoo":
    st.subheader("🔭 Cazador de oportunidades — todo Yahoo Finance")
    st.caption(
        "Etapa 1: criba diaria de cientos de tickers por momentum, tendencia, "
        "RSI y liquidez. Etapa 2: ARIMA + GARCH + ML + plan de trade sólo "
        "sobre los finalistas. Salida: las mejores compras del horizonte rankeadas "
        "por Sharpe esperado."
    )

    col_l, col_r = st.columns([2, 1])
    with col_l:
        chosen_universes = st.multiselect(
            "Universos a barrer",
            list_universes(),
            default=["S&P 500 large caps", "Nasdaq-100"],
        )
    with col_r:
        n_finalists = st.number_input(
            "Nº de finalistas (deep-analysis)",
            min_value=5, max_value=30, value=12,
            help="Más finalistas = más cobertura, pero cada uno requiere "
                 "ajustar ARIMA/GARCH/ML (~5-15 s por ticker).",
        )

    extra_raw = st.text_input(
        "Tickers extra (opcional, separados por coma)", value="",
    )
    extra_tickers = [t.strip().upper() for t in extra_raw.split(",") if t.strip()]

    universe_tickers = combine_universes(chosen_universes, extra=extra_tickers)
    st.caption(f"Universo total: **{len(universe_tickers)} tickers**")

    if st.button("🚀 Lanzar búsqueda", type="primary"):
        if not universe_tickers:
            st.error("Selecciona al menos un universo o aporta tickers.")
            st.stop()

        with st.spinner(
            f"Etapa 1 — cribando {len(universe_tickers)} tickers con datos diarios..."
        ):
            scored = cached_stage1(",".join(universe_tickers))

        if scored.empty:
            st.error(
                "El screener no devolvió candidatos. Comprueba conexión a "
                "Yahoo Finance o filtros de liquidez."
            )
            st.stop()

        st.success(
            f"Etapa 1 completada: {len(scored)} tickers viables "
            f"(filtrado por liquidez y precio). Top 30 por score:"
        )
        st.dataframe(
            scored.head(30).style.format(
                {
                    "price": "${:,.2f}",
                    "ret_1d": "{:+.2%}", "ret_5d": "{:+.2%}", "ret_20d": "{:+.2%}",
                    "vol_20d_ann": "{:.1%}",
                    "avg_dollar_vol": "${:,.0f}",
                    "rsi_14": "{:.1f}",
                    "composite": "{:+.2f}",
                }
            ),
            use_container_width=True,
        )

        finalists = select_finalists(scored, int(n_finalists))
        st.markdown(
            f"### Etapa 2 — análisis profundo de {len(finalists)} finalistas"
        )
        st.write(", ".join(finalists))

        progress = st.progress(0.0)
        screener_results: Dict[str, Dict] = {}
        for i, tk in enumerate(finalists, 1):
            try:
                screener_results[tk] = analyze(
                    tk, interval, capital, horizon_name,
                    buy_threshold, risk_per_trade_pct,
                )
            except Exception as exc:  # noqa: BLE001
                screener_results[tk] = {"ticker": tk, "error": str(exc)}
            progress.progress(i / len(finalists))
        progress.empty()

        # Tabla de finalistas con sus señales
        sum_rows = []
        for tk, r in screener_results.items():
            if "error" in r:
                continue
            sig = r["signal"]
            plan = r["plan"]
            sum_rows.append(
                {
                    "Ticker": tk,
                    "Precio": r["market"].last_price,
                    "Señal": f"{COLOR[sig.action]} {sig.action}",
                    "Score": sig.score,
                    "Estrategia": plan.strategy,
                    "E[P&L $]": plan.expected_pnl_dollars,
                    "Sharpe": plan.sharpe_score,
                }
            )
        if sum_rows:
            st.dataframe(
                pd.DataFrame(sum_rows).set_index("Ticker").style.format(
                    {
                        "Precio": "${:,.2f}", "Score": "{:+.2f}",
                        "E[P&L $]": "${:+,.2f}", "Sharpe": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

        # Top picks finales rankeados cross-section
        plans = [r["plan"] for r in screener_results.values() if "plan" in r]
        render_top_picks(
            plans, capital, target_pct, top_n, horizon=horizon,
            title_prefix=f"🏆 Mejores compras — horizonte {horizon.name.lower()}",
        )

        st.caption(
            "Recuerda: el screener filtra por momentum reciente y liquidez. "
            "Las señales BUY siguen exigiendo EV positivo y R:R ≥ 1; si "
            "ningún finalista pasa esos filtros, la recomendación correcta "
            "es **no operar en este horizonte** (preservar capital es óptimo "
            "cuando no hay edge)."
        )

    st.stop()


# ---------------------------------------------------------------------------
# Modo "Watchlist": flujo clásico
# ---------------------------------------------------------------------------


if not tickers:
    st.warning("Introduce al menos un ticker en la barra lateral.")
    st.stop()

summary_rows = []
results: Dict[str, Dict] = {}

with st.spinner("Descargando datos y entrenando modelos..."):
    for tk in tickers:
        try:
            results[tk] = analyze(
                tk, interval, capital, horizon_name,
                buy_threshold, risk_per_trade_pct,
            )
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
# Panel de diagnóstico: por qué cada ticker no entra a Top Picks
# ---------------------------------------------------------------------------


with st.expander(
    f"🔬  Diagnóstico — umbral activo {buy_threshold:+.2f} ({sensitivity})",
    expanded=False,
):
    diag_rows = []
    for tk, r in results.items():
        if "error" in r:
            continue
        sig = r["signal"]
        plan = r.get("plan")
        diag_rows.append(
            {
                "Ticker": tk,
                "Señal": f"{COLOR[sig.action]} {sig.action}",
                "Score": sig.score,
                "ML": sig.components.get("ml", 0.0),
                "ARIMA": sig.components.get("arima", 0.0),
                "Técnico": sig.components.get("technical", 0.0),
                "Shares": plan.shares if plan else 0,
                "E[P&L $]": plan.expected_pnl_dollars if plan else 0.0,
                "R:R": plan.risk_reward if plan else 0.0,
                "¿Por qué no BUY?": sig.block_reason or "—",
            }
        )
    if diag_rows:
        st.dataframe(
            pd.DataFrame(diag_rows).set_index("Ticker").style.format(
                {
                    "Score": "{:+.2f}", "ML": "{:+.2f}", "ARIMA": "{:+.2f}",
                    "Técnico": "{:+.2f}", "E[P&L $]": "${:+,.2f}", "R:R": "{:.2f}",
                }
            ),
            use_container_width=True,
        )
        st.caption(
            "**ML / ARIMA / Técnico** son los aportes ya ponderados (peso × valor). "
            "Suman al **Score**. Si el score no llega al umbral, la fila explica "
            "qué componente está más débil. Baja la sensibilidad a *Agresivo* "
            "para operar con scores menores, o cambia el horizonte (semanal/mensual "
            "tienden a generar scores más altos al integrar más estructura)."
        )


all_plans = [r["plan"] for r in results.values() if "plan" in r]
render_top_picks(all_plans, capital, target_pct, top_n, horizon=horizon)


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
plan = r["plan"]


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


# ----- Plan de trade concreto -----
if plan.strategy != "AVOID":
    st.markdown(
        f"### 📝 Plan de trade — **{plan.strategy}** "
        f"(régimen {plan.regime}, Hurst {plan.hurst:.2f})"
    )
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Comprar", f"{plan.shares} acc.",
               delta=f"@ ${plan.entry:,.2f}")
    pc2.metric("Stop-loss", f"${plan.stop_loss:,.2f}",
               delta=f"{(plan.stop_loss/plan.entry-1)*100:+.2f}%")
    pc3.metric("Take-profit", f"${plan.take_profit:,.2f}",
               delta=f"{(plan.take_profit/plan.entry-1)*100:+.2f}%")
    pc4.metric("R:R", f"{plan.risk_reward:.2f}",
               delta=f"E[P&L] ${plan.expected_pnl_dollars:+,.0f}")
    with st.expander("Detalle del plan"):
        for n in plan.rationale:
            st.write(f"- {n}")
else:
    st.info(
        f"📝 Plan: **AVOID**. Señal actual = {plan.action}. "
        "No se abre largo en este ticker."
    )


# ----- Econometría avanzada -----
ec1, ec2, ec3, ec4 = st.columns(4)
ec1.metric("Hurst", f"{plan.hurst:.2f}",
           delta="trending" if plan.hurst > 0.55 else
                 ("mean-rev" if plan.hurst < 0.45 else "neutro"))
ec2.metric("Sharpe intradía (anual.)", f"{r['sharpe']:.2f}")
ec3.metric("Z-score precio (20)", f"{r['zscore']:+.2f}",
           delta="extremo" if abs(r['zscore']) > 2 else "normal")
ec4.metric("Momentum (corto-medio)", f"{r['momentum']*100:+.2f}%")


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
        st.write(
            f"IC 95 %: `[${arima.forecast_lower:,.2f}, ${arima.forecast_upper:,.2f}]`"
        )
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

# Marcador para la previsión ARIMA + IC al 95 %
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
    fig.add_trace(
        go.Scatter(
            x=[next_ts, next_ts],
            y=[arima.forecast_lower, arima.forecast_upper],
            name="ARIMA IC 95%",
            mode="lines",
            line=dict(color="magenta", width=8),
            opacity=0.25,
        ),
        row=1, col=1,
    )

# Líneas de stop-loss y take-profit del plan vigente
if plan.strategy != "AVOID":
    fig.add_hline(
        y=plan.entry, line_dash="dot", line_color="white",
        annotation_text=f"Entrada {plan.entry:.2f}", row=1, col=1,
    )
    fig.add_hline(
        y=plan.stop_loss, line_dash="dash", line_color="red",
        annotation_text=f"Stop {plan.stop_loss:.2f}", row=1, col=1,
    )
    fig.add_hline(
        y=plan.take_profit, line_dash="dash", line_color="lime",
        annotation_text=f"Target {plan.take_profit:.2f}", row=1, col=1,
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
