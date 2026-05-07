# Motor de Decisiones para Investopedia

Aplicación web interactiva en **Streamlit** que combina econometría
(ARIMA, GARCH), Machine Learning (Gradient Boosting) e indicadores
técnicos para emitir señales **COMPRAR / MANTENER / VENDER** y un
tamaño de posición sobre el capital ficticio del simulador de Investopedia.

## Estructura del proyecto

| Archivo         | Responsabilidad |
|-----------------|-----------------|
| `data.py`       | Descarga (yfinance) e indicadores técnicos (RSI, MACD, Bollinger, SMA) |
| `models.py`     | ARIMA(1,1,1) con IC 95 %, GARCH(1,1) y Gradient Boosting con `TimeSeriesSplit` |
| `portfolio.py`  | VaR paramétrico al 95 % y Criterio de Kelly fraccional |
| `strategies.py` | Hurst, ATR, Sharpe, planes ejecutables y proyección de P&L diario |
| `universe.py`   | Universos de búsqueda (S&P 500, Nasdaq-100, Dow 30, Mag 7, ETFs) |
| `screener.py`   | Screener bursátil de dos etapas (cribado masivo + deep-analysis) |
| `decision.py`   | Pondera los modelos y emite la señal final + tamaño de posición |
| `app.py`        | Dashboard de Streamlit con Plotly |
| `requirements.txt` | Dependencias |

## Cómo se construye la señal

La puntuación final está acotada en `[-1, +1]` y se pondera así:

```
score = 0.40·ML  +  0.25·ARIMA  +  0.35·Técnico
```

- **ML (40 %)**: probabilidad estimada de retorno positivo en el próximo bar,
  centrada en cero. P(subida) > 55 % cuenta como bullish, < 45 % bearish.
- **ARIMA (25 %)**: el % de cambio previsto se satura en ±1 % por bar para
  que un *outlier* no domine la señal.
- **Técnico (35 %)**: votación de RSI (sobreventa/sobrecompra), histograma
  MACD, posición vs Bollinger (%B) y cruce de SMA20/SMA50.

**Umbrales:** `score ≥ +0.30 → BUY`, `score ≤ −0.30 → SELL`, resto `HOLD`.

**GARCH no vota la dirección.** Modula el TAMAÑO de la posición a través
de un *haircut* sobre la fracción de Kelly:

| Volatilidad anualizada | Multiplicador |
|------------------------|---------------|
| < 30 %                 | 1.00          |
| 30 % – 60 %            | 0.80          |
| 60 % – 80 %            | 0.55          |
| > 80 %                 | 0.30          |

Sobre Kelly se aplica además **Half-Kelly** y un **cap del 25 %** del
capital, porque Kelly puro asume que conocemos las probabilidades reales,
algo que en mercados nunca se cumple.

## Cómo ejecutar la app

1. **Clonar el repo y cambiar a la rama:**
   ```bash
   git clone https://github.com/julio-arevalo-hernandez/investopedia.git
   cd investopedia
   git checkout claude/financial-decision-engine-BBBTI
   ```

2. **Crear un entorno virtual e instalar dependencias:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Lanzar Streamlit:**
   ```bash
   streamlit run app.py
   ```
   La app abrirá `http://localhost:8501` en el navegador.

4. **Uso:**
   - Introduce tus tickers separados por coma (`AAPL, TSLA, SPY, ...`).
   - Selecciona intervalo (`1m`, `5m`, `15m`, `30m`, `1h`, `1d`).
   - Indica tu capital ficticio actual de Investopedia.
   - Pulsa **🔄 Refrescar análisis** para invalidar la caché y descargar
     datos recientes (la caché expira sola cada 5 minutos).
   - Revisa la **tabla resumen** (semáforo) y luego entra al **detalle**
     de un ticker para ver razonamiento, métricas econométricas y gráficos.

## Modo "Cazador en Yahoo" (screener de dos etapas)

En el sidebar se elige el modo de operación:

- **📊 Mi watchlist** — analiza únicamente los tickers que escribas.
- **🔭 Cazador en Yahoo** — barre cientos de tickers de varios universos
  (S&P 500 large caps, Nasdaq-100, Dow 30, Mag 7, ETFs sectoriales) y
  devuelve las mejores compras del día.

El Cazador hace dos pasadas para no quemar CPU:

1. **Etapa 1 — cribado rápido (datos diarios, todo el universo).**
   Bulk-download de 3 meses de cierres diarios y scoring compuesto con:

   ```
   composite =  0.40·z(retorno_5d)
              + 0.20·z(retorno_20d)
              + 0.20·trend(SMA20/SMA50)
              + 0.10·rsi_score
              − 0.10·z(|vol_20d − 30%|)
   ```

   Filtros duros: precio ≥ $5 y dollar-volume medio ≥ $5 M (descarta
   ilíquidos donde el slippage destruye el edge). Resultado: ranking de
   los ~110 tickers viables del S&P 500 + ~70 del Nasdaq-100.

2. **Etapa 2 — análisis profundo (intradía, sólo finalistas).**
   Sobre los Top-N (configurable, 12 por defecto) se ejecuta el pipeline
   completo: ARIMA + GARCH + Gradient Boosting + clasificador de régimen
   (Hurst) + plan de trade con stop-loss, take-profit, R:R y E[P&L].

3. **Ranking cross-section por Sharpe esperado** y proyección contra el
   objetivo diario. Si la suma de E[P&L] no cubre el objetivo, el panel
   te lo dice — la decisión correcta es **no operar** ese día, no
   sobre-apalancar para fingir que se cumple.

Caché: la etapa 1 se cachea 1 hora (los datos diarios cambian poco),
la etapa 2 cinco minutos. El botón "🔄 Refrescar análisis" invalida ambas.

## Estrategias concretas de compra

Para maximizar el ratio de ganancias diario, el módulo `strategies.py`
clasifica el régimen de cada ticker con el **exponente de Hurst** y aplica
una de tres tácticas long-only. Para cada una se calcula entrada,
stop-loss, take-profit y número de acciones a comprar:

| Estrategia | Cuándo se activa | Stop | Take-Profit |
|------------|------------------|------|-------------|
| `MOMENTUM_LONG` | Hurst > 0.55 (tendencia persistente) | `entrada − 1.5·ATR` | `entrada + 2·R` |
| `MEAN_REVERSION_LONG` | Hurst < 0.45 y precio en banda extrema (%B<0.2 ó >0.8) | `entrada − 1·ATR` | `min(BB media, 2·R)` |
| `BREAKOUT_LONG` | Régimen mixto, ARIMA proyecta techo | `entrada − 1.5·ATR` | `máx(IC superior ARIMA, 2·R)` |

Cada plan reporta:
- **Acciones a comprar** (capital_kelly / precio).
- **R:R** = recompensa / riesgo.
- **P(win)** del modelo ML y **E[Retorno]** = `p·win% − (1−p)·loss%`.
- **Sharpe esperado** del trade = `E[retorno] / σ_GARCH`.

## Ranking cross-sectional y proyección diaria

Los planes de todos los tickers se rankean por **Sharpe esperado** y se
filtran los que tienen `EV ≤ 0` o `R:R < 1`. Las **Top-N** oportunidades
se muestran como tarjetas con orden ejecutable lista para copiar al
simulador.

La barra lateral incluye un **objetivo de retorno diario** (% sobre
capital). El motor proyecta la suma de E[P&L] de los Top Picks y avisa
si **no cubre el objetivo** — en cuyo caso recomienda esperar mejor
set-up en lugar de sobre-apalancar (filosofía Kelly: el camino más
rápido a la quiebra es ignorar la varianza).

## Econometría adicional disponible

- **Exponente de Hurst** (escalado de varianza) — clasificador de régimen.
- **ATR(14)** de Wilder — volatilidad absoluta para sizing de stops.
- **Sharpe intradía anualizado** — calidad del retorno por unidad de riesgo.
- **Z-score del precio** vs media de 20 — gatillo de mean-reversion.
- **Momentum cross-term** (retorno corto − retorno medio) — confirmación.
- **Intervalo de confianza al 95 % de ARIMA** — usado como techo de
  beneficio en estrategias de breakout.

## Limitaciones conocidas

- yfinance limita el histórico intradía: 7 días para `1m`, 60 días para
  `5m`/`15m`/`30m`/`1h`. La app pide automáticamente el rango máximo
  permitido para cada intervalo.
- El modelo de ML se reentrena en cada análisis. Para tickers con poco
  histórico (< 200 barras) o un único régimen direccional la señal se
  degrada a `HOLD`.
- ARIMA y GARCH pueden no converger en regímenes muy ruidosos; en ese
  caso la subseñal aporta `0` y la decisión se apoya en los componentes
  restantes.
- **No es asesoramiento financiero.** La herramienta es educativa,
  pensada para operar sobre el simulador de Investopedia.
