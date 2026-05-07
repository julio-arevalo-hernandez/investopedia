# Motor de Decisiones para Investopedia

Aplicación web interactiva en **Streamlit** que combina econometría
(ARIMA, GARCH), Machine Learning (Gradient Boosting) e indicadores
técnicos para emitir señales **COMPRAR / MANTENER / VENDER** y un
tamaño de posición sobre el capital ficticio del simulador de Investopedia.

## Estructura del proyecto

| Archivo         | Responsabilidad |
|-----------------|-----------------|
| `data.py`       | Descarga (yfinance) e indicadores técnicos (RSI, MACD, Bollinger, SMA) |
| `models.py`     | ARIMA(1,1,1), GARCH(1,1) y Gradient Boosting con `TimeSeriesSplit` |
| `portfolio.py`  | VaR paramétrico al 95 % y Criterio de Kelly fraccional |
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
