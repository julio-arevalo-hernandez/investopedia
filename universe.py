"""
Universos de búsqueda (tickers candidatos a operar).

Listas curadas de acciones US líquidas — buenas para day-trading en
Investopedia porque ofrecen spread bajo y datos intradía fiables en
yfinance. La idea es que el screener barra estos universos en lugar
de limitarse a la watchlist manual del usuario.

Las listas están hard-codeadas (no requieren red) para garantizar que
el screener funcione siempre. Aun así, el usuario puede combinar
universos o añadir tickers extra desde la UI.
"""

from __future__ import annotations

from typing import Dict, List


# S&P 500 — top ~110 nombres por capitalización y liquidez. Cubrir los 500
# completos no aporta mucho a un trader intradía (los nombres pequeños son
# ilíquidos y meten ruido al ranking).
SP500_LARGECAP: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "JPM", "V", "WMT", "MA", "JNJ", "XOM", "PG", "ORCL", "HD", "COST", "ABBV",
    "BAC", "KO", "MRK", "CVX", "PEP", "ADBE", "CRM", "AMD", "NFLX", "MCD",
    "TMO", "CSCO", "LIN", "ACN", "ABT", "DIS", "WFC", "INTC", "VZ", "QCOM",
    "DHR", "TXN", "INTU", "PFE", "AMGN", "PM", "T", "UNP", "COP", "BMY",
    "LOW", "RTX", "HON", "GS", "UPS", "BLK", "IBM", "CAT", "AMAT", "SCHW",
    "DE", "AXP", "BKNG", "SPGI", "PLD", "ELV", "GILD", "ADI", "C", "MDLZ",
    "VRTX", "TJX", "SYK", "MS", "ISRG", "MMC", "REGN", "LMT", "ETN", "ADP",
    "BX", "PANW", "BA", "PGR", "BSX", "MU", "CB", "FI", "LRCX", "ZTS",
    "MO", "EQIX", "NOW", "TGT", "SO", "DUK", "CI", "SLB", "SHW", "PYPL",
    "USB", "ICE", "ITW", "FCX", "HCA", "CL", "EMR", "BDX", "CMG", "MAR",
    "F", "GM", "GE", "UBER", "LYFT", "SNAP", "PINS", "RBLX", "DDOG", "NET",
]

# Nasdaq-100 — sesgo tech, alta beta y mucho volumen intradía: el caladero
# preferido para estrategias de momentum.
NASDAQ100: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "COST", "ADBE", "PEP", "NFLX", "AMD", "INTC", "CMCSA", "QCOM", "TMUS",
    "INTU", "AMAT", "TXN", "BKNG", "ISRG", "AMGN", "ADI", "VRTX", "LRCX",
    "REGN", "ADP", "GILD", "MU", "PANW", "PYPL", "MELI", "MRVL", "FTNT",
    "KLAC", "ASML", "PDD", "ABNB", "CRWD", "SNPS", "CDNS", "CTAS", "ORLY",
    "ROST", "CHTR", "DASH", "WDAY", "FANG", "MAR", "ODFL", "NXPI", "PCAR",
    "MNST", "ADSK", "PAYX", "FAST", "MDLZ", "EXC", "CSGP", "AEP", "DXCM",
    "AZN", "EA", "BIIB", "CTSH", "IDXX", "VRSK", "TEAM", "ZS", "DDOG", "ILMN",
]

DOW30: List[str] = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
]

MAG7: List[str] = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

# ETFs sectoriales: útiles para diversificar el ranking del screener.
SECTOR_ETFS: List[str] = [
    "SPY", "QQQ", "DIA", "IWM",  # índices
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE",
    "SMH", "SOXX",  # semiconductores
    "ARKK",         # innovación
    "GDX",          # mineras
    "TLT", "HYG",   # bonos
]


UNIVERSES: Dict[str, List[str]] = {
    "S&P 500 large caps": SP500_LARGECAP,
    "Nasdaq-100": NASDAQ100,
    "Dow 30": DOW30,
    "Mag 7": MAG7,
    "ETFs sectoriales": SECTOR_ETFS,
}


def list_universes() -> List[str]:
    return list(UNIVERSES.keys())


def combine_universes(names: List[str], extra: List[str] | None = None) -> List[str]:
    """Une varios universos sin duplicados, opcionalmente añade tickers extra."""
    bag: set[str] = set()
    for n in names:
        bag.update(UNIVERSES.get(n, []))
    if extra:
        bag.update([t.upper().strip() for t in extra if t.strip()])
    return sorted(bag)
