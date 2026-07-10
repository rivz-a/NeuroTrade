"""Configuration loading for Market Snapshot Copier.

Reads environment variables (via .env) for the manual position block, BingX
tuning defaults, and (optionally) the AI API used to auto-analyze the
collected snapshot. BingX access itself is always unauthenticated (public
market-data endpoints only) and nothing is ever traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

VALID_POSITIONS = {"NONE", "LONG", "SHORT"}

BASE_URL = "https://open-api.bingx.com"
SYMBOL = os.getenv("SYMBOL", "ETHUSDT").upper().replace("-", "").replace("/", "")
EXCHANGE_NAME = "BingX"

# Analysis style: "scalping" (tight 1m/5m-based stops, minutes-long holds) or
# "swing" (wider stops/targets anchored to 15m/1h structure, hours-long holds).
VALID_TRADING_MODES = {"scalping", "swing"}
TRADING_MODE = os.getenv("TRADING_MODE", "scalping").strip().lower()
if TRADING_MODE not in VALID_TRADING_MODES:
    TRADING_MODE = "scalping"
TIMEFRAMES = ["1m", "5m", "15m", "1h"]
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "220"))
# Levels actually fetched from the order book (used to compute the bid/ask
# imbalance) vs. levels printed as raw rows in the report — fetching deeper
# gives a more accurate imbalance read without bloating the AI prompt with
# dozens of raw price/qty lines.
ORDERBOOK_DEPTH = int(os.getenv("ORDERBOOK_DEPTH", "20"))
ORDERBOOK_DISPLAY_DEPTH = int(os.getenv("ORDERBOOK_DISPLAY_DEPTH", "8"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
REPORT_FILE = Path(__file__).resolve().parent / "market_snapshot.txt"

# How many past funding-rate settlements (BingX pays these roughly every 8h)
# to show, so the report reflects a trend instead of one point-in-time value.
FUNDING_HISTORY_LIMIT = int(os.getenv("FUNDING_HISTORY_LIMIT", "5"))

# Local rolling history of open-interest samples (see oi_history.py). BingX's
# public API only exposes the *current* open interest, not a history
# endpoint, so we record one sample per snapshot fetch and diff against the
# previous sample to show a real (not synthetic) OI trend.
OI_HISTORY_FILE = Path(__file__).resolve().parent / "oi_history.jsonl"
OI_HISTORY_MAX_ENTRIES = int(os.getenv("OI_HISTORY_MAX_ENTRIES", "500"))

# Prediction accuracy tracking (see prediction_tracker.py). Every real (ok)
# AI verdict is logged with the price at prediction time; once `horizon`
# seconds have passed since a given prediction, the next snapshot fetch
# scores it by comparing price movement against the called direction. This
# is directional-only (did price move the right way by the deadline) — it
# does not simulate whether stop loss/take profit would have been hit first.
PREDICTION_HISTORY_FILE = Path(__file__).resolve().parent / "predictions.jsonl"
PREDICTION_HORIZON_SECONDS = {
    "scalping": int(os.getenv("PREDICTION_HORIZON_SCALPING_SECONDS", str(30 * 60))),
    "swing": int(os.getenv("PREDICTION_HORIZON_SWING_SECONDS", str(8 * 3600))),
}
PREDICTION_HISTORY_MAX_ENTRIES = int(os.getenv("PREDICTION_HISTORY_MAX_ENTRIES", "2000"))

# AI analysis (optional): up to 3 OpenAI-compatible chat-completions APIs used
# to analyze the market snapshot automatically and compare their signals side
# by side. Models with no API key set are skipped; the whole step is skipped
# with --no-ai. All 3 share the same request/response contract regardless of
# which underlying provider they proxy to.
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://apinet.cloud/v1").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini").strip()
AI_LABEL = os.getenv("AI_LABEL", "GPT-4o mini").strip()

AI2_API_KEY = os.getenv("AI2_API_KEY", "").strip()
AI2_BASE_URL = os.getenv("AI2_BASE_URL", AI_BASE_URL).strip()
AI2_MODEL = os.getenv("AI2_MODEL", "claude-sonnet-5").strip()
AI2_LABEL = os.getenv("AI2_LABEL", "Claude Sonnet 5").strip()

AI3_API_KEY = os.getenv("AI3_API_KEY", "").strip()
AI3_BASE_URL = os.getenv("AI3_BASE_URL", AI_BASE_URL).strip()
AI3_MODEL = os.getenv("AI3_MODEL", "gemini-2.5-pro").strip()
AI3_LABEL = os.getenv("AI3_LABEL", "Gemini 2.5 Pro").strip()

AI_REQUEST_TIMEOUT = float(os.getenv("AI_REQUEST_TIMEOUT", "90"))
AI_ANALYSIS_FILE = Path(__file__).resolve().parent / "ai_analysis.txt"
AI_DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard.html"

# Local live-dashboard server (see server.py / `python main.py --serve`).
# Bound to 127.0.0.1 only by default — it can trigger real (paid) AI calls
# and shows your position/analysis, so don't change the host to 0.0.0.0
# without adding auth/a firewall in front of it. The port is just an
# uncommon default to avoid clashing with other local dev servers — it is
# NOT a security boundary; the localhost bind is what actually protects it.
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1").strip()
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "47913"))

# Server process state cache (pickle — internal use only, same-machine,
# same-Python-version). Lets `python server.py` restart (e.g. after a code
# change) and re-serve the last known results without re-fetching BingX or
# re-querying any AI model, so a restart never costs API quota by itself.
DASHBOARD_CACHE_FILE = Path(__file__).resolve().parent / "dashboard_cache.pkl"


@dataclass(frozen=True)
class AIModelConfig:
    label: str
    api_key: str
    base_url: str
    model: str


AI_MODELS: list[AIModelConfig] = [
    AIModelConfig(AI_LABEL, AI_API_KEY, AI_BASE_URL, AI_MODEL),
    AIModelConfig(AI2_LABEL, AI2_API_KEY, AI2_BASE_URL, AI2_MODEL),
    AIModelConfig(AI3_LABEL, AI3_API_KEY, AI3_BASE_URL, AI3_MODEL),
]


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class PositionConfig:
    position: str
    entry_price: float | None
    leverage: float | None
    stop_loss: float | None
    take_profit: float | None


def load_position_config() -> PositionConfig:
    raw_position = os.getenv("POSITION", "NONE").strip().upper()
    if raw_position not in VALID_POSITIONS:
        raw_position = "NONE"

    return PositionConfig(
        position=raw_position,
        entry_price=_to_float(os.getenv("ENTRY_PRICE")),
        leverage=_to_float(os.getenv("LEVERAGE")),
        stop_loss=_to_float(os.getenv("STOP_LOSS")),
        take_profit=_to_float(os.getenv("TAKE_PROFIT")),
    )


def to_bingx_symbol(symbol: str) -> str:
    """Convert e.g. 'ETHUSDT' to BingX's 'ETH-USDT' format."""
    symbol = symbol.upper().replace("-", "").replace("/", "")
    for quote in ("USDT", "USDC", "BUSD"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            return f"{base}-{quote}"
    raise ValueError(
        f"Не удалось распознать торговую пару '{symbol}'. "
        "Используйте формат вида ETHUSDT."
    )
