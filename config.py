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

# market_data_engine.py: a separate, quality-checked normalized snapshot
# layer (Stage 2 of the roadmap) — independent of fetch_snapshot/market_data.py,
# not wired into the AI pipeline yet. market_data_history.jsonl is a compact
# top-of-book + instrument-spec log (see market_data_history.py), the same
# append/trim pattern as oi_history.py above.
MARKET_DATA_HISTORY_FILE = Path(__file__).resolve().parent / "market_data_history.jsonl"
MARKET_DATA_HISTORY_MAX_ENTRIES = int(os.getenv("MARKET_DATA_HISTORY_MAX_ENTRIES", "500"))
MARKET_DATA_TRADES_LIMIT = int(os.getenv("MARKET_DATA_TRADES_LIMIT", "20"))
# A candle gap/staleness beyond this multiple of the timeframe's own interval
# is treated as a hard data-quality failure (NO_TRADE), not just a warning.
MARKET_DATA_MAX_CANDLE_GAP_MULTIPLIER = float(os.getenv("MARKET_DATA_MAX_CANDLE_GAP_MULTIPLIER", "2.0"))
MARKET_DATA_MAX_ORDERBOOK_AGE_SECONDS = float(os.getenv("MARKET_DATA_MAX_ORDERBOOK_AGE_SECONDS", "15"))
MARKET_DATA_MAX_SOURCE_TIME_SKEW_SECONDS = float(os.getenv("MARKET_DATA_MAX_SOURCE_TIME_SKEW_SECONDS", "60"))

# feature_engine.py (Stage 3): computes trend/momentum/volatility/volume/
# futures/orderbook features from a market_data_engine.MarketDataSnapshot.
# Not wired into the AI pipeline yet — same standalone-first approach as
# market_data_engine.py in Stage 2.
FEATURE_EMA_SLOPE_LOOKBACK = int(os.getenv("FEATURE_EMA_SLOPE_LOOKBACK", "5"))
FEATURE_RATE_OF_CHANGE_LOOKBACK = int(os.getenv("FEATURE_RATE_OF_CHANGE_LOOKBACK", "5"))
FEATURE_ADX_PERIOD = int(os.getenv("FEATURE_ADX_PERIOD", "14"))
FEATURE_SUPERTREND_PERIOD = int(os.getenv("FEATURE_SUPERTREND_PERIOD", "10"))
FEATURE_SUPERTREND_MULTIPLIER = float(os.getenv("FEATURE_SUPERTREND_MULTIPLIER", "3.0"))
FEATURE_STRUCTURE_SWING_WINDOW = int(os.getenv("FEATURE_STRUCTURE_SWING_WINDOW", "3"))
FEATURE_ATR_PERCENTILE_LOOKBACK = int(os.getenv("FEATURE_ATR_PERCENTILE_LOOKBACK", "100"))
FEATURE_RANGE_COMPRESSION_PERCENTILE = float(os.getenv("FEATURE_RANGE_COMPRESSION_PERCENTILE", "20"))
FEATURE_VOLATILITY_EXPANSION_PERCENTILE = float(os.getenv("FEATURE_VOLATILITY_EXPANSION_PERCENTILE", "80"))
FEATURE_REALIZED_VOL_WINDOW = int(os.getenv("FEATURE_REALIZED_VOL_WINDOW", "20"))
FEATURE_VOLUME_SMA_WINDOW = int(os.getenv("FEATURE_VOLUME_SMA_WINDOW", "20"))
FEATURE_VOLUME_SPIKE_MULTIPLIER = float(os.getenv("FEATURE_VOLUME_SPIKE_MULTIPLIER", "2.0"))
FEATURE_VOLUME_TREND_FAST_WINDOW = int(os.getenv("FEATURE_VOLUME_TREND_FAST_WINDOW", "10"))
FEATURE_VOLUME_TREND_SLOW_WINDOW = int(os.getenv("FEATURE_VOLUME_TREND_SLOW_WINDOW", "50"))
FEATURE_LARGE_WALL_MULTIPLIER = float(os.getenv("FEATURE_LARGE_WALL_MULTIPLIER", "5.0"))
FEATURE_ORDERBOOK_IMBALANCE_MIN_WINDOW_SECONDS = float(os.getenv("FEATURE_ORDERBOOK_IMBALANCE_MIN_WINDOW_SECONDS", "30"))
FEATURE_ORDERBOOK_IMBALANCE_MAX_WINDOW_SECONDS = float(os.getenv("FEATURE_ORDERBOOK_IMBALANCE_MAX_WINDOW_SECONDS", "120"))

# market_regime.py (Stage 4): classifies one of 10 market regimes per
# timeframe from a feature_engine.FeatureSet, purely a decision tree over
# already-computed features — no new indicator math. Not wired into the
# AI pipeline yet.
REGIME_ADX_TREND_THRESHOLD = float(os.getenv("REGIME_ADX_TREND_THRESHOLD", "25"))
REGIME_ADX_RANGE_THRESHOLD = float(os.getenv("REGIME_ADX_RANGE_THRESHOLD", "18"))
REGIME_FUNDING_EXTREME_THRESHOLD = float(os.getenv("REGIME_FUNDING_EXTREME_THRESHOLD", "0.0005"))

# strategy_engine.py (Stage 5): LLM-independent LONG/SHORT/NO_TRADE
# scoring from a feature_engine.FeatureSet + market_regime.RegimeResult,
# via a versioned, data-driven rule list (see RULES in strategy_engine.py).
# Not wired into the AI pipeline yet.
STRATEGY_RULESET_VERSION = "v1"
STRATEGY_BASELINE_SCORE = float(os.getenv("STRATEGY_BASELINE_SCORE", "50"))
STRATEGY_TIMEFRAME_ALIGNMENT_THRESHOLD = float(os.getenv("STRATEGY_TIMEFRAME_ALIGNMENT_THRESHOLD", "0.75"))
STRATEGY_RSI_OVERBOUGHT = float(os.getenv("STRATEGY_RSI_OVERBOUGHT", "70"))
STRATEGY_RSI_OVERSOLD = float(os.getenv("STRATEGY_RSI_OVERSOLD", "30"))
STRATEGY_LOW_VOLUME_RATIO = float(os.getenv("STRATEGY_LOW_VOLUME_RATIO", "0.5"))
STRATEGY_FUNDING_EXTREME_THRESHOLD = float(os.getenv("STRATEGY_FUNDING_EXTREME_THRESHOLD", "0.0005"))
STRATEGY_ORDERBOOK_IMBALANCE_THRESHOLD = float(os.getenv("STRATEGY_ORDERBOOK_IMBALANCE_THRESHOLD", "0.3"))
STRATEGY_NEAR_LEVEL_ATR_THRESHOLD = float(os.getenv("STRATEGY_NEAR_LEVEL_ATR_THRESHOLD", "0.5"))
STRATEGY_MIN_RISK_REWARD = float(os.getenv("STRATEGY_MIN_RISK_REWARD", "1.5"))
STRATEGY_BIAS_MARGIN = float(os.getenv("STRATEGY_BIAS_MARGIN", "15"))
STRATEGY_BIAS_MIN_SCORE = float(os.getenv("STRATEGY_BIAS_MIN_SCORE", "55"))
STRATEGY_NO_TRADE_DOMINANT_SCORE = float(os.getenv("STRATEGY_NO_TRADE_DOMINANT_SCORE", "60"))
STRATEGY_SCALPING_PRIMARY_TIMEFRAME = os.getenv("STRATEGY_SCALPING_PRIMARY_TIMEFRAME", "5m")
STRATEGY_SWING_PRIMARY_TIMEFRAME = os.getenv("STRATEGY_SWING_PRIMARY_TIMEFRAME", "1h")
STRATEGY_SWING_FUNDING_WEIGHT_MULTIPLIER = float(os.getenv("STRATEGY_SWING_FUNDING_WEIGHT_MULTIPLIER", "2.0"))

# Prediction accuracy tracking (see prediction_tracker.py). Every real (ok)
# AI verdict is logged with its trade plan; once `horizon` seconds have
# passed since a given prediction, the next snapshot fetch walks the actual
# BingX candle path over that window (outcome_simulator.py) to see whether
# stop loss or a take profit was hit first — not just whether price ended up
# higher/lower at a fixed deadline.
PREDICTION_HISTORY_FILE = Path(__file__).resolve().parent / "predictions.jsonl"
PREDICTION_HORIZON_SECONDS = {
    "scalping": int(os.getenv("PREDICTION_HORIZON_SCALPING_SECONDS", str(30 * 60))),
    "swing": int(os.getenv("PREDICTION_HORIZON_SWING_SECONDS", str(8 * 3600))),
}
PREDICTION_HISTORY_MAX_ENTRIES = int(os.getenv("PREDICTION_HISTORY_MAX_ENTRIES", "2000"))

# Outcome simulation (see outcome_simulator.py): fixed round-trip cost
# assumptions used to convert a raw price move into an R-multiple, since we
# have no access to a real account's actual fee tier/fill quality.
TRADING_COMMISSION_PCT = float(os.getenv("TRADING_COMMISSION_PCT", "0.0005"))  # 0.05% per side
TRADING_SLIPPAGE_PCT = float(os.getenv("TRADING_SLIPPAGE_PCT", "0.0002"))  # 0.02%
OUTCOME_KLINE_INTERVAL = os.getenv("OUTCOME_KLINE_INTERVAL", "1m").strip()
MIN_SAMPLE_FOR_STATS = int(os.getenv("MIN_SAMPLE_FOR_STATS", "3"))

# AI analysis (optional): up to 2 OpenAI-compatible chat-completions APIs used
# to analyze the market snapshot automatically and compare their signals side
# by side. Models with no API key set are skipped; the whole step is skipped
# with --no-ai. Both share the same request/response contract regardless of
# which underlying provider they proxy to.
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://apinet.cloud/v1").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-5.1-all").strip()
AI_LABEL = os.getenv("AI_LABEL", "NeuroTrade-gpt").strip()

AI2_API_KEY = os.getenv("AI2_API_KEY", "").strip()
AI2_BASE_URL = os.getenv("AI2_BASE_URL", AI_BASE_URL).strip()
AI2_MODEL = os.getenv("AI2_MODEL", "claude-opus-5").strip()
AI2_LABEL = os.getenv("AI2_LABEL", "NeuroTrade-claude").strip()

AI_REQUEST_TIMEOUT = float(os.getenv("AI_REQUEST_TIMEOUT", "180"))
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

# Stamped onto every AI call's log entries (see app_log.py) so historical
# stats/logs can later be filtered by which prompt/schema produced them —
# bump PROMPT_VERSION whenever SYSTEM_PROMPTS/JSON_INSTRUCTIONS changes
# meaningfully, and SCHEMA_VERSION whenever ai_schema.TradePlan's shape does.
PROMPT_VERSION = "v2-json-schema"
SCHEMA_VERSION = 1

# User's risk-management settings (see risk_manager.py, risk_settings_store.py).
# Local JSON file, no API keys — created with sensible defaults on first
# read if it doesn't exist yet. Not committed (per-user local state, same
# treatment as .env/dashboard_cache.pkl).
RISK_SETTINGS_FILE = Path(__file__).resolve().parent / "risk_settings.json"

# journal_db.py: standalone SQLite journal — one durable, queryable record
# per pipeline stage (market snapshot -> features -> strategy score -> AI
# predictions -> consensus trade plan -> outcome), plus schema-only
# paper/real order + position/fill tables for a future execution stage.
# Additive/parallel to prediction_tracker.py's predictions.jsonl, not a
# replacement (Stage 7 of the roadmap). Not wired into the live refresh
# cycle yet — a standalone, fully-tested module first, same approach as
# market_data_engine.py/feature_engine.py/market_regime.py/strategy_engine.py.
JOURNAL_DB_FILE = Path(__file__).resolve().parent / "journal.db"
JOURNAL_DB_BUSY_TIMEOUT_MS = int(os.getenv("JOURNAL_DB_BUSY_TIMEOUT_MS", "5000"))

# paper_trading.py (Stage 8): live, tick-driven simulation of order fills and
# position exits against real BingX book-ticker prices, using risk_manager's
# fee/slippage model (not the flatter config.TRADING_COMMISSION_PCT/
# TRADING_SLIPPAGE_PCT pair below, which stays reserved for
# outcome_simulator.py/prediction_tracker.py). Not wired into a scheduler
# yet — process_tick() must be called repeatedly by something external.
PAPER_TRADING_BREAKEVEN_TRIGGER_R = float(os.getenv("PAPER_TRADING_BREAKEVEN_TRIGGER_R", "1.0"))
PAPER_TRADING_MAX_HOLD_SECONDS = {
    "scalping": int(os.getenv("PAPER_TRADING_MAX_HOLD_SCALPING_SECONDS", str(2 * 3600))),
    "swing": int(os.getenv("PAPER_TRADING_MAX_HOLD_SWING_SECONDS", str(24 * 3600))),
}
# A single flat maintenance-margin-rate approximation for paper_trading.py's
# liquidation-price modeling — real exchanges use a notional-tiered MMR
# table (higher position size = higher rate), which isn't available without
# live BingX risk-limit data (same "can't confirm live exchange specifics"
# situation as bingx_private_client.py). Reasonable for retail position sizes.
PAPER_TRADING_MAINTENANCE_MARGIN_RATE = float(os.getenv("PAPER_TRADING_MAINTENANCE_MARGIN_RATE", "0.005"))

# backtest_engine.py (Stage 9): walks historical candles through the SAME
# feature_engine/market_regime/strategy_engine pipeline live code uses (no
# separate "backtest rules") — no AI (score_strategy's decision IS the rule
# being tested), no per-bar journal_db writes (in-memory BacktestResult
# only). strategy_engine only produces a score/decision, not price levels
# (those come from the AI's TradePlan live) — BACKTEST_STOP_ATR_MULTIPLIER/
# BACKTEST_TP_LEVELS synthesize a simple, honestly-documented ATR/RR bracket
# from the same FeatureSet just computed, standing in for what the AI would
# have proposed. Execution economics (trailing trigger, max hold) reuse the
# PAPER_TRADING_* constants above for consistency, not separate values.
BACKTEST_STOP_ATR_MULTIPLIER = float(os.getenv("BACKTEST_STOP_ATR_MULTIPLIER", "1.5"))
# Each entry: (multiple of RiskSettings.min_risk_reward, close_percent) for one partial take-profit target.
# risk_manager.PositionCalculator requires at least ONE individual take-profit
# (not just the blended average) to clear min_risk_reward net of fees, or the
# scenario is rejected as FEES_TOO_HIGH — a 50/50 split at (1.0x, 2.0x) fails
# this per-target check (each target's net RR is diluted by its own
# close_percent), even though the blended RR clears the floor comfortably.
# (1.5x/40%, 3.0x/60%) reliably satisfies the per-target gate via TP2.
BACKTEST_TP_LEVELS = ((1.5, 40.0), (3.0, 60.0))
BACKTEST_KLINE_PAGE_LIMIT = int(os.getenv("BACKTEST_KLINE_PAGE_LIMIT", "1000"))

# model_weights.py (Stage 10): dynamic per-model vote weighting, derived
# from prediction_tracker.py's predictions.jsonl — the only existing
# source with a genuine, independently-scored outcome per model, not just
# the consensus winner (journal.db's trade_outcomes is 1:1 with the single
# consensus-selected plan per signal, not per model). Standalone module —
# not wired into consensus_engine.py's live voting this stage.
MODEL_WEIGHT_MIN_SAMPLE = int(os.getenv("MODEL_WEIGHT_MIN_SAMPLE", "10"))
MODEL_WEIGHT_FULL_SAMPLE = int(os.getenv("MODEL_WEIGHT_FULL_SAMPLE", "30"))
MODEL_WEIGHT_NEUTRAL = float(os.getenv("MODEL_WEIGHT_NEUTRAL", "0.5"))
MODEL_WEIGHT_MIN = float(os.getenv("MODEL_WEIGHT_MIN", "0.1"))
MODEL_WEIGHT_MAX = float(os.getenv("MODEL_WEIGHT_MAX", "1.0"))
MODEL_WEIGHT_EXPECTANCY_SCALE = float(os.getenv("MODEL_WEIGHT_EXPECTANCY_SCALE", "0.25"))
# (upper_bound_inclusive, label) — confidence 0..100 mapped to a bucket.
CONFIDENCE_BUCKETS = ((59, "LOW"), (74, "MEDIUM"), (89, "HIGH"), (100, "VERY_HIGH"))

# bingx_private_client.py / execution_engine.py (Stage 11): the FIRST code
# in this app able to place REAL orders with REAL money.
#
# BINGX_API_KEY/BINGX_API_SECRET MUST belong to an API key that is
# TRADING-ONLY (no withdrawal permission) and IP-whitelisted to this
# machine — both are BingX-ACCOUNT-level settings configured on BingX's
# own website; this code cannot enforce or verify either one. Do this
# BEFORE ever setting EXECUTION_DRY_RUN=false.
#
# Endpoint paths/params below are best-effort from public docs/SDKs
# (BingX's own docs are a JS-rendered site this tooling couldn't fully
# verify) — EXECUTION_DRY_RUN defaults to true specifically so every
# request can be built, signed, and inspected before it's ever sent.
BINGX_API_KEY = os.getenv("BINGX_API_KEY", "").strip()
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET", "").strip()
BINGX_PRIVATE_RECV_WINDOW_MS = int(os.getenv("BINGX_PRIVATE_RECV_WINDOW_MS", "5000"))
EXECUTION_DRY_RUN = os.getenv("EXECUTION_DRY_RUN", "true").strip().lower() != "false"

EXECUTION_KILL_SWITCH_FILE = Path(__file__).resolve().parent / "kill_switch.flag"
EXECUTION_MAX_TRADES_PER_DAY = int(os.getenv("EXECUTION_MAX_TRADES_PER_DAY", "5"))
EXECUTION_DAILY_LOSS_LIMIT_USDT = float(os.getenv("EXECUTION_DAILY_LOSS_LIMIT_USDT", "10"))
EXECUTION_COOLDOWN_AFTER_STOP_SECONDS = int(os.getenv("EXECUTION_COOLDOWN_AFTER_STOP_SECONDS", str(30 * 60)))
# Mirrors consensus_engine._PRICE_ZONE_TOLERANCE_FRACTION — same rationale
# (absorb quote noise/rounding at the price re-check, not a real tolerance
# for chasing price), reproduced here rather than imported since the two
# modules are meant to stay independently testable.
EXECUTION_PRICE_ZONE_TOLERANCE_FRACTION = 0.0005

# position_manager.py (Stage 12): once 2+ take-profit levels have filled,
# manage_stop_loss switches from a single breakeven jump to a real trailing
# stop that ratchets toward the current price, staying this many multiples
# of the position's original risk (R) behind it.
EXECUTION_TRAILING_STOP_R_MULTIPLE = float(os.getenv("EXECUTION_TRAILING_STOP_R_MULTIPLE", "1.0"))

# runtime/ (trading_runtime.py): the long-lived process that actually calls
# paper_trading.process_tick/execution_engine.monitor on a schedule — every
# engine before this stage was deliberately "called from outside," with no
# loop of its own. The two modes are mutually exclusive on purpose:
# "PAPER" runs ONLY paper_trading.process_tick — it must stay entirely
# inert with respect to real_orders/positions(source="REAL"), even though
# EXECUTION_DRY_RUN would short-circuit any actual network call, so
# execution_engine.monitor is never invoked in this mode at all.
# "MONITOR_ONLY" runs ONLY execution_engine.monitor (already-open REAL
# positions + data quality) — no paper simulation. Neither mode places a
# new real order (execution_engine.confirm_and_execute stays a separate,
# manual action) — so there's no distinct "SEMI_AUTO"/"AUTO" runtime mode
# yet, both would behave identically to plain monitoring today.
RUNTIME_MODE = os.getenv("RUNTIME_MODE", "PAPER").strip().upper()
RUNTIME_FAST_INTERVAL_SECONDS = float(os.getenv("RUNTIME_FAST_INTERVAL_SECONDS", "2"))
RUNTIME_MEDIUM_INTERVAL_SECONDS = float(os.getenv("RUNTIME_MEDIUM_INTERVAL_SECONDS", "10"))
RUNTIME_LOCK_FILE = Path(__file__).resolve().parent / "runtime.lock"
RUNTIME_HEARTBEAT_FILE = Path(__file__).resolve().parent / "runtime_heartbeat.json"
RUNTIME_HEARTBEAT_STALE_SECONDS = float(os.getenv("RUNTIME_HEARTBEAT_STALE_SECONDS", "15"))

# runtime/ai_cycle.py (Stage: online paper trading): how often the runtime
# fetches fresh data, asks the AI, and — if the consensus is actionable —
# opens a paper position from it. Defaults to this process's own trading
# mode's prediction horizon (no point re-asking mid-horizon, before the
# last prediction could even resolve). Skipped entirely (not just
# rescheduled) while a pending/open PAPER order already exists for the
# symbol — see TradingRuntime.run_once.
RUNTIME_AI_CYCLE_INTERVAL_SECONDS = float(
    os.getenv(
        "RUNTIME_AI_CYCLE_INTERVAL_SECONDS",
        str(PREDICTION_HORIZON_SECONDS.get(TRADING_MODE, PREDICTION_HORIZON_SECONDS["scalping"])),
    )
)
# An AI cycle's 2 parallel model calls (up to AI_REQUEST_TIMEOUT each) plus
# the rest of the pipeline can legitimately block run_once() far longer
# than RUNTIME_HEARTBEAT_STALE_SECONDS — heartbeat_status() uses this wider
# budget instead, but only while the last-written heartbeat's own
# `activity` field says a cycle was in flight, so a genuine hang still
# alerts using the normal (short) threshold once heartbeat writes stop.
RUNTIME_HEARTBEAT_BUSY_STALE_SECONDS = float(os.getenv("RUNTIME_HEARTBEAT_BUSY_STALE_SECONDS", "240"))

# Dashboard's "Обновить" button for TRADING_MODE touches this file instead
# of paying for its own independent AI call — trading_runtime.py already
# pays for that mode's analysis on its own schedule, so a second, separate
# call would just be the same money spent twice for near-duplicate
# results. TradingRuntime.run_once() treats this file's mere existence as
# "run the AI cycle now, interval or not" and deletes it once consumed.
RUNTIME_AI_CYCLE_TRIGGER_FILE = Path(__file__).resolve().parent / "ai_cycle_trigger.flag"


@dataclass(frozen=True)
class AIModelConfig:
    label: str
    api_key: str
    base_url: str
    model: str


AI_MODELS: list[AIModelConfig] = [
    AIModelConfig(AI_LABEL, AI_API_KEY, AI_BASE_URL, AI_MODEL),
    AIModelConfig(AI2_LABEL, AI2_API_KEY, AI2_BASE_URL, AI2_MODEL),
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
