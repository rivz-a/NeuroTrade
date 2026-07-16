"""SQL DDL for the journal database — pure schema, no I/O of its own.

One `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` statement per
table, in dependency order (parents before children), aggregated into
`ALL_STATEMENTS` and applied by `create_schema()`. See journal_db.py for the
connection management, serialization, and insert/read functions that use this
schema — this file never touches disk itself.

Pipeline chain (mirrors the FK chain below): market_snapshots -> feature_snapshots
-> strategy_scores (market regime denormalized in) -> ai_predictions (N per
score, one per AI model) -> trade_plans (1 per score — the consensus-selected
final decision, with risk-manager position sizing denormalized in) ->
trade_outcomes (1 per trade_plan). paper_orders/positions/fills are populated
by paper_trading.py (Stage 8); real_orders stays schema-only (no real BingX
order placement exists in this app).
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 3

_MARKET_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol                      TEXT NOT NULL,
    timestamp                   REAL NOT NULL,
    price                       REAL,
    bid                         REAL,
    ask                         REAL,
    spread                      REAL,
    spread_percent              REAL,
    funding_rate                REAL,
    open_interest               REAL,
    volume_24h                  REAL,
    data_quality                TEXT NOT NULL CHECK (data_quality IN ('GOOD','DEGRADED','NO_TRADE')),
    quality_issues_json         TEXT NOT NULL DEFAULT '[]',
    timeframes_json             TEXT NOT NULL DEFAULT '{}',
    funding_history_json        TEXT NOT NULL DEFAULT '[]',
    open_interest_history_json  TEXT NOT NULL DEFAULT '[]',
    orderbook_json              TEXT,
    orderbook_history_json      TEXT NOT NULL DEFAULT '[]',
    recent_trades_json          TEXT NOT NULL DEFAULT '[]',
    instrument_rules_json       TEXT,
    created_at                  REAL NOT NULL
)
"""
_MARKET_SNAPSHOTS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_market_snapshots_symbol_ts ON market_snapshots(symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_market_snapshots_quality ON market_snapshots(data_quality)",
)

_FEATURE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS feature_snapshots (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_snapshot_id              INTEGER NOT NULL REFERENCES market_snapshots(id) ON DELETE CASCADE,
    symbol                          TEXT NOT NULL,
    timestamp                       REAL NOT NULL,
    data_quality                    TEXT NOT NULL CHECK (data_quality IN ('GOOD','DEGRADED','NO_TRADE')),
    timeframe_alignment             REAL NOT NULL,
    distance_to_support_atr         REAL,
    distance_to_resistance_atr      REAL,
    trend_json                      TEXT NOT NULL DEFAULT '{}',
    momentum_json                   TEXT NOT NULL DEFAULT '{}',
    volatility_json                 TEXT NOT NULL DEFAULT '{}',
    volume_json                     TEXT NOT NULL DEFAULT '{}',
    trade_flow_buy_volume           REAL NOT NULL,
    trade_flow_sell_volume          REAL NOT NULL,
    trade_flow_delta                REAL NOT NULL,
    trade_flow_cvd                  REAL NOT NULL,
    funding_rate                    REAL,
    funding_change                  REAL,
    open_interest                   REAL,
    open_interest_change            REAL,
    open_interest_change_pct        REAL,
    oi_price_regime                 TEXT CHECK (oi_price_regime IN ('NEW_LONGS','SHORT_COVERING','NEW_SHORTS','LONG_COVERING') OR oi_price_regime IS NULL),
    orderbook_imbalance             REAL,
    orderbook_imbalance_avg_30_120s REAL,
    orderbook_spread                REAL,
    orderbook_spread_change         REAL,
    orderbook_microprice            REAL,
    orderbook_has_large_wall        INTEGER,
    created_at                      REAL NOT NULL
)
"""
_FEATURE_SNAPSHOTS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_ts ON feature_snapshots(symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_feature_snapshots_market_id ON feature_snapshots(market_snapshot_id)",
)

_STRATEGY_SCORES = """
CREATE TABLE IF NOT EXISTS strategy_scores (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_snapshot_id       INTEGER NOT NULL REFERENCES feature_snapshots(id) ON DELETE CASCADE,
    symbol                    TEXT NOT NULL,
    timestamp                 REAL NOT NULL,
    mode                      TEXT NOT NULL CHECK (mode IN ('scalping','swing')),
    ruleset_version           TEXT NOT NULL,
    long_score                REAL NOT NULL,
    short_score               REAL NOT NULL,
    no_trade_score            REAL NOT NULL,
    decision                  TEXT NOT NULL CHECK (decision IN ('LONG_BIAS','SHORT_BIAS','NO_TRADE','NEUTRAL')),
    quality                   TEXT NOT NULL CHECK (quality IN ('A','B','C','D')),
    contributions_json        TEXT NOT NULL DEFAULT '[]',
    regime                    TEXT NOT NULL CHECK (regime IN
        ('TREND_UP','TREND_DOWN','RANGE','BREAKOUT_UP','BREAKOUT_DOWN',
         'VOLATILITY_COMPRESSION','VOLATILITY_EXPANSION','REVERSAL_RISK','UNSTABLE','NO_DATA')),
    regime_by_timeframe_json  TEXT NOT NULL DEFAULT '{}',
    regime_reasons_json       TEXT NOT NULL DEFAULT '[]',
    regime_strategy_hint      TEXT NOT NULL DEFAULT '',
    created_at                REAL NOT NULL
)
"""
_STRATEGY_SCORES_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_strategy_scores_symbol_mode_ts ON strategy_scores(symbol, mode, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_strategy_scores_decision ON strategy_scores(decision)",
    "CREATE INDEX IF NOT EXISTS idx_strategy_scores_regime ON strategy_scores(regime)",
    "CREATE INDEX IF NOT EXISTS idx_strategy_scores_feature_id ON strategy_scores(feature_snapshot_id)",
)

_AI_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS ai_predictions (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_score_id               INTEGER NOT NULL REFERENCES strategy_scores(id) ON DELETE CASCADE,
    symbol                          TEXT NOT NULL,
    mode                            TEXT NOT NULL CHECK (mode IN ('scalping','swing')),
    label                           TEXT NOT NULL,
    model                           TEXT NOT NULL,
    created_at_epoch                REAL,
    latency_seconds                 REAL,
    content                         TEXT,
    error                           TEXT,
    error_code                      TEXT,
    repaired                        INTEGER NOT NULL DEFAULT 0,
    ok                               INTEGER NOT NULL,
    signal                          TEXT CHECK (signal IN ('LONG','SHORT','WAIT') OR signal IS NULL),
    entry_status                    TEXT,
    confidence                      INTEGER,
    market_regime                   TEXT,
    entry_type                      TEXT,
    entry_from                      REAL,
    entry_to                        REAL,
    entry_trigger                   TEXT,
    stop_loss                       REAL,
    take_profits_json               TEXT NOT NULL DEFAULT '[]',
    time_horizon_minutes            INTEGER,
    valid_for_minutes               INTEGER,
    reasons_json                    TEXT NOT NULL DEFAULT '[]',
    risks_json                      TEXT NOT NULL DEFAULT '[]',
    invalidation_conditions_json    TEXT NOT NULL DEFAULT '[]',
    wait_conditions_json            TEXT NOT NULL DEFAULT '[]',
    contradictions_json             TEXT NOT NULL DEFAULT '[]',
    missing_context_json            TEXT NOT NULL DEFAULT '[]',
    summary                         TEXT NOT NULL DEFAULT '',
    validation_status               TEXT CHECK (validation_status IN ('valid','warning','rejected') OR validation_status IS NULL),
    validation_issues_json          TEXT NOT NULL DEFAULT '[]',
    inserted_at                     REAL NOT NULL
)
"""
_AI_PREDICTIONS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_ai_predictions_score_id ON ai_predictions(strategy_score_id)",
    "CREATE INDEX IF NOT EXISTS idx_ai_predictions_label_mode ON ai_predictions(label, mode)",
    "CREATE INDEX IF NOT EXISTS idx_ai_predictions_signal ON ai_predictions(signal)",
    "CREATE INDEX IF NOT EXISTS idx_ai_predictions_model ON ai_predictions(model)",
)

_TRADE_PLANS = """
CREATE TABLE IF NOT EXISTS trade_plans (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_score_id             INTEGER NOT NULL UNIQUE REFERENCES strategy_scores(id) ON DELETE CASCADE,
    symbol                        TEXT NOT NULL,
    mode                          TEXT NOT NULL CHECK (mode IN ('scalping','swing')),
    timestamp                     REAL NOT NULL,
    overall_signal                TEXT NOT NULL CHECK (overall_signal IN ('LONG','SHORT','WAIT')),
    consensus_state                TEXT NOT NULL CHECK (consensus_state IN ('strong','moderate','weak','conflict','insufficient_data')),
    agreement_fraction             REAL NOT NULL,
    agreeing_count                 INTEGER NOT NULL,
    vote_count                     INTEGER NOT NULL,
    total_models                   INTEGER NOT NULL,
    avg_confidence                  REAL,
    trade_permission                TEXT NOT NULL,
    trade_permission_reason         TEXT NOT NULL DEFAULT '',
    reasons_json                    TEXT NOT NULL DEFAULT '[]',
    risks_json                      TEXT NOT NULL DEFAULT '[]',
    wait_or_invalidation_json       TEXT NOT NULL DEFAULT '[]',
    source_label                    TEXT,
    entry_status                    TEXT,
    entry_type                      TEXT,
    entry_from                      REAL,
    entry_to                        REAL,
    stop_loss                       REAL,
    take_profits_json               TEXT NOT NULL DEFAULT '[]',
    risk_reward_tp1                 REAL,
    time_horizon_minutes            INTEGER,
    valid_for_minutes               INTEGER,
    formed_at                       REAL,
    position_status                 TEXT,
    limited_by                      TEXT CHECK (limited_by IN ('RISK','MARGIN') OR limited_by IS NULL),
    entry_price_calc                TEXT,
    entry_zone_low                  TEXT,
    entry_zone_high                 TEXT,
    entry_price_mode                TEXT,
    stop_loss_calc                  TEXT,
    stop_distance                   TEXT,
    account_balance_usdt            TEXT,
    risk_percent                    TEXT,
    risk_budget_usdt                TEXT,
    leverage                        INTEGER,
    margin_mode                     TEXT CHECK (margin_mode IN ('ISOLATED','CROSS') OR margin_mode IS NULL),
    position_size_coin              TEXT,
    position_size_coin_rounded      TEXT,
    position_notional_usdt          TEXT,
    required_margin_usdt            TEXT,
    max_allowed_margin_usdt         TEXT,
    entry_fee_usdt                  TEXT,
    stop_exit_fee_usdt              TEXT,
    slippage_estimate_usdt          TEXT,
    total_expected_loss_usdt        TEXT,
    actual_risk_percent             TEXT,
    take_profit_results_json        TEXT NOT NULL DEFAULT '[]',
    primary_target_label            TEXT,
    blended_json                    TEXT,
    free_balance_after_open_usdt    TEXT,
    position_warnings_json          TEXT NOT NULL DEFAULT '[]',
    position_issues_json            TEXT NOT NULL DEFAULT '[]',
    bingx_fields_json               TEXT,
    created_at                      REAL NOT NULL
)
"""
_TRADE_PLANS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_symbol_mode_ts ON trade_plans(symbol, mode, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_signal ON trade_plans(overall_signal)",
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_permission ON trade_plans(trade_permission)",
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_position_status ON trade_plans(position_status)",
)

_PAPER_ORDERS = """
CREATE TABLE IF NOT EXISTS paper_orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_plan_id       INTEGER UNIQUE REFERENCES trade_plans(id) ON DELETE SET NULL,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('LONG','SHORT')),
    order_type          TEXT NOT NULL CHECK (order_type IN ('MARKET','LIMIT','TRIGGER')),
    status              TEXT NOT NULL CHECK (status IN ('PENDING','OPEN','FILLED','PARTIALLY_FILLED','CANCELLED','REJECTED','EXPIRED')),
    price               TEXT,
    trigger_price       TEXT,
    entry_from          REAL,
    entry_to            REAL,
    quantity            TEXT NOT NULL,
    filled_quantity     TEXT NOT NULL DEFAULT '0',
    leverage            INTEGER,
    margin_mode         TEXT CHECK (margin_mode IN ('ISOLATED','CROSS') OR margin_mode IS NULL),
    stop_loss           TEXT,
    take_profit         TEXT,
    take_profits_json   TEXT NOT NULL DEFAULT '[]',
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    closed_at           REAL,
    notes               TEXT
)
"""
_PAPER_ORDERS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_paper_orders_symbol_status ON paper_orders(symbol, status)",
    "CREATE INDEX IF NOT EXISTS idx_paper_orders_trade_plan_id ON paper_orders(trade_plan_id)",
)

_REAL_ORDERS = """
CREATE TABLE IF NOT EXISTS real_orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_plan_id       INTEGER REFERENCES trade_plans(id) ON DELETE SET NULL,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('LONG','SHORT')),
    order_type          TEXT NOT NULL CHECK (order_type IN ('MARKET','LIMIT','TRIGGER')),
    status              TEXT NOT NULL CHECK (status IN ('PENDING','OPEN','FILLED','PARTIALLY_FILLED','CANCELLED','REJECTED','EXPIRED')),
    price               TEXT,
    trigger_price       TEXT,
    quantity            TEXT NOT NULL,
    filled_quantity     TEXT NOT NULL DEFAULT '0',
    leverage            INTEGER,
    margin_mode         TEXT CHECK (margin_mode IN ('ISOLATED','CROSS') OR margin_mode IS NULL),
    stop_loss           TEXT,
    take_profit         TEXT,
    take_profits_json   TEXT NOT NULL DEFAULT '[]',
    label               TEXT,
    exchange_order_id   TEXT UNIQUE,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    closed_at           REAL,
    notes               TEXT
)
"""
_REAL_ORDERS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_real_orders_symbol_status ON real_orders(symbol, status)",
    "CREATE INDEX IF NOT EXISTS idx_real_orders_trade_plan_id ON real_orders(trade_plan_id)",
)

_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_plan_id        INTEGER REFERENCES trade_plans(id) ON DELETE SET NULL,
    paper_order_id       INTEGER REFERENCES paper_orders(id) ON DELETE SET NULL,
    real_order_id        INTEGER REFERENCES real_orders(id) ON DELETE SET NULL,
    source               TEXT NOT NULL CHECK (source IN ('PAPER','REAL')),
    symbol               TEXT NOT NULL,
    side                 TEXT NOT NULL CHECK (side IN ('LONG','SHORT')),
    status               TEXT NOT NULL CHECK (status IN ('OPEN','CLOSED','LIQUIDATED')),
    entry_price          TEXT NOT NULL,
    quantity             TEXT NOT NULL,
    leverage              INTEGER,
    margin_mode           TEXT CHECK (margin_mode IN ('ISOLATED','CROSS') OR margin_mode IS NULL),
    stop_loss             TEXT,
    take_profit           TEXT,
    exit_price            TEXT,
    realized_pnl_usdt     TEXT,
    unrealized_pnl_usdt   TEXT,
    fees_paid_usdt        TEXT,
    opened_at             REAL NOT NULL,
    closed_at             REAL,
    notes                 TEXT
)
"""
_POSITIONS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status)",
    "CREATE INDEX IF NOT EXISTS idx_positions_source ON positions(source)",
    "CREATE INDEX IF NOT EXISTS idx_positions_trade_plan_id ON positions(trade_plan_id)",
)

_FILLS = """
CREATE TABLE IF NOT EXISTS fills (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id         INTEGER REFERENCES positions(id) ON DELETE CASCADE,
    paper_order_id      INTEGER REFERENCES paper_orders(id) ON DELETE SET NULL,
    real_order_id       INTEGER REFERENCES real_orders(id) ON DELETE SET NULL,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('LONG','SHORT')),
    fill_type           TEXT NOT NULL CHECK (fill_type IN ('ENTRY','TAKE_PROFIT','STOP_LOSS','TIMEOUT','MANUAL_CLOSE','LIQUIDATION','REGIME_CHANGE','DATA_QUALITY')),
    label               TEXT,
    price               TEXT NOT NULL,
    quantity            TEXT NOT NULL,
    fee_usdt            TEXT,
    realized_pnl_usdt   TEXT,
    filled_at           REAL NOT NULL
)
"""
_FILLS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_fills_position_id ON fills(position_id)",
    "CREATE INDEX IF NOT EXISTS idx_fills_symbol_time ON fills(symbol, filled_at)",
)

_TRADE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS trade_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_plan_id       INTEGER NOT NULL UNIQUE REFERENCES trade_plans(id) ON DELETE CASCADE,
    symbol              TEXT NOT NULL,
    mode                TEXT NOT NULL CHECK (mode IN ('scalping','swing')),
    status               TEXT NOT NULL CHECK (status IN ('PENDING','EVALUATED','SKIPPED')) DEFAULT 'PENDING',
    entry_price          REAL,
    price_after_1m       REAL,
    price_after_5m       REAL,
    price_after_15m      REAL,
    price_after_30m      REAL,
    exit_reason          TEXT CHECK (exit_reason IN ('SL','TP1','TP2','TP3','TIMEOUT','AMBIGUOUS') OR exit_reason IS NULL),
    exit_price           REAL,
    exit_time             REAL,
    duration_seconds      REAL,
    r_multiple            REAL,
    mfe_r                 REAL,
    mae_r                 REAL,
    commission_usdt       REAL,
    slippage_usdt         REAL,
    realized_pnl_usdt     REAL,
    evaluated_at           REAL,
    created_at             REAL NOT NULL
)
"""
_TRADE_OUTCOMES_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_trade_outcomes_symbol_mode_status ON trade_outcomes(symbol, mode, status)",
    "CREATE INDEX IF NOT EXISTS idx_trade_outcomes_exit_reason ON trade_outcomes(exit_reason)",
    "CREATE INDEX IF NOT EXISTS idx_trade_outcomes_r_multiple ON trade_outcomes(r_multiple)",
)

_MODEL_STATISTICS = """
CREATE TABLE IF NOT EXISTS model_statistics (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at                REAL NOT NULL,
    model_label                TEXT NOT NULL,
    mode                       TEXT NOT NULL CHECK (mode IN ('scalping','swing')),
    window_start                REAL NOT NULL,
    window_end                  REAL NOT NULL,
    total_predictions           INTEGER NOT NULL,
    evaluated_count              INTEGER NOT NULL,
    wins                        INTEGER NOT NULL,
    win_rate                    REAL,
    expectancy_r                 REAL,
    median_r                    REAL,
    profit_factor                REAL,
    profit_factor_undefined      INTEGER NOT NULL DEFAULT 0,
    max_drawdown_r               REAL,
    avg_mfe_r                    REAL,
    avg_mae_r                    REAL,
    avg_duration_seconds         REAL,
    exit_reason_counts_json      TEXT NOT NULL DEFAULT '{}',
    low_sample                   INTEGER NOT NULL DEFAULT 0
)
"""
_MODEL_STATISTICS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_model_statistics_label_mode_time ON model_statistics(model_label, mode, computed_at)",
)

_STRATEGY_STATISTICS = """
CREATE TABLE IF NOT EXISTS strategy_statistics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at          REAL NOT NULL,
    ruleset_version      TEXT NOT NULL,
    mode                TEXT NOT NULL CHECK (mode IN ('scalping','swing')),
    decision            TEXT CHECK (decision IN ('LONG_BIAS','SHORT_BIAS','NO_TRADE','NEUTRAL') OR decision IS NULL),
    regime              TEXT CHECK (regime IN
        ('TREND_UP','TREND_DOWN','RANGE','BREAKOUT_UP','BREAKOUT_DOWN',
         'VOLATILITY_COMPRESSION','VOLATILITY_EXPANSION','REVERSAL_RISK','UNSTABLE','NO_DATA') OR regime IS NULL),
    window_start         REAL NOT NULL,
    window_end           REAL NOT NULL,
    total_signals        INTEGER NOT NULL,
    evaluated_count       INTEGER NOT NULL,
    wins                 REAL NOT NULL,
    win_rate             REAL,
    expectancy_r          REAL,
    median_r             REAL,
    profit_factor         REAL,
    low_sample            INTEGER NOT NULL DEFAULT 0
)
"""
_STRATEGY_STATISTICS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_strategy_statistics_version_mode_time ON strategy_statistics(ruleset_version, mode, computed_at)",
    "CREATE INDEX IF NOT EXISTS idx_strategy_statistics_regime ON strategy_statistics(regime)",
)

_SYSTEM_EVENTS = """
CREATE TABLE IF NOT EXISTS system_events (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             REAL NOT NULL,
    level                 TEXT NOT NULL CHECK (level IN ('DEBUG','INFO','WARNING','ERROR','CRITICAL')),
    event_code            TEXT NOT NULL,
    source_module         TEXT NOT NULL,
    symbol                TEXT,
    message               TEXT NOT NULL,
    details_json          TEXT NOT NULL DEFAULT '{}',
    market_snapshot_id    INTEGER REFERENCES market_snapshots(id) ON DELETE SET NULL,
    feature_snapshot_id   INTEGER REFERENCES feature_snapshots(id) ON DELETE SET NULL,
    strategy_score_id     INTEGER REFERENCES strategy_scores(id) ON DELETE SET NULL,
    trade_plan_id         INTEGER REFERENCES trade_plans(id) ON DELETE SET NULL
)
"""
_SYSTEM_EVENTS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_system_events_time ON system_events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_system_events_level ON system_events(level)",
    "CREATE INDEX IF NOT EXISTS idx_system_events_code ON system_events(event_code)",
)

# Dependency order: every table before any table that REFERENCES it, and
# every table's indices immediately after its own CREATE TABLE.
ALL_STATEMENTS: tuple[str, ...] = (
    _MARKET_SNAPSHOTS,
    *_MARKET_SNAPSHOTS_IDX,
    _FEATURE_SNAPSHOTS,
    *_FEATURE_SNAPSHOTS_IDX,
    _STRATEGY_SCORES,
    *_STRATEGY_SCORES_IDX,
    _AI_PREDICTIONS,
    *_AI_PREDICTIONS_IDX,
    _TRADE_PLANS,
    *_TRADE_PLANS_IDX,
    _PAPER_ORDERS,
    *_PAPER_ORDERS_IDX,
    _REAL_ORDERS,
    *_REAL_ORDERS_IDX,
    _POSITIONS,
    *_POSITIONS_IDX,
    _FILLS,
    *_FILLS_IDX,
    _TRADE_OUTCOMES,
    *_TRADE_OUTCOMES_IDX,
    _MODEL_STATISTICS,
    *_MODEL_STATISTICS_IDX,
    _STRATEGY_STATISTICS,
    *_STRATEGY_STATISTICS_IDX,
    _SYSTEM_EVENTS,
    *_SYSTEM_EVENTS_IDX,
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Idempotent — safe to call on every startup / at the top of every test."""
    with conn:
        for statement in ALL_STATEMENTS:
            conn.execute(statement)
