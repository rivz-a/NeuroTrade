"""SQLite journal — durable storage for the full pipeline of one signal:
market snapshot -> features -> strategy score (market regime denormalized
in) -> AI model predictions -> consensus trade plan (position sizing
denormalized in) -> trade outcome. Also holds schema-only paper/real order,
position, and fill tables (no populating engine yet), periodic statistics
rollups, and a general system event log. See journal_schema.py for the DDL
and the Stage 7 plan for the full design rationale.

Additive/parallel to prediction_tracker.py's predictions.jsonl — that module
is untouched, this is a separate, wider journal. Not wired into the live
server.py/main.py refresh cycle yet.

Connection management: every function takes `conn` explicitly and never
opens its own — callers open one via `connect()`/`get_connection()`. This
keeps the door open for a future thread-local connection cache without
changing any call site, since sqlite3.Connection objects aren't safe to
share across threads without care and server.py uses ThreadingHTTPServer.

`insert_*`/`update_*` never call conn.commit() themselves — only `connect()`
(or a caller managing its own connection) commits, so a whole signal's
pipeline can be inserted as one atomic transaction:

    with journal_db.connect() as conn:
        snap_id = journal_db.insert_market_snapshot(conn, snapshot)
        feat_id = journal_db.insert_feature_snapshot(conn, features, market_snapshot_id=snap_id)
        ...

positions/fills convention: `paper_order_id`/`real_order_id` are both
nullable (SQLite can't FK to "one of two tables") — `source` is the single
source of truth; populate the FK column matching `source`, leave the other
NULL.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import config
import journal_schema
from ai_client import AIAnalysisResult
from consensus_engine import ConsensusResult
from feature_engine import FeatureSet
from market_data_engine import MarketDataSnapshot
from market_regime import RegimeResult
from outcome_simulator import SimulatedOutcome
from risk_manager import BingXManualFields, PositionCalculation
from strategy_engine import ScoreResult

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Opens a fresh connection (WAL mode, foreign keys on, busy_timeout from
    config). Does NOT create the schema — call init_db() for that. Caller
    owns the connection and must close() it.
    """
    path = db_path if db_path is not None else config.JOURNAL_DB_FILE
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {config.JOURNAL_DB_BUSY_TIMEOUT_MS}")
    return conn


def init_db(conn: sqlite3.Connection | None = None, db_path: Path | None = None) -> sqlite3.Connection:
    """Idempotent — safe to call on every startup / at the top of every test."""
    if conn is None:
        conn = get_connection(db_path)
    journal_schema.create_schema(conn)
    return conn


@contextlib.contextmanager
def connect(db_path: Path | None = None):
    """with journal_db.connect() as conn: ... — opens+initializes a
    connection, commits on clean exit, rolls back and re-raises on
    exception, always closes.
    """
    conn = init_db(db_path=db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _dt_to_epoch(dt: datetime) -> float:
    return dt.timestamp()


def _epoch_to_dt(epoch: float | None) -> datetime | None:
    return datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch is not None else None


def _dec_to_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _text_to_dec(value: str | None) -> Decimal | None:
    return Decimal(value) if value is not None else None


def _norm_dec_text(value: Decimal | str | int | None) -> str | None:
    """Like _dec_to_text but also accepts str/int inputs (schema-only CRUD
    functions declare their money/quantity params as `Decimal | str` so
    callers without a Decimal handy can still pass a plain string)."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return str(value)
    return str(Decimal(str(value)))


def _json_default(obj: Any):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):  # covers pandas.Timestamp, a datetime subclass
        return obj.timestamp()
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar types
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _json_loads(text: str | None):
    return json.loads(text) if text is not None else None


def _map_asdict(mapping: dict[str, Any]) -> dict[str, dict]:
    return {k: dataclasses.asdict(v) for k, v in mapping.items()}


def _serialize_timeframes(timeframes: dict[str, dict]) -> str:
    # Full OHLCV history (the "dataframe" key) is re-fetchable from BingX by
    # timestamp — only the compact candles/quality-flag view is journaled.
    clean = {tf: {k: v for k, v in data.items() if k != "dataframe"} for tf, data in timeframes.items()}
    return _json_dumps(clean)


def _row_to_dict(
    row: sqlite3.Row,
    json_fields: tuple[str, ...] = (),
    decimal_fields: tuple[str, ...] = (),
    bool_fields: tuple[str, ...] = (),
) -> dict:
    d = dict(row)
    for f in json_fields:
        if f in d:
            out_key = f[:-5] if f.endswith("_json") else f
            d[out_key] = _json_loads(d.pop(f))
    for f in decimal_fields:
        if f in d:
            d[f] = _text_to_dec(d[f])
    for f in bool_fields:
        if f in d:
            d[f] = None if d[f] is None else bool(d[f])
    return d


# ---------------------------------------------------------------------------
# Pipeline inserts
# ---------------------------------------------------------------------------


def insert_market_snapshot(conn: sqlite3.Connection, snapshot: MarketDataSnapshot) -> int:
    cur = conn.execute(
        """
        INSERT INTO market_snapshots (
            symbol, timestamp, price, bid, ask, spread, spread_percent,
            funding_rate, open_interest, volume_24h, data_quality,
            quality_issues_json, timeframes_json, funding_history_json,
            open_interest_history_json, orderbook_json, orderbook_history_json,
            recent_trades_json, instrument_rules_json, created_at
        ) VALUES (
            :symbol, :timestamp, :price, :bid, :ask, :spread, :spread_percent,
            :funding_rate, :open_interest, :volume_24h, :data_quality,
            :quality_issues_json, :timeframes_json, :funding_history_json,
            :open_interest_history_json, :orderbook_json, :orderbook_history_json,
            :recent_trades_json, :instrument_rules_json, :created_at
        )
        """,
        {
            "symbol": snapshot.symbol,
            "timestamp": _dt_to_epoch(snapshot.timestamp),
            "price": snapshot.price,
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "spread": snapshot.spread,
            "spread_percent": snapshot.spread_percent,
            "funding_rate": snapshot.funding_rate,
            "open_interest": snapshot.open_interest,
            "volume_24h": snapshot.volume_24h,
            "data_quality": snapshot.data_quality,
            "quality_issues_json": _json_dumps(snapshot.quality_issues),
            "timeframes_json": _serialize_timeframes(snapshot.timeframes),
            "funding_history_json": _json_dumps(snapshot.funding_history),
            "open_interest_history_json": _json_dumps(snapshot.open_interest_history),
            "orderbook_json": _json_dumps(snapshot.orderbook) if snapshot.orderbook is not None else None,
            "orderbook_history_json": _json_dumps(snapshot.orderbook_history),
            "recent_trades_json": _json_dumps(snapshot.recent_trades),
            "instrument_rules_json": (
                _json_dumps(snapshot.instrument_rules) if snapshot.instrument_rules is not None else None
            ),
            "created_at": time.time(),
        },
    )
    return cur.lastrowid


def insert_feature_snapshot(conn: sqlite3.Connection, features: FeatureSet, *, market_snapshot_id: int) -> int:
    tf = features.trade_flow
    fut = features.futures
    ob = features.orderbook
    cur = conn.execute(
        """
        INSERT INTO feature_snapshots (
            market_snapshot_id, symbol, timestamp, data_quality, timeframe_alignment,
            distance_to_support_atr, distance_to_resistance_atr,
            trend_json, momentum_json, volatility_json, volume_json,
            trade_flow_buy_volume, trade_flow_sell_volume, trade_flow_delta, trade_flow_cvd,
            funding_rate, funding_change, open_interest, open_interest_change, open_interest_change_pct,
            oi_price_regime, orderbook_imbalance, orderbook_imbalance_avg_30_120s, orderbook_spread,
            orderbook_spread_change, orderbook_microprice, orderbook_has_large_wall, created_at
        ) VALUES (
            :market_snapshot_id, :symbol, :timestamp, :data_quality, :timeframe_alignment,
            :distance_to_support_atr, :distance_to_resistance_atr,
            :trend_json, :momentum_json, :volatility_json, :volume_json,
            :trade_flow_buy_volume, :trade_flow_sell_volume, :trade_flow_delta, :trade_flow_cvd,
            :funding_rate, :funding_change, :open_interest, :open_interest_change, :open_interest_change_pct,
            :oi_price_regime, :orderbook_imbalance, :orderbook_imbalance_avg_30_120s, :orderbook_spread,
            :orderbook_spread_change, :orderbook_microprice, :orderbook_has_large_wall, :created_at
        )
        """,
        {
            "market_snapshot_id": market_snapshot_id,
            "symbol": features.symbol,
            "timestamp": _dt_to_epoch(features.timestamp),
            "data_quality": features.data_quality,
            "timeframe_alignment": features.timeframe_alignment,
            "distance_to_support_atr": features.distance_to_support_atr,
            "distance_to_resistance_atr": features.distance_to_resistance_atr,
            "trend_json": _json_dumps(_map_asdict(features.trend)),
            "momentum_json": _json_dumps(_map_asdict(features.momentum)),
            "volatility_json": _json_dumps(_map_asdict(features.volatility)),
            "volume_json": _json_dumps(_map_asdict(features.volume)),
            "trade_flow_buy_volume": tf.buy_volume,
            "trade_flow_sell_volume": tf.sell_volume,
            "trade_flow_delta": tf.delta,
            "trade_flow_cvd": tf.cumulative_volume_delta,
            "funding_rate": fut.funding_rate,
            "funding_change": fut.funding_change,
            "open_interest": fut.open_interest,
            "open_interest_change": fut.open_interest_change,
            "open_interest_change_pct": fut.open_interest_change_pct,
            "oi_price_regime": fut.oi_price_regime,
            "orderbook_imbalance": ob.imbalance,
            "orderbook_imbalance_avg_30_120s": ob.imbalance_avg_30_120s,
            "orderbook_spread": ob.spread,
            "orderbook_spread_change": ob.spread_change,
            "orderbook_microprice": ob.microprice,
            "orderbook_has_large_wall": None if ob.has_large_wall is None else int(ob.has_large_wall),
            "created_at": time.time(),
        },
    )
    return cur.lastrowid


def insert_strategy_score(
    conn: sqlite3.Connection, score: ScoreResult, regime: RegimeResult, *, feature_snapshot_id: int
) -> int:
    cur = conn.execute(
        """
        INSERT INTO strategy_scores (
            feature_snapshot_id, symbol, timestamp, mode, ruleset_version,
            long_score, short_score, no_trade_score, decision, quality, contributions_json,
            regime, regime_by_timeframe_json, regime_reasons_json, regime_strategy_hint, created_at
        ) VALUES (
            :feature_snapshot_id, :symbol, :timestamp, :mode, :ruleset_version,
            :long_score, :short_score, :no_trade_score, :decision, :quality, :contributions_json,
            :regime, :regime_by_timeframe_json, :regime_reasons_json, :regime_strategy_hint, :created_at
        )
        """,
        {
            "feature_snapshot_id": feature_snapshot_id,
            "symbol": score.symbol,
            "timestamp": _dt_to_epoch(score.timestamp),
            "mode": score.mode,
            "ruleset_version": score.ruleset_version,
            "long_score": score.long_score,
            "short_score": score.short_score,
            "no_trade_score": score.no_trade_score,
            "decision": score.decision,
            "quality": score.quality,
            "contributions_json": _json_dumps([dataclasses.asdict(c) for c in score.contributions]),
            "regime": regime.regime,
            "regime_by_timeframe_json": _json_dumps(regime.regime_by_timeframe),
            "regime_reasons_json": _json_dumps(regime.reasons),
            "regime_strategy_hint": regime.strategy_hint,
            "created_at": time.time(),
        },
    )
    return cur.lastrowid


def insert_ai_prediction(
    conn: sqlite3.Connection, result: AIAnalysisResult, *, strategy_score_id: int, symbol: str, mode: str
) -> int:
    plan = result.trade_plan
    if plan is not None:
        signal = plan.signal
        entry_status = plan.entry_status
        confidence = plan.confidence
        market_regime = plan.market_regime
        entry_type = plan.entry.type
        entry_from = plan.entry.from_
        entry_to = plan.entry.to
        entry_trigger = plan.entry.trigger
        stop_loss = plan.stop_loss
        take_profits_json = _json_dumps([tp.model_dump() for tp in plan.take_profits])
        time_horizon_minutes = plan.time_horizon_minutes
        valid_for_minutes = plan.valid_for_minutes
        reasons_json = _json_dumps(plan.reasons)
        risks_json = _json_dumps(plan.risks)
        invalidation_conditions_json = _json_dumps(plan.invalidation_conditions)
        wait_conditions_json = _json_dumps(plan.wait_conditions)
        contradictions_json = _json_dumps(plan.contradictions)
        missing_context_json = _json_dumps(plan.missing_context)
        summary = plan.summary
    else:
        signal = entry_status = confidence = market_regime = None
        entry_type = entry_from = entry_to = entry_trigger = stop_loss = None
        time_horizon_minutes = valid_for_minutes = None
        take_profits_json = "[]"
        reasons_json = risks_json = "[]"
        invalidation_conditions_json = wait_conditions_json = "[]"
        contradictions_json = missing_context_json = "[]"
        summary = ""

    validation = result.validation
    if validation is not None:
        validation_status = validation.status
        validation_issues_json = _json_dumps([dataclasses.asdict(i) for i in validation.issues])
    else:
        validation_status = None
        validation_issues_json = "[]"

    cur = conn.execute(
        """
        INSERT INTO ai_predictions (
            strategy_score_id, symbol, mode, label, model, created_at_epoch, latency_seconds,
            content, error, error_code, repaired, ok, signal, entry_status, confidence,
            market_regime, entry_type, entry_from, entry_to, entry_trigger, stop_loss,
            take_profits_json, time_horizon_minutes, valid_for_minutes, reasons_json, risks_json,
            invalidation_conditions_json, wait_conditions_json, contradictions_json,
            missing_context_json, summary, validation_status, validation_issues_json, inserted_at
        ) VALUES (
            :strategy_score_id, :symbol, :mode, :label, :model, :created_at_epoch, :latency_seconds,
            :content, :error, :error_code, :repaired, :ok, :signal, :entry_status, :confidence,
            :market_regime, :entry_type, :entry_from, :entry_to, :entry_trigger, :stop_loss,
            :take_profits_json, :time_horizon_minutes, :valid_for_minutes, :reasons_json, :risks_json,
            :invalidation_conditions_json, :wait_conditions_json, :contradictions_json,
            :missing_context_json, :summary, :validation_status, :validation_issues_json, :inserted_at
        )
        """,
        {
            "strategy_score_id": strategy_score_id,
            "symbol": symbol,
            "mode": mode,
            "label": result.label,
            "model": result.model,
            "created_at_epoch": result.created_at,
            "latency_seconds": result.latency_seconds,
            "content": result.content,
            "error": result.error,
            "error_code": result.error_code,
            "repaired": int(result.repaired),
            "ok": int(result.ok),
            "signal": signal,
            "entry_status": entry_status,
            "confidence": confidence,
            "market_regime": market_regime,
            "entry_type": entry_type,
            "entry_from": entry_from,
            "entry_to": entry_to,
            "entry_trigger": entry_trigger,
            "stop_loss": stop_loss,
            "take_profits_json": take_profits_json,
            "time_horizon_minutes": time_horizon_minutes,
            "valid_for_minutes": valid_for_minutes,
            "reasons_json": reasons_json,
            "risks_json": risks_json,
            "invalidation_conditions_json": invalidation_conditions_json,
            "wait_conditions_json": wait_conditions_json,
            "contradictions_json": contradictions_json,
            "missing_context_json": missing_context_json,
            "summary": summary,
            "validation_status": validation_status,
            "validation_issues_json": validation_issues_json,
            "inserted_at": time.time(),
        },
    )
    return cur.lastrowid


def insert_trade_plan(
    conn: sqlite3.Connection,
    consensus: ConsensusResult,
    *,
    strategy_score_id: int,
    symbol: str,
    timestamp: float,
    calculation: PositionCalculation | None = None,
    bingx_fields: BingXManualFields | None = None,
) -> int:
    plan = consensus.plan
    if plan is not None:
        source_label = plan.source_label
        entry_status = plan.entry_status
        entry_type = plan.entry_type
        entry_from = plan.entry_from
        entry_to = plan.entry_to
        stop_loss = plan.stop_loss
        take_profits_json = _json_dumps([list(tp) for tp in plan.take_profits])
        risk_reward_tp1 = plan.risk_reward_tp1
        time_horizon_minutes = plan.time_horizon_minutes
        valid_for_minutes = plan.valid_for_minutes
        formed_at = plan.formed_at
    else:
        source_label = entry_status = entry_type = None
        entry_from = entry_to = stop_loss = risk_reward_tp1 = None
        time_horizon_minutes = valid_for_minutes = formed_at = None
        take_profits_json = "[]"

    if calculation is not None:
        position_status = calculation.status.value
        limited_by = calculation.limited_by
        entry_price_calc = _dec_to_text(calculation.entry_price)
        entry_zone_low = _dec_to_text(calculation.entry_zone[0])
        entry_zone_high = _dec_to_text(calculation.entry_zone[1])
        entry_price_mode = calculation.entry_price_mode.value
        stop_loss_calc = _dec_to_text(calculation.stop_loss)
        stop_distance = _dec_to_text(calculation.stop_distance)
        account_balance_usdt = _dec_to_text(calculation.account_balance_usdt)
        risk_percent = _dec_to_text(calculation.risk_percent)
        risk_budget_usdt = _dec_to_text(calculation.risk_budget_usdt)
        leverage = calculation.leverage
        margin_mode = calculation.margin_mode.value
        position_size_coin = _dec_to_text(calculation.position_size_coin)
        position_size_coin_rounded = _dec_to_text(calculation.position_size_coin_rounded)
        position_notional_usdt = _dec_to_text(calculation.position_notional_usdt)
        required_margin_usdt = _dec_to_text(calculation.required_margin_usdt)
        max_allowed_margin_usdt = _dec_to_text(calculation.max_allowed_margin_usdt)
        entry_fee_usdt = _dec_to_text(calculation.entry_fee_usdt)
        stop_exit_fee_usdt = _dec_to_text(calculation.stop_exit_fee_usdt)
        slippage_estimate_usdt = _dec_to_text(calculation.slippage_estimate_usdt)
        total_expected_loss_usdt = _dec_to_text(calculation.total_expected_loss_usdt)
        actual_risk_percent = _dec_to_text(calculation.actual_risk_percent)
        take_profit_results_json = _json_dumps([dataclasses.asdict(t) for t in calculation.take_profit_results])
        primary_target_label = calculation.primary_target_label
        blended_json = (
            _json_dumps(dataclasses.asdict(calculation.blended)) if calculation.blended is not None else None
        )
        free_balance_after_open_usdt = _dec_to_text(calculation.free_balance_after_open_usdt)
        position_warnings_json = _json_dumps(calculation.warnings)
        position_issues_json = _json_dumps(calculation.issues)
    else:
        position_status = limited_by = None
        entry_price_calc = entry_zone_low = entry_zone_high = entry_price_mode = None
        stop_loss_calc = stop_distance = account_balance_usdt = risk_percent = risk_budget_usdt = None
        leverage = margin_mode = None
        position_size_coin = position_size_coin_rounded = position_notional_usdt = None
        required_margin_usdt = max_allowed_margin_usdt = None
        entry_fee_usdt = stop_exit_fee_usdt = slippage_estimate_usdt = total_expected_loss_usdt = None
        actual_risk_percent = primary_target_label = None
        take_profit_results_json = "[]"
        blended_json = None
        free_balance_after_open_usdt = None
        position_warnings_json = position_issues_json = "[]"

    bingx_fields_json = _json_dumps(dataclasses.asdict(bingx_fields)) if bingx_fields is not None else None

    cur = conn.execute(
        """
        INSERT INTO trade_plans (
            strategy_score_id, symbol, mode, timestamp, overall_signal, consensus_state,
            agreement_fraction, agreeing_count, vote_count, total_models, avg_confidence,
            trade_permission, trade_permission_reason, reasons_json, risks_json,
            wait_or_invalidation_json, source_label, entry_status, entry_type, entry_from,
            entry_to, stop_loss, take_profits_json, risk_reward_tp1, time_horizon_minutes,
            valid_for_minutes, formed_at, position_status, limited_by, entry_price_calc,
            entry_zone_low, entry_zone_high, entry_price_mode, stop_loss_calc, stop_distance,
            account_balance_usdt, risk_percent, risk_budget_usdt, leverage, margin_mode,
            position_size_coin, position_size_coin_rounded, position_notional_usdt,
            required_margin_usdt, max_allowed_margin_usdt, entry_fee_usdt, stop_exit_fee_usdt,
            slippage_estimate_usdt, total_expected_loss_usdt, actual_risk_percent,
            take_profit_results_json, primary_target_label, blended_json,
            free_balance_after_open_usdt, position_warnings_json, position_issues_json,
            bingx_fields_json, created_at
        ) VALUES (
            :strategy_score_id, :symbol, :mode, :timestamp, :overall_signal, :consensus_state,
            :agreement_fraction, :agreeing_count, :vote_count, :total_models, :avg_confidence,
            :trade_permission, :trade_permission_reason, :reasons_json, :risks_json,
            :wait_or_invalidation_json, :source_label, :entry_status, :entry_type, :entry_from,
            :entry_to, :stop_loss, :take_profits_json, :risk_reward_tp1, :time_horizon_minutes,
            :valid_for_minutes, :formed_at, :position_status, :limited_by, :entry_price_calc,
            :entry_zone_low, :entry_zone_high, :entry_price_mode, :stop_loss_calc, :stop_distance,
            :account_balance_usdt, :risk_percent, :risk_budget_usdt, :leverage, :margin_mode,
            :position_size_coin, :position_size_coin_rounded, :position_notional_usdt,
            :required_margin_usdt, :max_allowed_margin_usdt, :entry_fee_usdt, :stop_exit_fee_usdt,
            :slippage_estimate_usdt, :total_expected_loss_usdt, :actual_risk_percent,
            :take_profit_results_json, :primary_target_label, :blended_json,
            :free_balance_after_open_usdt, :position_warnings_json, :position_issues_json,
            :bingx_fields_json, :created_at
        )
        """,
        {
            "strategy_score_id": strategy_score_id,
            "symbol": symbol,
            "mode": consensus.mode,
            "timestamp": timestamp,
            "overall_signal": consensus.overall_signal,
            "consensus_state": consensus.state,
            "agreement_fraction": consensus.agreement_fraction,
            "agreeing_count": consensus.agreeing_count,
            "vote_count": consensus.vote_count,
            "total_models": consensus.total_models,
            "avg_confidence": consensus.avg_confidence,
            "trade_permission": consensus.trade_permission,
            "trade_permission_reason": consensus.trade_permission_reason,
            "reasons_json": _json_dumps(consensus.reasons),
            "risks_json": _json_dumps(consensus.risks),
            "wait_or_invalidation_json": _json_dumps(consensus.wait_or_invalidation),
            "source_label": source_label,
            "entry_status": entry_status,
            "entry_type": entry_type,
            "entry_from": entry_from,
            "entry_to": entry_to,
            "stop_loss": stop_loss,
            "take_profits_json": take_profits_json,
            "risk_reward_tp1": risk_reward_tp1,
            "time_horizon_minutes": time_horizon_minutes,
            "valid_for_minutes": valid_for_minutes,
            "formed_at": formed_at,
            "position_status": position_status,
            "limited_by": limited_by,
            "entry_price_calc": entry_price_calc,
            "entry_zone_low": entry_zone_low,
            "entry_zone_high": entry_zone_high,
            "entry_price_mode": entry_price_mode,
            "stop_loss_calc": stop_loss_calc,
            "stop_distance": stop_distance,
            "account_balance_usdt": account_balance_usdt,
            "risk_percent": risk_percent,
            "risk_budget_usdt": risk_budget_usdt,
            "leverage": leverage,
            "margin_mode": margin_mode,
            "position_size_coin": position_size_coin,
            "position_size_coin_rounded": position_size_coin_rounded,
            "position_notional_usdt": position_notional_usdt,
            "required_margin_usdt": required_margin_usdt,
            "max_allowed_margin_usdt": max_allowed_margin_usdt,
            "entry_fee_usdt": entry_fee_usdt,
            "stop_exit_fee_usdt": stop_exit_fee_usdt,
            "slippage_estimate_usdt": slippage_estimate_usdt,
            "total_expected_loss_usdt": total_expected_loss_usdt,
            "actual_risk_percent": actual_risk_percent,
            "take_profit_results_json": take_profit_results_json,
            "primary_target_label": primary_target_label,
            "blended_json": blended_json,
            "free_balance_after_open_usdt": free_balance_after_open_usdt,
            "position_warnings_json": position_warnings_json,
            "position_issues_json": position_issues_json,
            "bingx_fields_json": bingx_fields_json,
            "created_at": time.time(),
        },
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Trade outcome lifecycle: pending -> price checkpoints -> finalize
# ---------------------------------------------------------------------------


def insert_trade_outcome_pending(
    conn: sqlite3.Connection,
    *,
    trade_plan_id: int,
    symbol: str,
    mode: str,
    entry_price: float | None = None,
    now: float | None = None,
) -> int:
    now = now if now is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO trade_outcomes (trade_plan_id, symbol, mode, status, entry_price, created_at)
        VALUES (:trade_plan_id, :symbol, :mode, 'PENDING', :entry_price, :created_at)
        """,
        {"trade_plan_id": trade_plan_id, "symbol": symbol, "mode": mode, "entry_price": entry_price, "created_at": now},
    )
    return cur.lastrowid


_CHECKPOINT_COLUMNS = {1: "price_after_1m", 5: "price_after_5m", 15: "price_after_15m", 30: "price_after_30m"}


def update_trade_outcome_price_checkpoint(
    conn: sqlite3.Connection, trade_outcome_id: int, *, minute: Literal[1, 5, 15, 30], price: float
) -> None:
    """Idempotent: only the FIRST call for a given checkpoint sticks (via
    COALESCE) — safe to call on every tick past a checkpoint without
    clobbering the value recorded when it was first reached.
    """
    column = _CHECKPOINT_COLUMNS[minute]
    conn.execute(f"UPDATE trade_outcomes SET {column} = COALESCE({column}, ?) WHERE id = ?", (price, trade_outcome_id))


def update_trade_outcome_running_extremes(
    conn: sqlite3.Connection, trade_outcome_id: int, *, favorable_r: float, adverse_r: float
) -> None:
    """Tick-sampled running max of MFE/MAE (in R) — NOT OHLC-exact like
    outcome_simulator's candle walk, bounded by how often process_tick is
    called. The double-COALESCE trick seeds the MAX with the same value
    being offered so a NULL starting point never suppresses a negative
    first reading.
    """
    conn.execute(
        """
        UPDATE trade_outcomes SET
            mfe_r = MAX(COALESCE(mfe_r, :favorable_r), :favorable_r),
            mae_r = MAX(COALESCE(mae_r, :adverse_r), :adverse_r)
        WHERE id = :id
        """,
        {"favorable_r": favorable_r, "adverse_r": adverse_r, "id": trade_outcome_id},
    )


def finalize_trade_outcome(
    conn: sqlite3.Connection,
    trade_outcome_id: int,
    outcome: SimulatedOutcome,
    *,
    commission_usdt: float | None = None,
    slippage_usdt: float | None = None,
    realized_pnl_usdt: float | None = None,
    now: float | None = None,
) -> None:
    now = now if now is not None else time.time()
    conn.execute(
        """
        UPDATE trade_outcomes SET
            status = 'EVALUATED', exit_reason = :exit_reason, exit_price = :exit_price,
            exit_time = :exit_time, duration_seconds = :duration_seconds, r_multiple = :r_multiple,
            mfe_r = :mfe_r, mae_r = :mae_r, commission_usdt = :commission_usdt,
            slippage_usdt = :slippage_usdt, realized_pnl_usdt = :realized_pnl_usdt, evaluated_at = :evaluated_at
        WHERE id = :id
        """,
        {
            "exit_reason": outcome.exit_reason,
            "exit_price": outcome.exit_price,
            "exit_time": outcome.exit_time,
            "duration_seconds": outcome.duration_seconds,
            "r_multiple": outcome.r_multiple,
            "mfe_r": outcome.mfe_r,
            "mae_r": outcome.mae_r,
            "commission_usdt": commission_usdt,
            "slippage_usdt": slippage_usdt,
            "realized_pnl_usdt": realized_pnl_usdt,
            "evaluated_at": now,
            "id": trade_outcome_id,
        },
    )


# ---------------------------------------------------------------------------
# Schema-only CRUD: paper_orders / real_orders / positions / fills
# ---------------------------------------------------------------------------


def _insert_order(
    conn: sqlite3.Connection,
    table: Literal["paper_orders", "real_orders"],
    *,
    symbol: str,
    side: str,
    order_type: str,
    quantity: Decimal | str,
    price: Decimal | str | None = None,
    trigger_price: Decimal | str | None = None,
    stop_loss: Decimal | str | None = None,
    take_profit: Decimal | str | None = None,
    take_profits: list[tuple[str, float, float]] | None = None,
    leverage: int | None = None,
    margin_mode: str | None = None,
    trade_plan_id: int | None = None,
    status: str = "PENDING",
    notes: str | None = None,
    now: float | None = None,
    exchange_order_id: str | None = None,
    entry_from: float | None = None,
    entry_to: float | None = None,
    label: str | None = None,
) -> int:
    now = now if now is not None else time.time()
    columns = [
        "trade_plan_id",
        "symbol",
        "side",
        "order_type",
        "status",
        "price",
        "trigger_price",
        "quantity",
        "leverage",
        "margin_mode",
        "stop_loss",
        "take_profit",
        "take_profits_json",
        "created_at",
        "updated_at",
        "notes",
    ]
    params = {
        "trade_plan_id": trade_plan_id,
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "status": status,
        "price": _norm_dec_text(price),
        "trigger_price": _norm_dec_text(trigger_price),
        "quantity": _norm_dec_text(quantity),
        "leverage": leverage,
        "margin_mode": margin_mode,
        "stop_loss": _norm_dec_text(stop_loss),
        "take_profit": _norm_dec_text(take_profit),
        "take_profits_json": _json_dumps(take_profits) if take_profits is not None else "[]",
        "created_at": now,
        "updated_at": now,
        "notes": notes,
    }
    if table == "real_orders":
        columns.extend(["exchange_order_id", "label"])
        params["exchange_order_id"] = exchange_order_id
        params["label"] = label
    if table == "paper_orders":
        columns.extend(["entry_from", "entry_to"])
        params["entry_from"] = entry_from
        params["entry_to"] = entry_to
    column_list = ", ".join(columns)
    placeholders = ", ".join(f":{c}" for c in columns)
    cur = conn.execute(f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})", params)
    return cur.lastrowid


def insert_paper_order(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    order_type: str,
    quantity: Decimal | str,
    price: Decimal | str | None = None,
    trigger_price: Decimal | str | None = None,
    stop_loss: Decimal | str | None = None,
    take_profit: Decimal | str | None = None,
    take_profits: list[tuple[str, float, float]] | None = None,
    leverage: int | None = None,
    margin_mode: str | None = None,
    trade_plan_id: int | None = None,
    status: str = "PENDING",
    notes: str | None = None,
    now: float | None = None,
    entry_from: float | None = None,
    entry_to: float | None = None,
) -> int:
    return _insert_order(
        conn,
        "paper_orders",
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        trigger_price=trigger_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        take_profits=take_profits,
        leverage=leverage,
        margin_mode=margin_mode,
        trade_plan_id=trade_plan_id,
        status=status,
        notes=notes,
        now=now,
        entry_from=entry_from,
        entry_to=entry_to,
    )


def insert_real_order(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    order_type: str,
    quantity: Decimal | str,
    price: Decimal | str | None = None,
    trigger_price: Decimal | str | None = None,
    stop_loss: Decimal | str | None = None,
    take_profit: Decimal | str | None = None,
    take_profits: list[tuple[str, float, float]] | None = None,
    leverage: int | None = None,
    margin_mode: str | None = None,
    trade_plan_id: int | None = None,
    status: str = "PENDING",
    notes: str | None = None,
    now: float | None = None,
    exchange_order_id: str | None = None,
    label: str | None = None,
) -> int:
    return _insert_order(
        conn,
        "real_orders",
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        trigger_price=trigger_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        take_profits=take_profits,
        leverage=leverage,
        margin_mode=margin_mode,
        trade_plan_id=trade_plan_id,
        status=status,
        notes=notes,
        now=now,
        exchange_order_id=exchange_order_id,
        label=label,
    )


def update_order_status(
    conn: sqlite3.Connection,
    table: Literal["paper_orders", "real_orders"],
    order_id: int,
    status: str,
    *,
    filled_quantity: Decimal | str | None = None,
    now: float | None = None,
) -> None:
    now = now if now is not None else time.time()
    if filled_quantity is not None:
        conn.execute(
            f"UPDATE {table} SET status = ?, filled_quantity = ?, updated_at = ? WHERE id = ?",
            (status, _norm_dec_text(filled_quantity), now, order_id),
        )
    else:
        conn.execute(f"UPDATE {table} SET status = ?, updated_at = ? WHERE id = ?", (status, now, order_id))


def insert_position(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    source: Literal["PAPER", "REAL"],
    entry_price: Decimal | str,
    quantity: Decimal | str,
    leverage: int | None = None,
    margin_mode: str | None = None,
    stop_loss: Decimal | str | None = None,
    take_profit: Decimal | str | None = None,
    trade_plan_id: int | None = None,
    paper_order_id: int | None = None,
    real_order_id: int | None = None,
    status: str = "OPEN",
    now: float | None = None,
) -> int:
    now = now if now is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO positions (
            trade_plan_id, paper_order_id, real_order_id, source, symbol, side, status,
            entry_price, quantity, leverage, margin_mode, stop_loss, take_profit, opened_at
        ) VALUES (
            :trade_plan_id, :paper_order_id, :real_order_id, :source, :symbol, :side, :status,
            :entry_price, :quantity, :leverage, :margin_mode, :stop_loss, :take_profit, :opened_at
        )
        """,
        {
            "trade_plan_id": trade_plan_id,
            "paper_order_id": paper_order_id,
            "real_order_id": real_order_id,
            "source": source,
            "symbol": symbol,
            "side": side,
            "status": status,
            "entry_price": _norm_dec_text(entry_price),
            "quantity": _norm_dec_text(quantity),
            "leverage": leverage,
            "margin_mode": margin_mode,
            "stop_loss": _norm_dec_text(stop_loss),
            "take_profit": _norm_dec_text(take_profit),
            "opened_at": now,
        },
    )
    return cur.lastrowid


def close_position(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    exit_price: Decimal | str,
    realized_pnl_usdt: Decimal | str | None = None,
    fees_paid_usdt: Decimal | str | None = None,
    now: float | None = None,
) -> None:
    now = now if now is not None else time.time()
    conn.execute(
        """
        UPDATE positions SET status = 'CLOSED', exit_price = ?, realized_pnl_usdt = ?,
            fees_paid_usdt = ?, closed_at = ? WHERE id = ?
        """,
        (_norm_dec_text(exit_price), _norm_dec_text(realized_pnl_usdt), _norm_dec_text(fees_paid_usdt), now, position_id),
    )


def update_position_stop_loss(
    conn: sqlite3.Connection, position_id: int, new_stop_loss: Decimal | str, *, now: float | None = None
) -> None:
    """Mutates positions.stop_loss only — trailing stop / breakeven. `now` is
    accepted for signature symmetry with the rest of this module but unused
    today (positions has no updated_at column, only opened_at/closed_at).
    """
    conn.execute("UPDATE positions SET stop_loss = ? WHERE id = ?", (_norm_dec_text(new_stop_loss), position_id))


def update_position_add_fill(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    additional_quantity: Decimal | str,
    fill_price: Decimal | str,
    now: float | None = None,
) -> None:
    """Grows an OPEN position by one more ENTRY fill — recomputes entry_price
    as the quantity-weighted average of the old position and the new fill,
    and increments quantity. Used only while an order is still
    PARTIALLY_FILLED and accumulating; never called after a position starts
    closing. Read-then-write (fetch current entry_price/quantity, compute in
    Python with Decimal, write back) since SQLite has no Decimal type to do
    the weighted-average arithmetic itself.
    """
    row = conn.execute("SELECT entry_price, quantity FROM positions WHERE id = ?", (position_id,)).fetchone()
    old_price, old_qty = _text_to_dec(row["entry_price"]), _text_to_dec(row["quantity"])
    add_qty = Decimal(_norm_dec_text(additional_quantity))
    add_price = Decimal(_norm_dec_text(fill_price))
    new_qty = old_qty + add_qty
    new_price = (old_price * old_qty + add_price * add_qty) / new_qty
    conn.execute(
        "UPDATE positions SET entry_price = ?, quantity = ? WHERE id = ?",
        (_dec_to_text(new_price), _dec_to_text(new_qty), position_id),
    )


def insert_fill(
    conn: sqlite3.Connection,
    *,
    position_id: int,
    symbol: str,
    side: str,
    fill_type: str,
    price: Decimal | str,
    quantity: Decimal | str,
    fee_usdt: Decimal | str | None = None,
    realized_pnl_usdt: Decimal | str | None = None,
    paper_order_id: int | None = None,
    real_order_id: int | None = None,
    label: str | None = None,
    now: float | None = None,
) -> int:
    """`label` is the take-profit label ('TP1'/'TP2'/'TP3') when
    `fill_type='TAKE_PROFIT'` — the only way to know which specific target
    closed, since `fill_type` itself is generic. NULL for every other
    fill_type.
    """
    now = now if now is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO fills (
            position_id, paper_order_id, real_order_id, symbol, side, fill_type, label, price, quantity,
            fee_usdt, realized_pnl_usdt, filled_at
        ) VALUES (
            :position_id, :paper_order_id, :real_order_id, :symbol, :side, :fill_type, :label, :price, :quantity,
            :fee_usdt, :realized_pnl_usdt, :filled_at
        )
        """,
        {
            "position_id": position_id,
            "paper_order_id": paper_order_id,
            "real_order_id": real_order_id,
            "symbol": symbol,
            "side": side,
            "fill_type": fill_type,
            "label": label,
            "price": _norm_dec_text(price),
            "quantity": _norm_dec_text(quantity),
            "fee_usdt": _norm_dec_text(fee_usdt),
            "realized_pnl_usdt": _norm_dec_text(realized_pnl_usdt),
            "filled_at": now,
        },
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Statistics + events
# ---------------------------------------------------------------------------


def insert_model_statistics(
    conn: sqlite3.Connection,
    *,
    model_label: str,
    mode: str,
    window_start: float,
    window_end: float,
    total_predictions: int,
    evaluated_count: int,
    wins: int,
    win_rate: float | None,
    expectancy_r: float | None,
    median_r: float | None,
    profit_factor: float | None,
    profit_factor_undefined: bool,
    max_drawdown_r: float | None,
    avg_mfe_r: float | None,
    avg_mae_r: float | None,
    avg_duration_seconds: float | None,
    exit_reason_counts: dict[str, int],
    low_sample: bool,
    computed_at: float | None = None,
) -> int:
    computed_at = computed_at if computed_at is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO model_statistics (
            computed_at, model_label, mode, window_start, window_end, total_predictions,
            evaluated_count, wins, win_rate, expectancy_r, median_r, profit_factor,
            profit_factor_undefined, max_drawdown_r, avg_mfe_r, avg_mae_r, avg_duration_seconds,
            exit_reason_counts_json, low_sample
        ) VALUES (
            :computed_at, :model_label, :mode, :window_start, :window_end, :total_predictions,
            :evaluated_count, :wins, :win_rate, :expectancy_r, :median_r, :profit_factor,
            :profit_factor_undefined, :max_drawdown_r, :avg_mfe_r, :avg_mae_r, :avg_duration_seconds,
            :exit_reason_counts_json, :low_sample
        )
        """,
        {
            "computed_at": computed_at,
            "model_label": model_label,
            "mode": mode,
            "window_start": window_start,
            "window_end": window_end,
            "total_predictions": total_predictions,
            "evaluated_count": evaluated_count,
            "wins": wins,
            "win_rate": win_rate,
            "expectancy_r": expectancy_r,
            "median_r": median_r,
            "profit_factor": profit_factor,
            "profit_factor_undefined": int(profit_factor_undefined),
            "max_drawdown_r": max_drawdown_r,
            "avg_mfe_r": avg_mfe_r,
            "avg_mae_r": avg_mae_r,
            "avg_duration_seconds": avg_duration_seconds,
            "exit_reason_counts_json": _json_dumps(exit_reason_counts),
            "low_sample": int(low_sample),
        },
    )
    return cur.lastrowid


def insert_strategy_statistics(
    conn: sqlite3.Connection,
    *,
    ruleset_version: str,
    mode: str,
    decision: str | None,
    window_start: float,
    window_end: float,
    total_signals: int,
    evaluated_count: int,
    wins: int,
    win_rate: float | None,
    expectancy_r: float | None,
    median_r: float | None,
    profit_factor: float | None,
    low_sample: bool,
    regime: str | None = None,
    computed_at: float | None = None,
) -> int:
    computed_at = computed_at if computed_at is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO strategy_statistics (
            computed_at, ruleset_version, mode, decision, regime, window_start, window_end, total_signals,
            evaluated_count, wins, win_rate, expectancy_r, median_r, profit_factor, low_sample
        ) VALUES (
            :computed_at, :ruleset_version, :mode, :decision, :regime, :window_start, :window_end, :total_signals,
            :evaluated_count, :wins, :win_rate, :expectancy_r, :median_r, :profit_factor, :low_sample
        )
        """,
        {
            "computed_at": computed_at,
            "ruleset_version": ruleset_version,
            "mode": mode,
            "decision": decision,
            "regime": regime,
            "window_start": window_start,
            "window_end": window_end,
            "total_signals": total_signals,
            "evaluated_count": evaluated_count,
            "wins": wins,
            "win_rate": win_rate,
            "expectancy_r": expectancy_r,
            "median_r": median_r,
            "profit_factor": profit_factor,
            "low_sample": int(low_sample),
        },
    )
    return cur.lastrowid


def log_event(
    conn: sqlite3.Connection,
    *,
    level: str,
    event_code: str,
    source_module: str,
    message: str,
    symbol: str | None = None,
    details: dict | None = None,
    market_snapshot_id: int | None = None,
    feature_snapshot_id: int | None = None,
    strategy_score_id: int | None = None,
    trade_plan_id: int | None = None,
    now: float | None = None,
) -> int:
    now = now if now is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO system_events (
            timestamp, level, event_code, source_module, symbol, message, details_json,
            market_snapshot_id, feature_snapshot_id, strategy_score_id, trade_plan_id
        ) VALUES (
            :timestamp, :level, :event_code, :source_module, :symbol, :message, :details_json,
            :market_snapshot_id, :feature_snapshot_id, :strategy_score_id, :trade_plan_id
        )
        """,
        {
            "timestamp": now,
            "level": level,
            "event_code": event_code,
            "source_module": source_module,
            "symbol": symbol,
            "message": message,
            "details_json": _json_dumps(details or {}),
            "market_snapshot_id": market_snapshot_id,
            "feature_snapshot_id": feature_snapshot_id,
            "strategy_score_id": strategy_score_id,
            "trade_plan_id": trade_plan_id,
        },
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

_MARKET_SNAPSHOT_JSON_FIELDS = (
    "quality_issues_json",
    "timeframes_json",
    "funding_history_json",
    "open_interest_history_json",
    "orderbook_json",
    "orderbook_history_json",
    "recent_trades_json",
    "instrument_rules_json",
)


def get_market_snapshot(conn: sqlite3.Connection, snapshot_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM market_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
    return _row_to_dict(row, json_fields=_MARKET_SNAPSHOT_JSON_FIELDS) if row is not None else None


def get_recent_market_snapshots(conn: sqlite3.Connection, symbol: str, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM market_snapshots WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?", (symbol, limit)
    ).fetchall()
    return [_row_to_dict(r, json_fields=_MARKET_SNAPSHOT_JSON_FIELDS) for r in rows]


_FEATURE_SNAPSHOT_JSON_FIELDS = ("trend_json", "momentum_json", "volatility_json", "volume_json")
_FEATURE_SNAPSHOT_BOOL_FIELDS = ("orderbook_has_large_wall",)


def get_feature_snapshot(conn: sqlite3.Connection, feature_snapshot_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM feature_snapshots WHERE id = ?", (feature_snapshot_id,)).fetchone()
    return (
        _row_to_dict(row, json_fields=_FEATURE_SNAPSHOT_JSON_FIELDS, bool_fields=_FEATURE_SNAPSHOT_BOOL_FIELDS)
        if row is not None
        else None
    )


_STRATEGY_SCORE_JSON_FIELDS = ("contributions_json", "regime_by_timeframe_json", "regime_reasons_json")


def get_strategy_score(conn: sqlite3.Connection, score_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM strategy_scores WHERE id = ?", (score_id,)).fetchone()
    return _row_to_dict(row, json_fields=_STRATEGY_SCORE_JSON_FIELDS) if row is not None else None


def get_recent_strategy_scores(
    conn: sqlite3.Connection, symbol: str, mode: str | None = None, limit: int = 20
) -> list[dict]:
    if mode is not None:
        rows = conn.execute(
            "SELECT * FROM strategy_scores WHERE symbol = ? AND mode = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, mode, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM strategy_scores WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?", (symbol, limit)
        ).fetchall()
    return [_row_to_dict(r, json_fields=_STRATEGY_SCORE_JSON_FIELDS) for r in rows]


_AI_PREDICTION_JSON_FIELDS = (
    "take_profits_json",
    "reasons_json",
    "risks_json",
    "invalidation_conditions_json",
    "wait_conditions_json",
    "contradictions_json",
    "missing_context_json",
    "validation_issues_json",
)
_AI_PREDICTION_BOOL_FIELDS = ("repaired", "ok")


def get_ai_predictions_for_score(conn: sqlite3.Connection, strategy_score_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM ai_predictions WHERE strategy_score_id = ? ORDER BY id", (strategy_score_id,)
    ).fetchall()
    return [
        _row_to_dict(r, json_fields=_AI_PREDICTION_JSON_FIELDS, bool_fields=_AI_PREDICTION_BOOL_FIELDS) for r in rows
    ]


_TRADE_PLAN_JSON_FIELDS = (
    "reasons_json",
    "risks_json",
    "wait_or_invalidation_json",
    "take_profits_json",
    "take_profit_results_json",
    "blended_json",
    "position_warnings_json",
    "position_issues_json",
    "bingx_fields_json",
)
_TRADE_PLAN_DECIMAL_FIELDS = (
    "entry_price_calc",
    "entry_zone_low",
    "entry_zone_high",
    "stop_loss_calc",
    "stop_distance",
    "account_balance_usdt",
    "risk_percent",
    "risk_budget_usdt",
    "position_size_coin",
    "position_size_coin_rounded",
    "position_notional_usdt",
    "required_margin_usdt",
    "max_allowed_margin_usdt",
    "entry_fee_usdt",
    "stop_exit_fee_usdt",
    "slippage_estimate_usdt",
    "total_expected_loss_usdt",
    "actual_risk_percent",
    "free_balance_after_open_usdt",
)


def get_trade_plan(conn: sqlite3.Connection, trade_plan_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM trade_plans WHERE id = ?", (trade_plan_id,)).fetchone()
    return (
        _row_to_dict(row, json_fields=_TRADE_PLAN_JSON_FIELDS, decimal_fields=_TRADE_PLAN_DECIMAL_FIELDS)
        if row is not None
        else None
    )


def get_trade_plan_by_score(conn: sqlite3.Connection, strategy_score_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM trade_plans WHERE strategy_score_id = ?", (strategy_score_id,)).fetchone()
    return (
        _row_to_dict(row, json_fields=_TRADE_PLAN_JSON_FIELDS, decimal_fields=_TRADE_PLAN_DECIMAL_FIELDS)
        if row is not None
        else None
    )


def get_recent_trade_plans(
    conn: sqlite3.Connection, symbol: str, mode: str | None = None, limit: int = 20
) -> list[dict]:
    if mode is not None:
        rows = conn.execute(
            "SELECT * FROM trade_plans WHERE symbol = ? AND mode = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, mode, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trade_plans WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?", (symbol, limit)
        ).fetchall()
    return [_row_to_dict(r, json_fields=_TRADE_PLAN_JSON_FIELDS, decimal_fields=_TRADE_PLAN_DECIMAL_FIELDS) for r in rows]


def get_trade_outcome(conn: sqlite3.Connection, trade_plan_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM trade_outcomes WHERE trade_plan_id = ?", (trade_plan_id,)).fetchone()
    return _row_to_dict(row) if row is not None else None


def get_full_signal(conn: sqlite3.Connection, strategy_score_id: int) -> dict:
    """Reconstructs one signal end to end: market_snapshot <- feature_snapshot
    <- strategy_score -> ai_predictions[] -> trade_plan -> trade_outcome, all
    JSON/Decimal columns already deserialized/reconstructed.
    """
    score = get_strategy_score(conn, strategy_score_id)
    feature_snapshot = get_feature_snapshot(conn, score["feature_snapshot_id"]) if score else None
    market_snapshot = (
        get_market_snapshot(conn, feature_snapshot["market_snapshot_id"]) if feature_snapshot else None
    )
    ai_predictions = get_ai_predictions_for_score(conn, strategy_score_id)
    trade_plan = get_trade_plan_by_score(conn, strategy_score_id)
    trade_outcome = get_trade_outcome(conn, trade_plan["id"]) if trade_plan else None
    return {
        "market_snapshot": market_snapshot,
        "feature_snapshot": feature_snapshot,
        "strategy_score": score,
        "ai_predictions": ai_predictions,
        "trade_plan": trade_plan,
        "trade_outcome": trade_outcome,
    }


def get_model_statistics(
    conn: sqlite3.Connection, model_label: str | None = None, mode: str | None = None, limit: int = 50
) -> list[dict]:
    query = "SELECT * FROM model_statistics WHERE 1=1"
    params: list = []
    if model_label is not None:
        query += " AND model_label = ?"
        params.append(model_label)
    if mode is not None:
        query += " AND mode = ?"
        params.append(mode)
    query += " ORDER BY computed_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [
        _row_to_dict(r, json_fields=("exit_reason_counts_json",), bool_fields=("profit_factor_undefined", "low_sample"))
        for r in rows
    ]


def get_strategy_statistics(
    conn: sqlite3.Connection,
    ruleset_version: str | None = None,
    mode: str | None = None,
    regime: str | None = None,
    limit: int = 50,
) -> list[dict]:
    query = "SELECT * FROM strategy_statistics WHERE 1=1"
    params: list = []
    if ruleset_version is not None:
        query += " AND ruleset_version = ?"
        params.append(ruleset_version)
    if mode is not None:
        query += " AND mode = ?"
        params.append(mode)
    if regime is not None:
        query += " AND regime = ?"
        params.append(regime)
    query += " ORDER BY computed_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r, bool_fields=("low_sample",)) for r in rows]


def get_recent_system_events(
    conn: sqlite3.Connection, level: str | None = None, event_code: str | None = None, limit: int = 100
) -> list[dict]:
    query = "SELECT * FROM system_events WHERE 1=1"
    params: list = []
    if level is not None:
        query += " AND level = ?"
        params.append(level)
    if event_code is not None:
        query += " AND event_code = ?"
        params.append(event_code)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r, json_fields=("details_json",)) for r in rows]


# ---------------------------------------------------------------------------
# Paper trading reads (paper_orders / positions / fills)
# ---------------------------------------------------------------------------

_PAPER_ORDER_JSON_FIELDS = ("take_profits_json",)
_PAPER_ORDER_DECIMAL_FIELDS = ("price", "trigger_price", "quantity", "filled_quantity", "stop_loss", "take_profit")
_POSITION_DECIMAL_FIELDS = (
    "entry_price", "quantity", "stop_loss", "take_profit", "exit_price",
    "realized_pnl_usdt", "unrealized_pnl_usdt", "fees_paid_usdt",
)
_FILL_DECIMAL_FIELDS = ("price", "quantity", "fee_usdt", "realized_pnl_usdt")


def get_pending_paper_orders(conn: sqlite3.Connection, symbol: str) -> list[dict]:
    """status IN ('PENDING','PARTIALLY_FILLED'), ordered by id (deterministic
    processing order within one process_tick call)."""
    rows = conn.execute(
        "SELECT * FROM paper_orders WHERE symbol = ? AND status IN ('PENDING','PARTIALLY_FILLED') ORDER BY id",
        (symbol,),
    ).fetchall()
    return [_row_to_dict(r, json_fields=_PAPER_ORDER_JSON_FIELDS, decimal_fields=_PAPER_ORDER_DECIMAL_FIELDS) for r in rows]


def get_paper_order_by_trade_plan_id(conn: sqlite3.Connection, trade_plan_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM paper_orders WHERE trade_plan_id = ?", (trade_plan_id,)).fetchone()
    return (
        _row_to_dict(row, json_fields=_PAPER_ORDER_JSON_FIELDS, decimal_fields=_PAPER_ORDER_DECIMAL_FIELDS)
        if row is not None
        else None
    )


def get_open_positions(conn: sqlite3.Connection, symbol: str, source: str = "PAPER") -> list[dict]:
    """status = 'OPEN', ordered by id."""
    rows = conn.execute(
        "SELECT * FROM positions WHERE symbol = ? AND source = ? AND status = 'OPEN' ORDER BY id", (symbol, source)
    ).fetchall()
    return [_row_to_dict(r, decimal_fields=_POSITION_DECIMAL_FIELDS) for r in rows]


def get_position(conn: sqlite3.Connection, position_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    return _row_to_dict(row, decimal_fields=_POSITION_DECIMAL_FIELDS) if row is not None else None


def get_position_by_paper_order_id(conn: sqlite3.Connection, paper_order_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM positions WHERE paper_order_id = ? AND status = 'OPEN'", (paper_order_id,)
    ).fetchone()
    return _row_to_dict(row, decimal_fields=_POSITION_DECIMAL_FIELDS) if row is not None else None


def get_position_fills(conn: sqlite3.Connection, position_id: int) -> list[dict]:
    """Chronological (filled_at, id — id as tiebreaker for same-tick fills,
    since filled_at can repeat when `now` is caller-supplied)."""
    rows = conn.execute(
        "SELECT * FROM fills WHERE position_id = ? ORDER BY filled_at, id", (position_id,)
    ).fetchall()
    return [_row_to_dict(r, decimal_fields=_FILL_DECIMAL_FIELDS) for r in rows]


# ---------------------------------------------------------------------------
# Real order reads (order_manager.py, Stage 11)
# ---------------------------------------------------------------------------

_REAL_ORDER_JSON_FIELDS = ("take_profits_json",)
_REAL_ORDER_DECIMAL_FIELDS = ("price", "trigger_price", "quantity", "filled_quantity", "stop_loss", "take_profit")


def get_real_order_by_trade_plan_id(conn: sqlite3.Connection, trade_plan_id: int) -> dict | None:
    """Mirrors get_paper_order_by_trade_plan_id — the idempotency check
    order_manager.place_entry_order runs first: a trade_plan_id that
    already has a real_orders row is never placed twice.
    """
    row = conn.execute("SELECT * FROM real_orders WHERE trade_plan_id = ?", (trade_plan_id,)).fetchone()
    return (
        _row_to_dict(row, json_fields=_REAL_ORDER_JSON_FIELDS, decimal_fields=_REAL_ORDER_DECIMAL_FIELDS)
        if row is not None
        else None
    )


def get_pending_real_orders(conn: sqlite3.Connection, symbol: str) -> list[dict]:
    """Mirrors get_pending_paper_orders — "is there already a live real
    order working on this symbol" (the duplicate-order safety check),
    independent of which trade_plan_id it came from.
    """
    rows = conn.execute(
        "SELECT * FROM real_orders WHERE symbol = ? AND status IN ('PENDING','OPEN','PARTIALLY_FILLED') ORDER BY id",
        (symbol,),
    ).fetchall()
    return [_row_to_dict(r, json_fields=_REAL_ORDER_JSON_FIELDS, decimal_fields=_REAL_ORDER_DECIMAL_FIELDS) for r in rows]


def set_real_order_exchange_id(
    conn: sqlite3.Connection, order_id: int, exchange_order_id: str, *, status: str = "OPEN", now: float | None = None
) -> None:
    """Called AFTER a successful bingx_private_client.place_order() —
    the real_orders row itself is written BEFORE that call (status=
    'PENDING', exchange_order_id=NULL) so a crash between the two never
    risks a duplicate submission; see order_manager.place_entry_order.
    """
    now = now if now is not None else time.time()
    conn.execute(
        "UPDATE real_orders SET exchange_order_id = ?, status = ?, updated_at = ? WHERE id = ?",
        (exchange_order_id, status, now, order_id),
    )


def get_entry_real_order_for_trade_plan(conn: sqlite3.Connection, trade_plan_id: int) -> dict | None:
    """The ENTRY row for a trade_plan_id — distinguished from its SL/TP
    siblings by stop_loss/take_profit both being NULL (see Stage 11's
    order_manager.place_entry_order, which deliberately never sets either
    on the entry row for exactly this purpose).
    """
    row = conn.execute(
        "SELECT * FROM real_orders WHERE trade_plan_id = ? AND stop_loss IS NULL AND take_profit IS NULL",
        (trade_plan_id,),
    ).fetchone()
    return (
        _row_to_dict(row, json_fields=_REAL_ORDER_JSON_FIELDS, decimal_fields=_REAL_ORDER_DECIMAL_FIELDS)
        if row is not None
        else None
    )


def get_position_by_trade_plan_id(conn: sqlite3.Connection, trade_plan_id: int) -> dict | None:
    """Any status — "has a local position already been created for this
    trade_plan" (Stage 12: order_manager only creates one immediately for
    MARKET/DRY_RUN entries; position_manager.check_entry_fill creates it
    later for LIMIT/TRIGGER once BingX confirms the fill).
    """
    row = conn.execute("SELECT * FROM positions WHERE trade_plan_id = ?", (trade_plan_id,)).fetchone()
    return _row_to_dict(row, decimal_fields=_POSITION_DECIMAL_FIELDS) if row is not None else None


def reduce_position_quantity(
    conn: sqlite3.Connection, position_id: int, new_quantity: Decimal | str, *, now: float | None = None
) -> None:
    """Shrinks an OPEN position's remaining size after a partial close (a
    take-profit level filling while the position stays open) — `now` is
    accepted for signature symmetry with the rest of this module but unused
    today (positions has no updated_at column), same as
    update_position_stop_loss.
    """
    conn.execute("UPDATE positions SET quantity = ? WHERE id = ?", (_norm_dec_text(new_quantity), position_id))
