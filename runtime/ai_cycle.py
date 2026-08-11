"""The missing bridge: turns a scheduled AI analysis into an actual paper
trade, and persists every step of the pipeline the schema was built for
(market_snapshots -> feature_snapshots -> strategy_scores -> trade_plans)
instead of only ever computing features/regime/score in memory for a
prompt and discarding them.

Before this module, nothing in the live app ever called
`paper_trading.open_virtual_order()` outside tests -- `trading_runtime.py`
would run 24/7 and manage positions that never got created. This is the
piece that actually creates them: on a schedule (see
`runtime/service.py`), fetch fresh data, ask the AI, and if the resulting
consensus is actionable (`PositionServiceResult.can_open`), open a paper
position from it. A WAIT or non-actionable decision still gets its
trade_plans row (`overall_signal='WAIT'` or a non-VALID `position_status`)
-- the point is a complete decision log, not just the trades that fired.

Reuses the exact same report/validation/AI-call recipe as server.py's
`_refresh_mode` (kept in sync manually, same as that function's own
`_refresh_single` twin) and the exact same consensus/sizing path as the
dashboard's "BingX manual card" (`position_service.calculate_active_position`)
so an auto-opened paper trade is never sized or gated any differently than
what a human would see and could manually act on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import ai_client
import config
import dashboard_state
import feature_engine
import journal_db
import market_data_engine
import market_regime
import paper_trading
import position_service
import prediction_tracker
import risk_settings_store
import strategy_engine
from market_data import fetch_snapshot
from report_builder import AIContext, build_report
from trade_validator import build_validation_context


@dataclass(frozen=True)
class AICycleResult:
    ran: bool
    reason: str | None = None
    trade_plan_id: int | None = None
    paper_order_status: str | None = None
    paper_order_reason: str | None = None


def run_ai_cycle(conn, symbol: str, mode: str, *, now: float | None = None) -> AICycleResult:
    now = now if now is not None else time.time()
    try:
        engine_snapshot = market_data_engine.collect_snapshot(symbol, now=now)
        features = feature_engine.compute_features(engine_snapshot)
        regime = market_regime.classify_regime(features)
        score = strategy_engine.score_strategy(features, regime, mode=mode)
    except Exception as exc:
        return AICycleResult(ran=False, reason=f"feature/regime/score pipeline failed: {exc}")

    risk_settings = risk_settings_store.load()
    ai_context = AIContext(regime=regime, score=score, features=features, risk_settings=risk_settings)

    try:
        snapshot = fetch_snapshot(symbol)
    except Exception as exc:
        return AICycleResult(ran=False, reason=f"fetch_snapshot failed: {exc}")

    bingx_symbol = config.to_bingx_symbol(snapshot["symbol"])
    report_text = build_report({**snapshot, "mode_key": mode}, ai_context)
    validation_ctx = build_validation_context(snapshot, mode)

    try:
        results = ai_client.analyze_with_all(report_text, config.AI_MODELS, config.AI_REQUEST_TIMEOUT, validation_ctx, mode)
    except ai_client.AIConfigError as exc:
        return AICycleResult(ran=False, reason=str(exc))

    prediction_tracker.record_results(mode, results, snapshot["current_price"], bingx_symbol)
    dashboard_state.save_mode_results(mode, results, snapshot)

    snap_id = journal_db.insert_market_snapshot(conn, engine_snapshot)
    fs_id = journal_db.insert_feature_snapshot(conn, features, market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, score, regime, feature_snapshot_id=fs_id)

    instrument_rules = position_service.resolve_instrument_rules(bingx_symbol, risk_settings)
    service_result = position_service.calculate_active_position(
        results,
        mode,
        total_models=len(config.AI_MODELS),
        current_price=snapshot["current_price"],
        settings=risk_settings,
        instrument_rules=instrument_rules,
        now=now,
    )

    trade_plan_id = journal_db.insert_trade_plan(
        conn,
        service_result.consensus,
        strategy_score_id=score_id,
        symbol=bingx_symbol,
        timestamp=now,
        calculation=service_result.calculation,
        bingx_fields=service_result.bingx_fields,
    )

    # open_virtual_order re-derives "is this actually actionable" from the
    # trade_plans row itself (`_actionable_trade_plan`) rather than trusting
    # `service_result.can_open` blindly -- calling it unconditionally means
    # a WAIT/non-VALID plan just comes back SKIPPED_NOT_ACTIONABLE instead
    # of needing a second, possibly-drifting copy of that gate here.
    order_result = paper_trading.open_virtual_order(conn, trade_plan_id, now=now)

    return AICycleResult(
        ran=True,
        trade_plan_id=trade_plan_id,
        paper_order_status=order_result.status,
        paper_order_reason=order_result.reason,
    )
