"""The long-lived scheduler that actually calls paper_trading.process_tick
and execution_engine.monitor on a repeating cadence — see
runtime.service.TradingRuntime and trading_runtime.py (the entry point).
"""
