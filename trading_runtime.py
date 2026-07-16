"""Entry point: `python trading_runtime.py`. Runs as its own long-lived
process, separate from server.py (which stays HTTP/dashboard-only) — see
runtime/service.py for the actual loop and the plan for why the two are
kept apart (a dashboard restart must never interrupt position monitoring,
and vice versa).
"""

from __future__ import annotations

import config
from runtime.service import TradingRuntime


def main() -> None:
    runtime = TradingRuntime(config.SYMBOL)
    runtime.run()


if __name__ == "__main__":
    main()
