"""手写向量化回测"""
import numpy as np
import pandas as pd
from .metrics import calc_all_metrics


def _extract_trades(signal, close):
    position_change = signal.diff().fillna(0)
    trades = []
    for date in position_change.index:
        chg = position_change.loc[date]
        if chg > 0:
            trades.append({"Date": date, "Action": "买入", "Price": close.loc[date]})
        elif chg < 0:
            trades.append({"Date": date, "Action": "卖出", "Price": close.loc[date]})
    return pd.DataFrame(trades) if trades else pd.DataFrame(columns=["Date", "Action", "Price"])


def backtest_from_signal(close, signal):
    common_idx = close.index.intersection(signal.index)
    close = close.loc[common_idx].astype(float)
    signal = signal.loc[common_idx].astype(float)

    daily_ret = close.pct_change().fillna(0)
    strategy_ret = signal.shift(1).fillna(0) * daily_ret

    return {
        "daily_returns": strategy_ret,
        "cumulative_return": (1 + strategy_ret).cumprod(),
        "cumulative_benchmark": (1 + daily_ret).cumprod(),
        "metrics": calc_all_metrics(strategy_ret, daily_ret),
        "trades": _extract_trades(signal, close),
    }
