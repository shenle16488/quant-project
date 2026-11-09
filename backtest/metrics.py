"""绩效指标计算"""
import numpy as np
import pandas as pd
from config import RISK_FREE_RATE


def calc_annual_return(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).prod()
    return float(cum ** (252 / len(returns)) - 1)


def calc_annual_volatility(returns):
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(252))


def calc_sharpe_ratio(returns, risk_free_rate=None):
    risk_free_rate = risk_free_rate or RISK_FREE_RATE
    ann_ret = calc_annual_return(returns)
    ann_vol = calc_annual_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def calc_max_drawdown(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    return float((cum / cum.cummax() - 1).min())


def calc_all_metrics(returns, benchmark_returns=None):
    returns = returns.dropna()
    cum_s = float((1 + returns).prod() - 1) if len(returns) > 0 else 0.0
    result = {
        "累计收益": cum_s,
        "年化收益率": calc_annual_return(returns),
        "年化波动率": calc_annual_volatility(returns),
        "最大回撤": calc_max_drawdown(returns),
        "夏普比率": calc_sharpe_ratio(returns),
        "交易日数": len(returns),
    }
    if benchmark_returns is not None:
        bench = benchmark_returns.dropna()
        result["买入持有累计收益"] = float((1 + bench).prod() - 1)
        result["买入持有年化"] = calc_annual_return(bench)
    return result


def format_metrics(metrics):
    rows = []
    for key, val in metrics.items():
        if any(k in key for k in ["收益", "回撤", "波动", "胜率"]):
            rows.append({"指标": key, "数值": f"{val:.2%}"})
        elif isinstance(val, float):
            rows.append({"指标": key, "数值": f"{val:.4f}"})
        else:
            rows.append({"指标": key, "数值": val})
    return pd.DataFrame(rows)
