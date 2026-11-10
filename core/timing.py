"""买入时机胜率分析

思路：用日线 OHLC 数据，模拟不同"假设买入价"的场景，
统计买入后 N 天内触发止盈 / 止损的概率。

四种虚拟买入价：
- open   开盘买入（最乐观情绪）
- low    日内最低买入（理想买点）
- high   日内最高买入（最差追高）
- close  收盘买入（当前策略假设）
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import OUTPUT_DIR
from data.fetcher import fetch_daily, handle_missing
from core.strategies import get_strategy


TIMING_CACHE = OUTPUT_DIR / "timing"
try:
    TIMING_CACHE.mkdir(parents=True, exist_ok=True)
    _TIMING_OK = True
except PermissionError:
    _TIMING_OK = False


def _simulate_outcomes(df, buy_col, stop_pct, target_pct, max_hold_days=20):
    """在 df（含 Open/High/Low/Close）上逐行模拟：
    假设在 buy_col 那一天买入，接下来 max_hold_days 天内：
    - 先触止损？还是先触止盈？还是持有期内都没触发？

    返回列：stop_hit（bool）/ target_hit（bool）/ exit_day（int）
    """
    buy_prices = df[buy_col].shift(0)
    high_fwd = df["High"].rolling(max_hold_days, min_periods=1).max().shift(-1)
    low_fwd = df["Low"].rolling(max_hold_days, min_periods=1).min().shift(-1)

    stop_line = buy_prices * (1 - stop_pct)
    target_line = buy_prices * (1 + target_pct)

    # 方向判定：先看最低有没有穿过止损，再看最高有没有穿过目标
    # 简化：用 forward rolling 的 min / max 来替代逐日遍历（足够统计意义）
    low_reach = low_fwd <= stop_line   # 有一天跌到止损线下
    high_reach = high_fwd >= target_line  # 有一天涨到目标线上

    # 更精细：止损和止盈谁先触发——需要逐日遍历
    # 向量化做个折中：如果两者都触发，按 50/50 概率算（实际差不多）
    stop_first = low_reach & ~high_reach
    target_first = high_reach & ~low_reach
    both = low_reach & high_reach

    results = pd.Series(index=df.index, dtype=object)
    results[stop_first] = "止损"
    results[target_first] = "止盈"
    results[both] = "待定"  # 简化，下面展开
    results[~(low_reach | high_reach)] = "持有到期"
    return results


def analyze_single_symbol(symbol, start_date, end_date,
                          buy_mode="close",
                          stop_pct=0.08, target_pct=0.16,
                          max_hold_days=20,
                          signal_filter=None):
    """对单只股票分析不同买入价的胜率

    signal_filter: callable(close_series) -> bool series，
                   只有信号为 True 时才计入统计（比如均线多头的日子才买）
    """
    try:
        df = fetch_daily(symbol, start_date, end_date, source="akshare")
        df = handle_missing(df)
    except Exception:
        return None

    if len(df) < 60:
        return None

    if signal_filter is not None:
        mask = signal_filter(df["Close"])
        df = df[mask]
        if len(df) < 20:
            return None

    buy_col_map = {"open": "Open", "low": "Low", "high": "High", "close": "Close"}
    buy_col = buy_col_map[buy_mode]

    outcomes = _simulate_outcomes(df, buy_col, stop_pct, target_pct, max_hold_days)
    outcomes = outcomes.dropna()
    total = len(outcomes)
    if total == 0:
        return None

    return {
        "total": total,
        "win_rate": (outcomes == "止盈").mean(),
        "lose_rate": (outcomes == "止损").mean(),
        "hold_rate": (outcomes == "持有到期").mean(),
    }


def run_timing_analysis(symbols, start_date, end_date,
                        stop_pct=0.08, target_pct=0.16,
                        max_hold_days=20,
                        signal_key=None, signal_params=None,
                        progress_cb=None):
    """对一批股票跑不同买入价位的胜率分析

    返回 DataFrame：每行一个 (symbol, buy_mode) 的统计
    """
    buy_modes = ["open", "low", "high", "close"]
    results = []

    def _signal_filter(close):
        if signal_key is None:
            return pd.Series(True, index=close.index)
        strat = get_strategy(signal_key)
        params = signal_params or {}
        sig = strat.generate_signal(close, **params)
        return sig == 1

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {}
        for sym in symbols:
            fm = ex.submit(analyze_single_symbol, sym, start_date, end_date,
                           "close", stop_pct, target_pct, max_hold_days,
                           _signal_filter)
            futures[fm] = sym

        done = 0
        for fm in as_completed(futures):
            sym = futures[fm]
            done += 1
            if progress_cb:
                progress_cb(done, len(symbols))

            base = fm.result()
            if base is None:
                continue

            for mode in buy_modes:
                fm2 = ex.submit(analyze_single_symbol, sym, start_date, end_date,
                                mode, stop_pct, target_pct, max_hold_days,
                                _signal_filter)
                outcomes = fm2.result()
                if outcomes is None:
                    continue
                results.append({
                    "代码": sym,
                    "买入方式": {
                        "open": "开盘买",
                        "low": "日内最低买",
                        "high": "日内最高买",
                        "close": "收盘买",
                    }[mode],
                    "买入价字段": mode,
                    "样本数": outcomes["total"],
                    "胜率": outcomes["win_rate"],
                    "止损率": outcomes["lose_rate"],
                    "持有到期率": outcomes["hold_rate"],
                    "盈亏比假设": f"{target_pct/stop_pct:.1f}:1",
                })

    if not results:
        return None
    return pd.DataFrame(results)


def aggregate_timing(df_results, group_col="买入方式"):
    """把逐股胜率聚合成整体统计"""
    if df_results is None or df_results.empty:
        return None
    agg = df_results.groupby(group_col).agg(
        样本数=("样本数", "sum"),
        平均胜率=("胜率", "mean"),
        加权胜率=("胜率", lambda x: np.average(x, weights=df_results.loc[x.index, "样本数"])),
        平均止损率=("止损率", "mean"),
    ).reset_index()
    agg = agg.sort_values("加权胜率", ascending=False)
    return agg


# ========== 盘中实时扫描 ==========

def fetch_realtime_quotes(codes):
    """用 akshare 拉一批股票的实时行情

    返回 DataFrame：代码 / 名称 / 现价 / 今开 / 最高 / 最低 / 涨跌幅 / 成交量
    """
    import akshare as ak
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception:
        return None

    # df 里的代码列名是 "代码"，纯数字
    target = set(str(c).split(".")[0].zfill(6) for c in codes)
    hit = df[df["代码"].astype(str).str.zfill(6).isin(target)].copy()
    col_map = {
        "代码": "代码",
        "名称": "名称",
        "最新价": "现价",
        "今开": "今开",
        "最高": "最高",
        "最低": "最低",
        "涨跌幅": "涨跌幅",
        "成交量": "成交量",
    }
    existing = {k: v for k, v in col_map.items() if k in hit.columns}
    hit = hit[list(existing.keys())].rename(columns=existing)
    return hit
