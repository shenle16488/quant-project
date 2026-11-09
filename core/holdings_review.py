"""持仓体检：逐只判断「还该不该继续拿着」

判断顺序先硬后软：
1. 现价 ≤ 止损价        → 卖出（硬止损，不再讨论）
2. 现价 ≥ 目标价        → 卖出（止盈）
3. 看多策略不足 2 个     → 卖出（策略转空）
4. 跌破 20 日线且 20 日跌幅 > 8%  → 减仓一半
5. 距 250 日线低于 -5%   → 减仓一半
6. 单日跌幅 ≥ 6%        → 减仓一半（异常波动先降风险）
7. 其余                 → 继续持有
"""
import datetime

import pandas as pd

from config import STRATEGY_PARAMS
from core.strategies import get_strategy
from data.fetcher import fetch_daily, handle_missing

LOOKBACK_DAYS = 400
MIN_HITS = 2

COLUMNS = ["代码", "名称", "股数", "成本价", "现价", "浮盈%", "20日涨幅",
           "距年线", "共振数", "结论", "建议动作", "理由"]


def _count_hits(close):
    """当前有几个策略看多"""
    hits = 0
    for key, params in STRATEGY_PARAMS.items():
        p = dict(params)
        if "exit" in p:
            p["exit_"] = p.pop("exit")
        need = max(v for v in p.values() if isinstance(v, (int, float))) + 5
        if len(close) < need:
            continue
        try:
            sig = get_strategy(key).generate_signal(close, **p)
            if int(sig.iloc[-1]) == 1:
                hits += 1
        except Exception:
            continue
    return hits


def review_one(code, pos):
    """体检单只，返回一行结论；数据失败返回 None"""
    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() -
             datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    try:
        df = handle_missing(fetch_daily(code, start, end, source="akshare"))
        close = df["Close"]
        if len(close) < 60:
            return None
    except Exception:
        return None

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else price
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma250 = float(close.rolling(250).mean().iloc[-1]) if len(close) >= 250 else None
    ret20 = price / float(close.iloc[-21]) - 1 if len(close) > 21 else 0.0
    dist_ma = price / ma250 - 1 if ma250 else 0.0
    day_chg = price / prev - 1
    hits = _count_hits(close)

    hard, soft = [], []
    if price <= pos["stop"]:
        hard.append(f"跌破止损价 {pos['stop']:.2f}")
    if price >= pos["target"]:
        hard.append(f"到达目标价 {pos['target']:.2f}")
    if hits < MIN_HITS:
        hard.append(f"只剩 {hits} 个策略看多")

    if price < ma20 and ret20 <= -0.08:
        soft.append(f"跌破 20 日线且 20 日跌 {ret20:.1%}")
    if dist_ma <= -0.05:
        soft.append(f"距年线 {dist_ma:.1%}")
    if day_chg <= -0.06:
        soft.append(f"单日跌 {day_chg:.1%}")

    if hard:
        verdict, action, reasons = "卖出", "全部卖出", "；".join(hard)
    elif soft:
        verdict, action, reasons = "减仓", "卖出一半", "；".join(soft)
    else:
        verdict, action, reasons = "持有", "继续持有", "未触发任何卖出条件"

    return {
        "代码": code,
        "名称": pos["name"],
        "股数": pos["shares"],
        "成本价": pos["cost"],
        "现价": round(price, 2),
        "浮盈%": round(price / pos["cost"] - 1, 4),
        "20日涨幅": round(ret20, 4),
        "距年线": round(dist_ma, 4),
        "共振数": hits,
        "结论": verdict,
        "建议动作": action,
        "理由": reasons,
    }


def review_all(state):
    """体检全部持仓，卖出/减仓排前面"""
    rows = []
    for code, pos in state.get("positions", {}).items():
        row = review_one(code, pos)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows, columns=COLUMNS)
    order = {"卖出": 0, "减仓": 1, "持有": 2}
    df["_o"] = df["结论"].map(order)
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def summary_text(df):
    """一句话汇总，供推送 / 打印"""
    if df.empty:
        return "当前空仓，无需体检。"
    counts = df["结论"].value_counts().to_dict()
    parts = [f"{df['代码'].iloc[0]} 等 {len(df)} 只持仓"]
    for k in ["卖出", "减仓", "持有"]:
        if k in counts:
            parts.append(f"{k} {counts[k]} 只")
    return "，".join(parts)
