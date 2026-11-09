"""选股扫描核心逻辑（命令行与看板共用）

流程：
1. 取指数成分股（沪深300 / 中证500 / 中证1000）
2. 并发拉日线，跑 3 个策略，统计当前处于「持有多头」状态的策略数
3. 共振 ≥2 个策略的入选，再过健康度门槛
4. 返回候选清单

健康度门槛的作用：布林带回归是「跌到超卖就买」，在下跌趋势里会一路接刀。
没有这层过滤，扫出来的全是 20 日跌幅 -20%~-50% 的暴跌股。
"""
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, STRATEGY_PARAMS
from core.strategies import get_strategy
from data.fetcher import fetch_daily, handle_missing

LOOKBACK_DAYS = 400
MAX_WORKERS = 5
MIN_STRATEGY_HITS = 2

MAX_VOL = 0.50            # 年化波动 ≤ 50%
MIN_DIST_MA = -0.12       # 距 250 日均线不低于 -12%
MAX_MDD = -0.40           # 近 1 年最大回撤不深于 -40%
MIN_RET20 = -0.20         # 近 20 日跌幅不超过 -20%

# 交易价位参数（规则固定、可解释，不做任何"预测"）
BUY_ZONE_PCT = 0.015      # 建议买入区间：现价 ±1.5%
STOP_LOSS_PCT = 0.08      # 止损：跌破买入价 8%
REWARD_RATIO = 2.0        # 目标价：按 2:1 盈亏比推算
CAPITAL = 100000          # 默认本金（元）
MAX_POSITION_PCT = 0.10   # 单只股票最多占总资金 10%
MAX_CHASE_PCT = 0.08      # 现价高于 20 日线 8% 视为追高，建议等回踩
MAX_ORDERS = 5            # 一次操作单最多买几只（与模拟盘持仓上限一致）

INDEX_POOLS = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
}

CACHE_DIR = OUTPUT_DIR / "cache"

COLUMNS = ["代码", "名称", "共振数", "共振策略", "收盘价",
           "昨开", "昨收", "今开", "跳空%",
           "建议买入", "止损价", "目标价", "建议股数", "操作建议",
           "距半年高点", "20日涨幅", "年化波动", "距年线", "近1年最大回撤"]


def get_universe(index_code="000300"):
    """指数成分股名单，缓存 30 天"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"index_cons_{index_code}.csv"
    if cache.exists():
        age = (datetime.date.today() -
               datetime.date.fromtimestamp(cache.stat().st_mtime)).days
        if age <= 30:
            df = pd.read_csv(cache, dtype={"code": str})
            df["code"] = df["code"].str.zfill(6)
            return df

    import akshare as ak
    raw = ak.index_stock_cons_csindex(symbol=index_code)
    df = pd.DataFrame({
        "code": raw["成分券代码"].astype(str).str.zfill(6),
        "name": raw["成分券名称"].astype(str).str.replace(" ", "", regex=False),
    }).drop_duplicates("code").reset_index(drop=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def risk_profile(close):
    """客观风险指标，不含策略观点"""
    ret = close.pct_change().dropna()
    vol = float(ret.tail(60).std() * np.sqrt(252)) if len(ret) >= 30 else np.nan
    ma250 = close.rolling(250).mean().iloc[-1] if len(close) >= 250 else np.nan
    dist_ma = float(close.iloc[-1] / ma250 - 1) if pd.notna(ma250) and ma250 else np.nan
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else np.nan
    cum = (1 + ret).cumprod()
    mdd = float((cum / cum.cummax() - 1).min()) if len(cum) else np.nan
    return vol, dist_ma, ret20, mdd


def trade_plan(close, capital=CAPITAL):
    """按固定规则给出买入区间 / 止损 / 目标 / 建议股数 / 一句话建议

    规则全部是"机械计算"，不是预测：
    - 买入区间：现价 ±1.5%（同一买点，避免为了几分钱错过）
    - 止损：近 20 日最低价下方 1% 与 -8% 中较近的一个，但不紧于 -3%
    - 目标：按 2:1 盈亏比推算，且不越过近半年最高价（压力位）
    - 股数：单只最多 10% 本金，按 100 股一手向下取整
    """
    p = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else p
    low20 = float(close.tail(20).min())
    high120 = float(close.tail(120).max()) if len(close) >= 120 else float(close.max())

    # 买入参考价：正常情况用现价；追高/在均线下方时用 20 日线做参照，和一句话建议保持一致
    dev = p / ma20 - 1 if ma20 else 0.0
    if dev > MAX_CHASE_PCT:
        center = ma20
        advice = f"高于20日线{dev * 100:.0f}%，等回踩{ma20:.2f}"
    elif p < ma20:
        center = ma20
        advice = f"在20日线下方，站上{ma20:.2f}再买"
    else:
        center = p
        advice = ""

    buy_lo = round(center * (1 - BUY_ZONE_PCT), 2)
    buy_hi = round(center * (1 + BUY_ZONE_PCT), 2)

    struct_stop = round(low20 * 0.99, 2)
    pct_stop = round(buy_lo * (1 - STOP_LOSS_PCT), 2)
    stop = max(struct_stop, pct_stop)             # 取较近的，控制单笔亏损
    stop = min(stop, round(buy_lo * 0.97, 2))     # 但不允许比 -3% 更紧

    target = round(buy_hi + REWARD_RATIO * (buy_hi - stop), 2)
    if high120 > buy_hi:
        target = min(target, round(high120, 2))   # 前高是压力位，目标不越过

    if not advice:
        advice = f"现价分批买，跌破{stop:.2f}离场"

    budget = capital * MAX_POSITION_PCT
    lots = int(budget // (p * 100))
    if lots > 0:
        shares_txt = f"{lots * 100}股(约{lots * 100 * p:,.0f}元)"
    else:
        shares_txt = f"1手需{p * 100:,.0f}元，超上限"

    return {
        "建议买入": f"{buy_lo:.2f} ~ {buy_hi:.2f}",
        "止损价": stop,
        "目标价": target,
        "建议股数": shares_txt,
        "操作建议": advice,
        "距半年高点": float(p / high120 - 1) if high120 else np.nan,
    }


def scan_one(code, name, start, end, capital=CAPITAL):
    """扫一只股票：返回结果 dict 或 None"""
    try:
        df = handle_missing(fetch_daily(code, start, end, source="akshare"))
        close = df["Close"]
    except Exception:
        return None
    if len(close) < 60:
        return None

    hits = []
    for key, params in STRATEGY_PARAMS.items():
        p = dict(params)
        if "exit" in p:
            p["exit_"] = p.pop("exit")

        need = max(v for v in p.values() if isinstance(v, (int, float))) + 5
        if len(close) < need:
            continue

        try:
            strat = get_strategy(key)
            sig = strat.generate_signal(close, **p)
        except Exception:
            continue

        if int(sig.iloc[-1]) == 1:
            held = 0
            for v in reversed(sig.tolist()):
                if int(v) == 1:
                    held += 1
                else:
                    break
            hits.append((strat.name, held))

    if not hits:
        return None

    vol, dist_ma, ret20, mdd = risk_profile(close)

    open_ = df["Open"]
    prev_close = float(close.iloc[-2])
    today_open = float(open_.iloc[-1])
    gap = today_open / prev_close - 1 if prev_close else float("nan")

    row = {
        "代码": code,
        "名称": name,
        "共振数": len(hits),
        "共振策略": " + ".join(f"{n}({h}天)" for n, h in hits),
        "收盘价": round(float(close.iloc[-1]), 2),
        "昨开": round(float(open_.iloc[-2]), 2),
        "昨收": round(prev_close, 2),
        "今开": round(today_open, 2),
        "跳空%": gap,
        "20日涨幅": ret20,
        "年化波动": vol,
        "距年线": dist_ma,
        "近1年最大回撤": mdd,
    }
    row.update(trade_plan(close, capital))
    return row


def screen(index_code="000300", top_n=None, min_hits=MIN_STRATEGY_HITS,
           progress_cb=None, capital=CAPITAL):
    """扫描并返回结果

    参数:
        index_code: 指数代码，000300 / 000905 / 000852
        top_n: 最多扫多少只（None = 全部成分股）
        min_hits: 至少几个策略共振
        progress_cb: 进度回调 callback(done, total)
        capital: 本金，用于计算建议股数（单只最多 10%）

    返回 dict: result / all_hits / scanned / failed / message / capital
    """
    uni = get_universe(index_code)
    uni = uni[~uni["name"].str.contains("ST|退|停", na=False)]
    if top_n:
        uni = uni.head(top_n)
    uni = uni.reset_index(drop=True)

    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() -
             datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")

    rows, failed, done = [], 0, 0
    total = len(uni)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_one, r.code, r.name, start, end, capital): r.code
                   for r in uni.itertuples()}
        for fut in as_completed(futures):
            done += 1
            if progress_cb:
                progress_cb(done, total)
            try:
                res = fut.result()
            except Exception:
                failed += 1
                continue
            if res:
                rows.append(res)

    out = pd.DataFrame(rows)
    if out.empty:
        return {"result": out, "all_hits": out, "scanned": total,
                "failed": failed, "capital": capital,
                "message": "当前没有任何股票触发买入信号"}

    hits = out[out["共振数"] >= min_hits].copy()
    if hits.empty:
        return {"result": hits, "all_hits": hits, "scanned": total,
                "failed": failed, "capital": capital,
                "message": f"没有股票达到 {min_hits} 个策略共振"}

    healthy = hits[
        (hits["年化波动"] <= MAX_VOL)
        & (hits["距年线"] >= MIN_DIST_MA)
        & (hits["近1年最大回撤"] >= MAX_MDD)
        & (hits["20日涨幅"] >= MIN_RET20)
    ].copy()

    if healthy.empty:
        return {"result": healthy, "all_hits": hits, "scanned": total,
                "failed": failed, "capital": capital,
                "message": (f"{len(hits)} 只共振股全部未通过健康度门槛，"
                            "说明当前市场处于急跌阶段，此时买入信号多为接刀")}

    healthy = healthy.sort_values(["共振数", "距年线"],
                                  ascending=[False, False]).reset_index(drop=True)
    return {"result": healthy[COLUMNS], "all_hits": hits, "scanned": total,
            "failed": failed, "capital": capital, "message": ""}


def format_for_display(df):
    """百分比列转成带 % 的字符串，供表格展示"""
    show = df.copy()
    for col in ["距半年高点", "20日涨幅", "年化波动", "距年线", "近1年最大回撤",
                "距买点", "浮盈%", "跳空%"]:
        if col in show:
            show[col] = (show[col] * 100).round(1).astype(str) + "%"
    return show


def order_plan(df, capital=CAPITAL, min_ratio=1.5):
    """把选股结果变成可执行操作单：买哪只、什么价买、买多少、什么价卖

    规则：
    - 买入价：现价在建议区间内就用现价，否则用区间中点作为等待价
    - 可买：现价处于建议区间内 + 至少能买 100 股 + 盈亏比 ≥ 1.5
    - 盈亏比 =（目标价 - 买入价）/（买入价 - 止损价）
    - 股数：单只最多 10% 本金，按 100 股一手向下取整
    """
    cols = ["代码", "名称", "现价", "买入区间", "买入价", "股数", "金额",
            "止损价", "目标价", "盈亏比", "买入时机", "距买点", "共振数", "备注", "可买"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)

    budget = capital * MAX_POSITION_PCT
    rows = []
    picked = 0
    for _, s in df.iterrows():
        price = float(s["收盘价"])
        stop = float(s["止损价"])
        target = float(s["目标价"])
        try:
            lo, hi = [float(v) for v in str(s["建议买入"]).split("~")]
        except (ValueError, AttributeError):
            continue

        in_zone = lo <= price <= hi
        entry = price if in_zone else round((lo + hi) / 2, 2)

        if in_zone:
            when = f"今天挂限价 {price:.2f}"
            dist = 0.0
        elif price > hi:
            when = f"等回踩 {lo:.2f}~{hi:.2f}"
            dist = price / hi - 1
        else:
            when = f"等收盘站上 {lo:.2f}"
            dist = price / lo - 1

        risk = entry - stop
        ratio = (target - entry) / risk if risk > 0 else 0.0
        lots = int(budget // (entry * 100))
        shares = lots * 100
        amount = shares * entry

        note = ""
        ok = in_zone and shares > 0 and ratio >= min_ratio
        if shares <= 0:
            note = f"1 手需 {entry * 100:,.0f} 元，超过单只 10% 上限"
        elif ratio < min_ratio:
            note = f"目标价被近半年高点压制，盈亏比只有 {ratio:.1f}:1"
        elif not in_zone:
            note = "现价不在买入区间，等价格到位再动手"
        elif picked >= MAX_ORDERS:
            note = f"已满 {MAX_ORDERS} 只持仓上限，列为备选"
            ok = False
        if ok:
            picked += 1

        rows.append({
            "代码": s["代码"], "名称": s["名称"],
            "现价": round(price, 2),
            "买入区间": s["建议买入"],
            "买入价": round(entry, 2),
            "股数": shares,
            "金额": round(amount, 0),
            "止损价": stop,
            "目标价": target,
            "盈亏比": round(ratio, 1),
            "买入时机": when,
            "距买点": round(dist, 4),
            "共振数": int(s["共振数"]),
            "备注": note,
            "可买": bool(ok),
        })
    return pd.DataFrame(rows, columns=cols)
