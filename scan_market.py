r"""全市场选股扫描器

做什么：
1. 拉全市场实时快照，剔除 ST / 退市 / 停牌 / 低价 / 低流动性
2. 按成交额排序，取最活跃的前 N 只
3. 并发拉日线，跑 3 个策略（双均线 / 海龟 / 布林带）
4. 输出「当前有几个策略同时认为应该持有」的候选清单

怎么读结果：
- 共振策略数 = 3 最好，2 次之。1 个策略的单独信号噪音很大。
- 附加了客观风险数据（波动、回撤、距年线），自己按承受力挑。

必须清楚：这是策略信号的筛选，不是收益预测。教程里所有回测都证明了
单靠信号无法稳定赚钱，历史信号不代表未来。仓位与止损仍需自己决定。

运行：
    cd D:\Software\quant-project
    D:\Software\quant-env\Scripts\python.exe scan_market.py --top 200
"""
import argparse
import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, REPORT_DIR, STRATEGY_PARAMS
from core.strategies import get_strategy
from data.fetcher import fetch_daily, handle_missing

LOOKBACK_DAYS = 400
MAX_WORKERS = 5           # 并发别太高，避免被数据源封
DEFAULT_TOP_N = 300
MIN_STRATEGY_HITS = 2     # 至少 2 个策略共振才入选

# 健康度门槛：没有这几条，扫出来的全是「暴跌到布林下轨」的接刀股
MAX_VOL = 0.50            # 年化波动 ≤ 50%
MIN_DIST_MA = -0.12       # 距 250 日均线不低于 -12%（趋势没崩）
MAX_MDD = -0.40           # 近 1 年最大回撤不深于 -40%
MIN_RET20 = -0.20         # 近 20 日跌幅不超过 -20%（防接刀）

# 可选股票池：指数成分股（流动性好、可交易性强，比随机全市场更合理）
INDEX_POOLS = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
}

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_universe(index_code="000300"):
    """指数成分股名单，缓存 30 天（成分股半年才调整一次）"""
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
    """客观风险指标，不含任何策略观点"""
    ret = close.pct_change().dropna()

    vol = float(ret.tail(60).std() * np.sqrt(252)) if len(ret) >= 30 else np.nan

    ma250 = close.rolling(250).mean().iloc[-1] if len(close) >= 250 else np.nan
    dist_ma = float(close.iloc[-1] / ma250 - 1) if pd.notna(ma250) and ma250 else np.nan

    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else np.nan

    cum = (1 + ret).cumprod()
    mdd = float((cum / cum.cummax() - 1).min()) if len(cum) else np.nan

    return vol, dist_ma, ret20, mdd


def scan_one(code, name, start, end):
    """扫一只股票：返回共振结果，或 None"""
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
    return {
        "代码": code,
        "名称": name,
        "共振数": len(hits),
        "共振策略": " + ".join(f"{n}({h}天)" for n, h in hits),
        "收盘价": round(float(close.iloc[-1]), 2),
        "20日涨幅": ret20,
        "年化波动": vol,
        "距年线": dist_ma,
        "近1年最大回撤": mdd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="000300", choices=list(INDEX_POOLS),
                    help="股票池：000300 沪深300 / 000905 中证500 / 000852 中证1000")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                    help=f"最多扫描 N 只（默认 {DEFAULT_TOP_N}）")
    ap.add_argument("--min-hits", type=int, default=MIN_STRATEGY_HITS,
                    help=f"至少几个策略共振（默认 {MIN_STRATEGY_HITS}）")
    args = ap.parse_args()

    pool_name = INDEX_POOLS[args.index]
    print("=" * 72)
    print(f"选股扫描 · {pool_name}  {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 72)

    print(f"\n[1/3] 读取 {pool_name} 成分股 ...")
    uni = get_universe(args.index)
    uni = uni[~uni["name"].str.contains("ST|退|停", na=False)].head(args.top)
    print(f"      待扫描 {len(uni)} 只")

    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")

    print(f"\n[2/3] 并发拉日线并跑策略（{MAX_WORKERS} 线程，约 1~3 分钟）...")
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_one, r.code, r.name, start, end): r.code
                   for r in uni.itertuples()}
        for fut in as_completed(futures):
            done += 1
            if done % 20 == 0:
                print(f"      进度 {done}/{len(futures)}，已命中 {len(rows)} 只", flush=True)
            res = fut.result()
            if res:
                rows.append(res)

    print(f"\n[3/3] 汇总 ...")
    if not rows:
        print("\n📭 当前没有任何股票满足 ≥2 个策略共振。这是正常结果，说明市场处于震荡或下跌状态。")
        return

    out = pd.DataFrame(rows)
    raw_count = len(out)
    out = out[out["共振数"] >= args.min_hits]
    if out.empty:
        print(f"\n📭 没有股票达到 {args.min_hits} 个策略共振。")
        return

    # 健康度门槛：布林带在下跌趋势里会一路接刀，必须排除
    healthy = out[
        (out["年化波动"] <= MAX_VOL)
        & (out["距年线"] >= MIN_DIST_MA)
        & (out["近1年最大回撤"] >= MAX_MDD)
        & (out["20日涨幅"] >= MIN_RET20)
    ].copy()
    if healthy.empty:
        print(f"\n📭 {len(out)} 只共振股全部未通过健康度门槛"
              f"（波动 ≤50%、距年线 ≥-12%、回撤 ≤-40%、20日跌幅 ≤-20%）。")
        print("    这通常意味着当前市场处于急跌阶段，此时任何买入信号都属于接刀。")
        return

    healthy = healthy.sort_values(["共振数", "距年线"], ascending=[False, False]).reset_index(drop=True)

    show = healthy.copy()
    for col in ["20日涨幅", "年化波动", "距年线", "近1年最大回撤"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"

    print(f"\n{'=' * 78}")
    print(f"候选清单：{len(show)} 只通过健康度门槛"
          f"（共振 {len(out)} 只中筛出）")
    print("=" * 78)
    print(show.to_string(index=False, justify="left"))

    today = datetime.date.today().strftime("%Y%m%d")
    out_file = REPORT_DIR / f"scan_{today}.csv"
    healthy.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已存: {out_file}")

    print("""
⚠️  这不是投资建议，只是「几个策略当前都认为该持有 + 走势没崩」的客观记录。
    教程实测结论：单靠信号无法稳定赚钱（双均线跑输买入持有、配对交易亏损）。
    共振数只代表策略观点一致，不代表胜率更高。仓位、止损、卖出时机仍由你自己决定。
""")


if __name__ == "__main__":
    main()
