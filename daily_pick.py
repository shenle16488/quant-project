"""每日操作指令输出

用法: python daily_pick.py            # 输出沪深300今日建议
      python daily_pick.py --pool 500 # 换成中证500
      python daily_pick.py --realtime # 盘中运行，加实时行情对比

输出示例:
  📅 2026-11-10 ｜ 沪深300 ｜ 推荐 3 只
  ───────────────────────────────────────
  1. 600519 贵州茅台   ⏰ 9:45-10:30  买≤1680.50  止损1546.06  目标1814.94
     理由: 双均线+海龟共振 | 历史胜率 62%（收盘买最高）
  2. 300750 宁德时代   ⏰ 9:45-10:30  买≤218.30   止损200.84   目标235.76
  3. 002594 比亚迪     ⏰ 9:45-10:30  买≤245.10   止损225.49   目标264.71
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from core.screener import screen, order_plan, INDEX_POOLS
from core.timing import run_timing_analysis, aggregate_timing, fetch_realtime_quotes


# 胜率 → 时段映射（A 股日内行为学经验值）
# 历史统计"收盘买胜率最高" → 对应 9:45-10:30（开盘情绪释放后的第一个稳定窗口）
# 历史统计"日内最低买胜率最高" → 对应 14:30-15:00（尾盘可能砸出低点）
# 历史统计"开盘买胜率最高" → 对应 9:30-9:45（极少数强势股适用）
MODE_TO_WINDOW = {
    "open": ("09:30", "09:45"),
    "low":  ("14:30", "15:00"),
    "close":("09:45", "10:30"),  # 收盘附近 → 次日开盘后第一个稳定窗口
    "high": ("避开", "任何时段追高都最差"),
}
MODE_LABEL_CN = {
    "open": "开盘价",
    "low": "日内低点",
    "high": "日内高点",
    "close": "收盘价",
}


def _resolve_best_timing(pool_code, stop_pct=0.08, target_pct=0.16):
    """跑胜率分析，返回整体最佳买入价位 + 对应时段"""
    from core.screener import get_universe
    try:
        universe = get_universe(pool_code)
    except Exception:
        return "close", ("09:45", "10:30"), 0.0

    end = datetime.date.today()
    start = end - datetime.timedelta(days=365 * 3)

    # 只扫 50 只就够了（整体统计不需要全量）
    codes = universe["code"].tolist()[:50]

    results = run_timing_analysis(
        codes, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
        stop_pct=stop_pct, target_pct=target_pct,
        max_hold_days=20,
        signal_key=None, signal_params=None,
    )
    if results is None:
        return "close", ("09:45", "10:30"), 0.0

    agg = aggregate_timing(results)
    best_row = agg.iloc[0]
    best_mode_cn = best_row["买入方式"]
    best_rate = best_row["加权胜率"]

    cn_to_mode = {"开盘买": "open", "日内最低买": "low",
                  "日内最高买": "high", "收盘买": "close"}
    mode = cn_to_mode.get(best_mode_cn, "close")
    window = MODE_TO_WINDOW[mode]
    return mode, window, best_rate


def main():
    ap = argparse.ArgumentParser(description="每日量化操作指令")
    ap.add_argument("--pool", choices=list(INDEX_POOLS.keys()), default="000300",
                    help="选股池（默认沪深300）")
    ap.add_argument("--capital", type=int, default=100000, help="本金（默认10万）")
    ap.add_argument("--realtime", action="store_true", help="盘中运行，拉实时行情对比")
    ap.add_argument("--no-timing", action="store_true",
                    help="跳过胜率统计（省时间，直接用默认时段）")
    args = ap.parse_args()

    pool_label = INDEX_POOLS[args.pool]
    today = datetime.date.today()
    stop_pct, target_pct = 0.08, 0.16  # 2:1 盈亏比

    # 1. 胜率分析 → 确定今日最佳时段
    print(f"⏳ [{today}] 先跑胜率统计（{pool_label} 近3年）...")
    if args.no_timing:
        best_mode = "close"
        best_window = ("09:45", "10:30")
        best_rate = 0.0
    else:
        best_mode, best_window, best_rate = _resolve_best_timing(
            args.pool, stop_pct, target_pct)

    if best_rate > 0:
        print(f"   ✅ 历史上「{MODE_LABEL_CN[best_mode]}」加权胜率最高 "
              f"({best_rate:.1%}) → 今日推荐时段 {best_window[0]}~{best_window[1]}")
    else:
        print(f"   ⚠️ 胜率统计跳过或无数据，默认时段 {best_window[0]}~{best_window[1]}")

    # 2. 扫描候选股
    print(f"⏳ 扫描 {pool_label} 成分股...")
    try:
        res = screen(args.pool, capital=args.capital)
    except Exception as e:
        print(f"❌ 扫描失败：{e}")
        return

    if res["message"] or not res["result"].empty is False:
        print(f"⚠️ {res['message'] or '无候选股'}")
        return

    # 3. 生成操作单
    plan = order_plan(res["result"], args.capital)
    if plan.empty:
        print("⚠️ 今天没有满足条件的标的，建议空仓等待")
        return

    buy = plan[plan["可买"] == True].copy()
    if buy.empty:
        print("⚠️ 候选股都不在买点区间，下面列出等待条件单：")
        print(plan[["代码", "名称", "现价", "买入区间", "买入时机", "距买点", "止损价", "目标价"]].to_string(index=False))
        return

    # 4. 盘中实时对比（可选）
    rt_map = {}
    if args.realtime:
        print(f"⏳ 拉实时行情...")
        rt = fetch_realtime_quotes(buy["代码"].tolist())
        if rt is not None:
            rt_map = dict(zip(rt["代码"].astype(str).str.zfill(6), rt["现价"]))

    # 5. 输出
    print()
    print(f"📅 {today} ｜ {pool_label} ｜ 推荐 {len(buy)} 只 ｜ "
          f"今日最佳时段 ⏰ {best_window[0]}~{best_window[1]} "
          f"（历史胜率 {best_rate:.0%}）" if best_rate > 0 else
          f"📅 {today} ｜ {pool_label} ｜ 推荐 {len(buy)} 只 ｜ "
          f"默认时段 ⏰ {best_window[0]}~{best_window[1]}")
    print("─" * 72)

    for i, (_, row) in enumerate(buy.iterrows(), 1):
        code = str(row["代码"]).zfill(6)
        name = row["名称"]
        buy_price = row["买入价"]
        stop = row["止损价"]
        target = row["目标价"]
        lots = int(row["股数"])
        ratio = row["盈亏比"]
        resonance = int(row["共振数"])

        rt_price = rt_map.get(code)
        rt_note = ""
        if rt_price is not None:
            if float(rt_price) <= float(buy_price):
                rt_note = f"  🟢 现价 {rt_price:.2f} 已在买点内"
            else:
                rt_note = f"  🟡 现价 {rt_price:.2f} 高于买点 {(float(rt_price)/float(buy_price)-1)*100:.1f}%"

        print(f"{i}. {code} {name}")
        print(f"   ⏰ {best_window[0]}~{best_window[1]}  "
              f"买≤{buy_price:.2f}  止损{stop:.2f}  目标{target:.2f}  "
              f"买{lots}股  盈亏比{ratio:.1f}:1")
        print(f"   理由: {resonance}个策略共振{rt_note}")
        print()

    total_amount = buy["金额"].sum()
    print(f"💰 合计 {total_amount:,.0f} 元，占本金 {total_amount/args.capital:.1%}，"
          f"剩余 {args.capital - total_amount:,.0f} 元")
    print(f"📤 止损必须市价委托，止盈必须限价委托，有效期设 90 天")
    print(f"⚠️ 仅供学习研究，历史胜率不代表未来，单只不超 10% 仓位")


if __name__ == "__main__":
    main()
