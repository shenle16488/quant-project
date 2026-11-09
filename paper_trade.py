r"""模拟盘自动交易命令行入口（虚拟资金，不产生真实委托）

运行：
    cd D:\Software\quant-project
    D:\Software\quant-env\Scripts\python.exe paper_trade.py --pool 000300 --capital 100000

计划任务 QuantPaperTrade 每天 15:40 自动执行同一命令。
"""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import REPORT_DIR
from core.paper_trader import run_once
from core.screener import INDEX_POOLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="000300", choices=list(INDEX_POOLS),
                    help="股票池：000300 沪深300 / 000905 中证500 / 000852 中证1000")
    ap.add_argument("--capital", type=float, default=100000,
                    help="虚拟本金（首次开户生效）")
    args = ap.parse_args()

    print("=" * 74)
    print(f"模拟盘自动交易 · {INDEX_POOLS[args.pool]} · "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 74)

    def _progress(done, total):
        if done % 50 == 0 or done == total:
            print(f"  扫描进度 {done}/{total}", flush=True)

    res = run_once(args.pool, capital=args.capital, progress_cb=_progress)

    print(f"\n{res['message']}")

    if res["trades"]:
        print("\n本次虚拟成交：")
        for t in res["trades"]:
            pnl = f"  盈亏 {t['盈亏']:+,.0f}" if t["方向"] == "卖出" else ""
            print(f"  {t['方向']} {t['代码']} {t['名称']} "
                  f"{t['股数']}股 @ {t['价格']:.2f}  [{t['原因']}]{pnl}")
    else:
        print("\n本次无成交，持仓保持不变。")

    print(f"\n现金 {res['cash']:,.2f} | 持仓市值 {res['market_value']:,.2f} | "
          f"总资产 {res['total']:,.2f} | 累计 {res['pnl_pct']:+.2%}")

    if res["positions"]:
        print("\n当前持仓：")
        for p in res["positions"]:
            print(f"  {p['代码']} {p['name']} {p['shares']}股 "
                  f"成本 {p['cost']:.3f} 现价 {p['last_price']:.2f} "
                  f"止损 {p['stop']:.2f} 目标 {p['target']:.2f}")

    today = datetime.date.today().strftime("%Y%m%d")
    out = REPORT_DIR / f"paper_trade_{today}.csv"
    if res["trades"]:
        import pandas as pd
        pd.DataFrame(res["trades"]).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n✅ 成交明细已存: {out}")

    print("\n⚠️ 以上均为虚拟成交，账户里没有一分真钱，也没有向券商发送任何委托。")


if __name__ == "__main__":
    main()
