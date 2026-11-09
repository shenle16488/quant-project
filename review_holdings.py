r"""持仓体检命令行入口

运行：
    D:\Software\quant-env\Scripts\python.exe review_holdings.py

读取模拟盘持仓（output/paper/state.json），逐只给出
「卖出 / 减仓 / 继续持有」的结论和理由，并存成 CSV。
"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import REPORT_DIR
from core.holdings_review import review_all, summary_text
from core.paper_trader import read_state


def main():
    print("=" * 70)
    print(f"持仓体检 {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)

    state = read_state()
    df = review_all(state)
    if df.empty:
        print("\n📭 当前空仓，没有需要体检的持仓。")
        return

    show = df.copy()
    show["浮盈%"] = (show["浮盈%"] * 100).round(2).astype(str) + "%"
    show["20日涨幅"] = (show["20日涨幅"] * 100).round(1).astype(str) + "%"
    show["距年线"] = (show["距年线"] * 100).round(1).astype(str) + "%"

    print("\n" + show[["代码", "名称", "股数", "现价", "浮盈%",
                      "共振数", "结论", "建议动作"]].to_string(index=False))
    print("\n" + "-" * 70)
    for _, r in df.iterrows():
        print(f"  {r['代码']} {r['名称']}：{r['结论']}（{r['建议动作']}）— {r['理由']}")

    print("\n" + summary_text(df))
    out = REPORT_DIR / f"review_{datetime.date.today():%Y%m%d}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ 已存: {out}")
    print("\n⚠️ 体检结论基于收盘价与固定规则，不构成投资建议。")


if __name__ == "__main__":
    main()
