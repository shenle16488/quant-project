r"""每日信号扫描器（半自动交易：程序出信号 → 通知 → 人工下单）

教程落地路径：
- 策略代码完全复用教程第 12/13/14 章（core/strategies.py）
- 严格执行教程纪律：信号基于"昨日收盘"，次日才能执行（shift 思想）
- A 股 15:00 收盘，建议每天 15:30 之后运行本脚本

运行方式（cmd）：
    cd D:\Software\quant-project
    D:\Software\quant-env\Scripts\python.exe daily_signal.py

可选：微信推送（Server酱）
    1. 微信打开 https://sct.ftqq.com 扫码登录，拿到 SendKey
    2. 在下面 PUSHER["sendkey"] 填入你的 SendKey
"""
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import pandas as pd

from config import STOCK_POOL, REPORT_DIR, STRATEGY_PARAMS
from data.fetcher import fetch_daily, handle_missing
from core.strategies import get_strategy

# ========== 微信推送配置（可选，不用就留空）==========
PUSHER = {
    "enabled": False,          # 拿到 SendKey 后改为 True
    "sendkey": "",             # 例: "SCT123456Txxxxxx"
}

# 扫描近 60 个自然日足够覆盖最长均线（200 日均线需 200 交易日，
# 这里直接取近 1 年数据保证 MA200 有效，可自行缩短）
LOOKBACK_DAYS = 365


def fetch_with_retry(symbol, start, end, retries=2):
    """拉数据，失败重试，间隔 10 秒"""
    import time
    for attempt in range(retries + 1):
        try:
            df = fetch_daily(symbol, start, end, source="akshare")
            return handle_missing(df)
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(10)
    return None


def check_signals_for_stock(symbol: str, name: str, close: pd.Series):
    """
    对一只股票的收盘价序列，跑全部 3 个策略，检查最新交易日的新信号
    （数据只拉一次，多策略共用，避免频繁请求触发限流）
    """
    hits = []
    for strat_key, params in STRATEGY_PARAMS.items():
        p = dict(params)
        if "exit" in p:
            p["exit_"] = p.pop("exit")

        min_len = max(v for v in p.values() if isinstance(v, (int, float))) + 5
        if len(close) < min_len:
            continue

        try:
            strategy = get_strategy(strat_key)
            signal = strategy.generate_signal(close, **p)
            last_change = signal.diff().iloc[-1]
            if last_change == 1:
                action = "买入"
            elif last_change == -1:
                action = "卖出"
            else:
                continue
            hits.append({
                "日期": close.index[-1].strftime("%Y-%m-%d"),
                "股票": f"{symbol} {name}",
                "策略": strategy.name,
                "信号": action,
                "收盘价": round(float(close.iloc[-1]), 2),
                "参数": str(params),
            })
        except Exception as e:
            print(f"  ⚠️ {name} {strat_key} 策略异常: {e}")
    return hits


def push_wechat(title: str, content: str):
    """Server酱微信推送（需自行申请 SendKey）"""
    if not PUSHER["enabled"] or not PUSHER["sendkey"]:
        return
    import urllib.request
    import urllib.parse
    url = f"https://sctapi.ftqq.com/{PUSHER['sendkey']}.send"
    data = urllib.parse.urlencode({"title": title, "desp": content}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
        print("📨 微信推送已发送")
    except Exception as e:
        print(f"⚠️ 微信推送失败: {e}")


def main():
    print("=" * 60)
    print(f"📡 每日信号扫描 {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 60)

    signals = []
    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")

    for symbol, name in STOCK_POOL.items():
        try:
            # 每股只拉 1 次数据，3 个策略共用（请求量从 30 次降到 10 次）
            df = fetch_with_retry(symbol, start, end)
            hits = check_signals_for_stock(symbol, name, df["Close"])
            for hit in hits:
                signals.append(hit)
                print(f"  🔔 {hit['股票']} | {hit['策略']} | {hit['信号']} @ {hit['收盘价']}")
        except Exception as e:
            print(f"  ⚠️ {name} 数据获取失败: {e}")

    print("-" * 60)
    today = datetime.date.today().strftime("%Y-%m-%d")

    if signals:
        df_out = pd.DataFrame(signals)
        out_file = REPORT_DIR / f"signals_{today}.csv"
        df_out.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"✅ 共 {len(signals)} 条信号，已存: {out_file}")

        # Windows 蜂鸣提醒
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

        # 微信推送
        lines = [f"{s['股票']} {s['策略']} {s['信号']}@{s['收盘价']}" for s in signals]
        push_wechat(f"量化信号 {today}: {len(signals)}条", "\n\n".join(lines))

        print("\n⚠️ 以上为策略信号，仅供参考。请人工核实后自行决定是否下单。")
    else:
        print("📭 今日所有股票所有策略均无新信号")


if __name__ == "__main__":
    main()
