"""胜率分组诊断：找出什么条件下胜率最高"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from data.fetcher import fetch_daily
from data.cleaner import handle_missing
from core.strategies import get_strategy
from config import STRATEGY_PARAMS
from datetime import date, timedelta

# 扫描 20 只沪深300成分股
SAMPLE = ["sh600519", "sh601318", "sh600036", "sh601012", "sh601888",
          "sh600276", "sh600030", "sh601166", "sh600887", "sh600000",
          "sh601899", "sh600585", "sh601628", "sh601398", "sh601988",
          "sz000001", "sz000858", "sz000651", "sz002594", "sz300750"]

STOP = 0.08
TARGET = 0.16
HOLD = 20

end = date.today()
start = end - timedelta(days=365*3)

def outcomes(df, buy_col="Close"):
    buy = df[buy_col].values
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    out = []
    for i in range(n-1):
        bp = buy[i]
        stop_line = bp * (1 - STOP)
        target_line = bp * (1 + TARGET)
        res = "hold"
        for j in range(i+1, min(i+1+HOLD, n)):
            if lows[j] <= stop_line:
                res = "loss"
                break
            if highs[j] >= target_line:
                res = "win"
                break
        out.append(res)
    out.append("hold")
    return pd.Series(out, index=df.index)

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    tr = pd.DataFrame({
        "hl": df["High"] - df["Low"],
        "hc": (df["High"] - df["Close"].shift(1)).abs(),
        "lc": (df["Low"] - df["Close"].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()

rows = []
for sym in SAMPLE:
    try:
        df = handle_missing(fetch_daily(sym, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), source="akshare"))
    except Exception:
        continue
    if len(df) < 200:
        continue

    close = df["Close"]
    vol = df["Volume"]
    ma200 = close.rolling(200).mean()
    rsi14 = rsi(close)
    atr14 = atr(df)
    atr_pct = atr14 / close

    # 3 个策略信号
    hits = pd.Series(0, index=close.index)
    strat_names = list(STRATEGY_PARAMS.keys())
    sigs = {}
    for key, params in STRATEGY_PARAMS.items():
        p = dict(params)
        if "exit" in p:
            p["exit_"] = p.pop("exit")
        s = get_strategy(key)
        sig = s.generate_signal(close, **p)
        sigs[key] = sig
        hits += (sig == 1).astype(int)

    oc = outcomes(df, "Close")
    df2 = pd.DataFrame({
        "outcome": oc,
        "hits": hits,
        "above_ma200": close > ma200,
        "rsi": rsi14,
        "atr_pct": atr_pct,
        "vol_ratio": vol / vol.rolling(60).mean(),
        "ret20": close / close.shift(20) - 1,
    }).dropna()

    rows.append(df2)

big = pd.concat(rows)
total = len(big)

print(f"样本: {total} 根 K 线, {len(SAMPLE)} 只股票\n")

# ========== 分组 1：共振数 ==========
print("=== 分组 1：共振数 vs 胜率 ===")
for h in [0, 1, 2, 3]:
    sub = big[big["hits"] == h]
    if len(sub) < 5:
        continue
    wr = (sub["outcome"] == "win").mean()
    lr = (sub["outcome"] == "loss").mean()
    hr = (sub["outcome"] == "hold").mean()
    print(f"  hits={h}: n={len(sub):>5}  胜率={wr:.1%}  止损率={lr:.1%}  持有={hr:.1%}  期望={wr*TARGET - lr*STOP:.2%}")

# ========== 分组 2：200 日线 ==========
print("\n=== 分组 2：200 日线 + hits>=2 ===")
for label, mask in [("均线上方 & hits>=2", (big["above_ma200"]) & (big["hits"]>=2)),
                     ("均线下方 & hits>=2", (~big["above_ma200"]) & (big["hits"]>=2)),
                     ("均线上方 & hits>=3", (big["above_ma200"]) & (big["hits"]>=3))]:
    sub = big[mask]
    if len(sub) < 5:
        continue
    wr = (sub["outcome"] == "win").mean()
    lr = (sub["outcome"] == "loss").mean()
    print(f"  {label}: n={len(sub):>5}  胜率={wr:.1%}  止损率={lr:.1%}  期望={wr*TARGET - lr*STOP:.2%}")

# ========== 分组 3：RSI ==========
print("\n=== 分组 3：RSI + hits>=2 + 均线上方 ===")
base = big[(big["hits"]>=2) & (big["above_ma200"])]
for label, mask in [("RSI<30 超卖", base["rsi"]<30),
                     ("RSI 30-50", (base["rsi"]>=30) & (base["rsi"]<50)),
                     ("RSI 50-70", (base["rsi"]>=50) & (base["rsi"]<70)),
                     ("RSI>70 超买", base["rsi"]>=70)]:
    sub = base[mask]
    if len(sub) < 3:
        continue
    wr = (sub["outcome"] == "win").mean()
    print(f"  {label}: n={len(sub):>5}  胜率={wr:.1%}")

# ========== 分组 4：ATR 波动 ==========
print("\n=== 分组 4：ATR% + hits>=2 + 均线上方 + RSI 30-70 ===")
base2 = base[(base["rsi"]>=30) & (base["rsi"]<=70)]
for label, mask in [("ATR<2% 低波动", base2["atr_pct"]<0.02),
                     ("ATR 2-4%", (base2["atr_pct"]>=0.02) & (base2["atr_pct"]<0.04)),
                     ("ATR>4% 高波动", base2["atr_pct"]>=0.04)]:
    sub = base2[mask]
    if len(sub) < 3:
        continue
    wr = (sub["outcome"] == "win").mean()
    print(f"  {label}: n={len(sub):>5}  胜率={wr:.1%}")

# ========== 分组 5：成交量 ==========
print("\n=== 分组 5：量比 + hits>=2 + 均线上方 ===")
base3 = base
for label, mask in [("量比<0.8 缩量", base3["vol_ratio"]<0.8),
                     ("量比 0.8-1.5", (base3["vol_ratio"]>=0.8) & (base3["vol_ratio"]<1.5)),
                     ("量比>1.5 放量", base3["vol_ratio"]>=1.5)]:
    sub = base3[mask]
    if len(sub) < 3:
        continue
    wr = (sub["outcome"] == "win").mean()
    print(f"  {label}: n={len(sub):>5}  胜率={wr:.1%}")

# ========== 组合最优 ==========
print("\n=== 🏆 组合最优条件 ===")
best = big[(big["hits"]>=2) & (big["above_ma200"]) & (big["rsi"]>=30) & (big["rsi"]<=65) & (big["atr_pct"]<0.04)]
wr = (best["outcome"] == "win").mean()
lr = (best["outcome"] == "loss").mean()
hr = (best["outcome"] == "hold").mean()
print(f"  hits>=2 + 均线上 + RSI 30-65 + ATR<4%:")
print(f"  n={len(best):>5}  胜率={wr:.1%}  止损率={lr:.1%}  持有={hr:.1%}  期望={wr*TARGET - lr*STOP:.2%}")
