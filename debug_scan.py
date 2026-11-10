"""Debug scan_one step by step"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import datetime
import numpy as np
import pandas as pd
from config import STRATEGY_PARAMS
from core.strategies import get_strategy
from data.fetcher import fetch_daily, handle_missing

code = "sh600519"
start = (datetime.date.today() - datetime.timedelta(days=400)).strftime("%Y%m%d")
end = datetime.date.today().strftime("%Y%m%d")

print(f"=== Testing {code} ===")
try:
    df = handle_missing(fetch_daily(code, start, end, source="akshare"))
    print(f"✅ Got data: {len(df)} rows")
except Exception as e:
    print(f"❌ Fetch failed: {e}")
    sys.exit(1)

close = df["Close"]
print(f"   Close last: {close.iloc[-1]:.2f}")

hits = []
for key, params in STRATEGY_PARAMS.items():
    p = dict(params)
    if "exit" in p:
        p["exit_"] = p.pop("exit")
    try:
        strat = get_strategy(key)
        sig = strat.generate_signal(close, **p)
        last_sig = int(sig.iloc[-1])
        print(f"   {strat.name}: last signal = {last_sig}")
        if last_sig == 1:
            held = 0
            for v in reversed(sig.tolist()):
                if int(v) == 1:
                    held += 1
                else:
                    break
            hits.append((strat.name, held))
    except Exception as e:
        print(f"   {key} ERROR: {e}")

print(f"\nHits: {len(hits)} -> {hits}")

if not hits:
    print("❌ No hits at all - that's why all stocks are None")
    # Check if there are ANY 1 signals in recent history
    for key, params in STRATEGY_PARAMS.items():
        p = dict(params)
        if "exit" in p:
            p["exit_"] = p.pop("exit")
        strat = get_strategy(key)
        sig = strat.generate_signal(close, **p)
        recent = sig.tail(30)
        ones = (recent == 1).sum()
        print(f"   {strat.name}: last 30 days had {ones} bullish days")
else:
    print("✅ Hits found, check later steps")
    
    # Test RSI calculation
    print("\n--- RSI/ATR calc test ---")
    try:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi14 = (100 - 100 / (1 + rs)).iloc[-1]
        print(f"   RSI14 = {rsi14:.1f}")
        
        tr = pd.DataFrame({
            "hl": df["High"] - df["Low"],
            "hc": (df["High"] - close.shift(1)).abs(),
            "lc": (df["Low"] - close.shift(1)).abs(),
        }).max(axis=1)
        atr14_pct = (tr.rolling(14).mean() / close).iloc[-1]
        print(f"   ATR% = {atr14_pct*100:.2f}%")
        
        vol_col = df.get("Volume") or df.get("vol")
        if vol_col is not None:
            vol_ratio = float(vol_col.iloc[-1] / vol_col.rolling(60).mean().iloc[-1])
        else:
            vol_ratio = 1.0
        print(f"   量比 = {vol_ratio:.2f}")
    except Exception as e:
        print(f"❌ Indicator calc failed: {e}")
        import traceback; traceback.print_exc()
