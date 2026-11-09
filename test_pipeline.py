"""验证脚本 - 独立于 Streamlit，跑通完整数据流"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

print("=" * 50)
print("🚀 量化项目数据流验证")
print("=" * 50)

# 1. 数据获取
print("\n[1/4] 获取贵州茅台日线数据 (akshare)...")
from data.fetcher import fetch_daily, handle_missing
df = fetch_daily("600519", "20190101", "20241231", source="akshare")
df = handle_missing(df)
print(f"   ✅ 数据 {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"   列: {list(df.columns)}")

# 2. 策略信号
print("\n[2/4] 双均线策略 (MA50/MA200)...")
from core.strategies import get_strategy
strategy = get_strategy("dual_ma")
signal = strategy.generate_signal(df["Close"], fast=50, slow=200)
buy_days = (signal.diff() == 1).sum()
sell_days = (signal.diff() == -1).sum()
print(f"   ✅ 买入 {buy_days} 次, 卖出 {sell_days} 次")

# 3. 回测
print("\n[3/4] 向量化回测...")
from backtest.vectorized import backtest_from_signal
result = backtest_from_signal(df["Close"], signal)

# 4. 绩效
print("\n[4/4] 绩效指标...")
m = result["metrics"]
for k, v in m.items():
    if any(x in k for x in ["收益", "回撤", "波动"]):
        print(f"   {k}: {v:.2%}")
    elif "夏普" in k:
        print(f"   {k}: {v:.3f}")
    else:
        print(f"   {k}: {v}")

print("\n" + "=" * 50)
print("✅ 完整数据流验证通过！")
print("=" * 50)
