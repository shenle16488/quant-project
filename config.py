"""全局配置"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"

DEFAULT_DATA_SOURCE = "akshare"

STOCK_POOL = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "601398": "工商银行",
    "600036": "招商银行",
    "600276": "恒瑞医药",
    "300750": "宁德时代",
    "002594": "比亚迪",
    "000333": "美的集团",
    "000651": "格力电器",
    "601668": "中国建筑",
}

DEFAULT_START_DATE = "20190101"
DEFAULT_END_DATE = "20241231"

STRATEGY_PARAMS = {
    "dual_ma": {"fast": 50, "slow": 200},
    "turtle": {"entry": 20, "exit": 10},
    "bollinger": {"period": 20, "std_dev": 2},
}

RISK_FREE_RATE = 0.03
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005

PAGE_TITLE = "量化策略回测看板"
PAGE_ICON = "📈"

for d in [OUTPUT_DIR, CHART_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
