"""
双均线策略（教程第 12 章）

思想：短期均线上穿长期均线 → 金叉买入，下穿 → 死叉卖出

参数：fast（快线周期，如 50），slow（慢线周期，如 200）
"""
import numpy as np
import pandas as pd

from .strategy_base import StrategyBase


class DualMAStrategy(StrategyBase):

    name = "双均线策略"
    description = "短期均线上穿长期均线买入（金叉），下穿卖出（死叉）"

    def generate_signal(self, close: pd.Series,
                        fast: int = 50, slow: int = 200) -> pd.Series:
        """
        生成 0/1 持仓信号

        参数:
            close: 收盘价 Series
            fast:  短期均线周期
            slow:  长期均线周期

        返回:
            signal: 1=持仓, 0=空仓
        """
        # 计算均线
        ma_fast = close.rolling(window=fast).mean()
        ma_slow = close.rolling(window=slow).mean()

        # 信号：快线在慢线上方 → 持仓
        signal = pd.Series(
            np.where(ma_fast > ma_slow, 1, 0),
            index=close.index,
        )

        # 前 slow 天均线还没算出来，强制空仓
        signal.iloc[:slow] = 0

        return signal


class TurtleStrategy(StrategyBase):
    """
    海龟交易法则（教程第 13 章）

    入场：收盘价突破过去 N 日最高价
    离场：收盘价跌破过去 M 日最低价
    """

    name = "海龟交易法则"
    description = "收盘价突破 20 日最高价入场，跌破 10 日最低价离场"

    def generate_signal(self, close: pd.Series,
                        entry: int = 20, exit_: int = 10) -> pd.Series:
        """
        参数:
            close: 收盘价 Series
            entry: 入场通道周期
            exit_: 离场通道周期
        """
        # 关键：rolling 之后必须 shift(1)，用截至"昨天"的数据
        upper = close.rolling(entry).max().shift(1)
        lower = close.rolling(exit_).min().shift(1)

        signal = pd.Series(0, index=close.index)
        pos = 0  # 当前持仓状态

        for i in range(1, len(close)):
            if pos == 0 and close.iloc[i] > upper.iloc[i]:
                pos = 1  # 突破上轨，入场
            elif pos == 1 and close.iloc[i] < lower.iloc[i]:
                pos = 0  # 跌破下轨，离场
            signal.iloc[i] = pos

        return signal


class BollingerReversionStrategy(StrategyBase):
    """
    布林带均值回归（教程第 14 章上半）

    价格跌破下轨（超卖）买入，回到中轨（均值）卖出
    """

    name = "布林带回归"
    description = "收盘价跌破布林带下轨买入，回到中轨卖出"

    def generate_signal(self, close: pd.Series,
                        period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        参数:
            close: 收盘价 Series
            period: 布林带周期
            std_dev: 标准差倍数
        """
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        lower = mid - std_dev * std  # 下轨

        signal = pd.Series(0, index=close.index)
        pos = 0

        for i in range(period, len(close)):
            if pos == 0 and close.iloc[i] < lower.iloc[i]:
                pos = 1  # 跌破下轨，博回归
            elif pos == 1 and close.iloc[i] >= mid.iloc[i]:
                pos = 0  # 回到中轨，落袋
            signal.iloc[i] = pos

        return signal


# ========== 策略注册表 ==========
# Streamlit 看板用这个表生成策略下拉菜单

STRATEGY_REGISTRY = {
    "dual_ma": DualMAStrategy(),
    "turtle": TurtleStrategy(),
    "bollinger": BollingerReversionStrategy(),
}


def get_strategy(name: str) -> StrategyBase:
    """按名字获取策略实例"""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"未知策略 {name}，可用: {available}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list:
    """列出所有可用策略的信息"""
    return [
        {"key": key, "name": strat.name, "description": strat.description}
        for key, strat in STRATEGY_REGISTRY.items()
    ]
