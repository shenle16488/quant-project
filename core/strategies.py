"""策略基类 + 3 个策略实现"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class StrategyBase(ABC):
    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def description(self): pass

    @abstractmethod
    def generate_signal(self, close, **params): pass


class DualMAStrategy(StrategyBase):
    name = "双均线策略"
    description = "短期均线上穿长期均线买入（金叉），下穿卖出（死叉）"

    def generate_signal(self, close, fast=50, slow=200):
        ma_fast = close.rolling(window=fast).mean()
        ma_slow = close.rolling(window=slow).mean()
        signal = pd.Series(np.where(ma_fast > ma_slow, 1, 0), index=close.index)
        signal.iloc[:slow] = 0
        return signal


class TurtleStrategy(StrategyBase):
    name = "海龟交易法则"
    description = "收盘价突破 20 日最高价入场，跌破 10 日最低价离场"

    def generate_signal(self, close, entry=20, exit_=10):
        upper = close.rolling(entry).max().shift(1)
        lower = close.rolling(exit_).min().shift(1)
        signal = pd.Series(0, index=close.index)
        pos = 0
        for i in range(1, len(close)):
            if pos == 0 and close.iloc[i] > upper.iloc[i]:
                pos = 1
            elif pos == 1 and close.iloc[i] < lower.iloc[i]:
                pos = 0
            signal.iloc[i] = pos
        return signal


class BollingerReversionStrategy(StrategyBase):
    name = "布林带回归"
    description = "收盘价跌破布林带下轨买入，回到中轨卖出"

    def generate_signal(self, close, period=20, std_dev=2.0):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        lower = mid - std_dev * std
        signal = pd.Series(0, index=close.index)
        pos = 0
        for i in range(period, len(close)):
            if pos == 0 and close.iloc[i] < lower.iloc[i]:
                pos = 1
            elif pos == 1 and close.iloc[i] >= mid.iloc[i]:
                pos = 0
            signal.iloc[i] = pos
        return signal


STRATEGY_REGISTRY = {
    "dual_ma": DualMAStrategy(),
    "turtle": TurtleStrategy(),
    "bollinger": BollingerReversionStrategy(),
}


def get_strategy(name):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略 {name}，可用: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def list_strategies():
    return [
        {"key": k, "name": s.name, "description": s.description}
        for k, s in STRATEGY_REGISTRY.items()
    ]
