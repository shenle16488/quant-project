"""
策略基类

所有策略都继承这个基类，实现 generate_signal(close, **params) 方法
返回一个 pandas Series，值为 1=满仓，0=空仓（可扩展 -1 做空）
"""
from abc import ABC, abstractmethod
import pandas as pd


class StrategyBase(ABC):
    """策略基类，所有策略必须实现 generate_signal"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称，用于 UI 显示"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """策略简述"""
        pass

    @abstractmethod
    def generate_signal(self, close: pd.Series, **params) -> pd.Series:
        """
        根据收盘价生成持仓信号

        参数:
            close: 收盘价 Series（日期索引）
            **params: 策略特定参数（如均线周期）

        返回:
            signal Series，同索引，值为 1（持仓）或 0（空仓）
        """
        pass

    def generate_positions(self, close: pd.Series, **params) -> pd.Series:
        """
        生成开平仓信号（与 generate_signal 的区别）：
        - generate_signal: 每个时点该持多少仓（1/0）
        - generate_positions: 每个时点发生了什么动作（diff 结果）
          +2 金叉买入，-2 死叉卖出（仅用于可视化标记）
        """
        signal = self.generate_signal(close, **params)
        # 把 0/1 信号转成 -1/1（持仓状态）
        position = signal.replace({0: -1})
        return position.diff()
