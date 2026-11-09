"""
数据清洗与处理

教程第 6 章核心内容：
- 缺失值处理（停牌 ffill）
- 异常值检测（3σ 原则）
- 频率转换（resample 日线→周线/月线）
- 多股日期对齐
"""
import numpy as np
import pandas as pd


def handle_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    处理缺失值

    金融时间序列首选 ffill（前向填充）：停牌期间价格不变最合理
    严禁用 bfill（后向填充），属于未来函数
    """
    df = df.copy()
    if method == "ffill":
        df = df.ffill()
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"未知方法: {method}")

    # 如果开头仍有 NaN（上市初期或数据源问题），用第一行有效值填充
    df = df.bfill().ffill()
    return df


def detect_outliers(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """
    3σ 原则检测异常值（返回布尔序列，True 为异常）

    注意：异常值≠错误！涨跌停、政策驱动暴涨等都是真实行情
    本函数只负责"发现"，不负责"删除"
    """
    ret = series.pct_change().dropna()
    mu, sd = ret.mean(), ret.std()
    is_outlier = (ret - mu).abs() > n_sigma * sd

    # 把结果映射回原始 series 的索引
    full = pd.Series(False, index=series.index)
    full.loc[is_outlier.index] = is_outlier.values
    return full


def resample_freq(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    重采样到更低频率

    参数:
        freq: "W"=周线（默认周五）, "ME"=月线（月末）, "QE"=季线
    """
    rule = "W-FRI" if freq.upper() == "W" else freq
    resampled = {}

    # 不同列用不同聚合规则
    resampled["Open"] = df["Open"].resample(rule).first()
    resampled["High"] = df["High"].resample(rule).max()
    resampled["Low"] = df["Low"].resample(rule).min()
    resampled["Close"] = df["Close"].resample(rule).last()
    resampled["Volume"] = df["Volume"].resample(rule).sum()

    result = pd.DataFrame(resampled)
    result = result.dropna(how="all")  # 去掉全 NaN 行
    return result


def align_multi_stock(prices: pd.DataFrame) -> pd.DataFrame:
    """
    多只股票日期对齐：先对齐索引，再统一 ffill + dropna

    教程强调的顺序：先对齐 → 再填充 → 再计算
    """
    return prices.ffill().dropna()
