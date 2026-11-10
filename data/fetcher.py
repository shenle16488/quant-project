"""数据获取适配层

数据源策略（按优先级）：
1. 新浪全历史接口（akshare stock_zh_a_daily）—— 一次请求拿全部历史，不限流
2. 本地全量缓存 —— 历史数据不变，隔夜仍可复用，任意区间裁剪
3. 东方财富接口 —— 备用（按区间请求，易触发限流）

对外统一接口：fetch_daily(symbol, start, end, source, adjust)
返回 DataFrame：索引 Date，列 Open/High/Low/Close/Volume
"""
import json
import time
import random
import datetime
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd

from config import DEFAULT_DATA_SOURCE, OUTPUT_DIR

# ========== 本地全量缓存 ==========
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _sina_symbol(symbol):
    """纯数字代码 → 新浪格式 (sh600519 / sz000858)
    
    兼容输入格式：
      - sh600519 / sz000858（已带前缀） → 直接返回
      - 600519 / 000858（纯数字）     → 自动加前缀
    """
    s = str(symbol).split(".")[0]
    s_lower = s.lower()
    if s_lower.startswith("sh") or s_lower.startswith("sz"):
        return s_lower  # 已经带前缀了，直接用
    prefix = "sh" if s.startswith(("6", "9")) else "sz"
    return f"{prefix}{s}"


def _full_cache_path(symbol, adjust):
    return CACHE_DIR / f"{str(symbol).split('.')[0]}_{adjust}_all.csv"


# ========== 新浪全历史 ==========

def _fetch_all_via_sina(symbol, adjust="qfq"):
    """
    通过 akshare 的新浪接口一次性获取全部历史日线

    优点：一条请求返回上市至今全历史（如茅台 6000+ 条），不会像东财那样限流
    缓存：全量存本地，之后任意区间都从缓存裁剪
    """
    import akshare as ak

    sina_sym = _sina_symbol(symbol)
    time.sleep(random.uniform(1, 2))

    raw = ak.stock_zh_a_daily(symbol=sina_sym, adjust=adjust)
    if raw is None or raw.empty:
        raise ValueError(f"新浪返回空数据: {sina_sym}")

    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()

    rename = {"open": "Open", "high": "High", "low": "Low",
              "close": "Close", "volume": "Volume"}
    df = raw.rename(columns=rename)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _load_full_cache(symbol, adjust):
    """
    读取全量缓存

    有效期规则：历史数据不会变，所以缓存永远可用；
    只有缓存里最大日期落后今天超过 7 天时才重新拉取（补最新数据）
    """
    p = _full_cache_path(symbol, adjust)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception:
        return None
    if df.empty:
        return None
    # 缓存太旧（超过 7 天没更新）→ 视为无效，重新拉
    age = (pd.Timestamp.today().normalize() - df.index[-1]).days
    if age > 7:
        return None
    return df


def _save_full_cache(df, symbol, adjust):
    """写缓存。失败不影响回测（可能因权限或磁盘占用），只跳过缓存"""
    p = _full_cache_path(symbol, adjust)
    try:
        df.to_csv(p, encoding="utf-8-sig")
    except Exception:
        pass


# ========== 东方财富（备用，按区间请求）==========

EM_BASE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EM_UT = "7eea3edcaed734bea9cbfc24409ed989"


def _em_secid(symbol):
    s = str(symbol).split(".")[0]
    return f"1.{s}" if s.startswith(("6", "9")) else f"0.{s}"


def _fetch_via_eastmoney(symbol, start, end, adjust="qfq"):
    """备用数据源：东方财富按区间请求（容易限流）"""
    time.sleep(random.uniform(2, 5))

    fqt = {"qfq": 1, "hfq": 2}.get(adjust, 0)
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": EM_UT, "klt": "101", "fqt": str(fqt),
        "secid": _em_secid(symbol),
        "beg": str(start).replace("-", ""),
        "end": str(end).replace("-", ""),
    }
    url = EM_BASE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    klines = data.get("data", {}).get("klines", [])
    if not klines:
        raise ValueError(f"东财返回空数据: {symbol} {start}~{end}")

    rows = []
    for line in klines:
        p = line.split(",")
        rows.append({"Date": p[0], "Open": float(p[1]), "Close": float(p[2]),
                     "High": float(p[3]), "Low": float(p[4]), "Volume": float(p[5])})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()[["Open", "High", "Low", "Close", "Volume"]]


# ========== 统一入口 ==========

def _norm_date(d):
    """统一成 YYYYMMDD 字符串"""
    return str(d).replace("-", "")


def _slice(df, start, end):
    """按 YYYYMMDD 区间裁剪"""
    s = pd.to_datetime(_norm_date(start), format="%Y%m%d")
    e = pd.to_datetime(_norm_date(end), format="%Y%m%d")
    out = df.loc[(df.index >= s) & (df.index <= e)]
    if out.empty:
        raise ValueError(f"区间内无数据: {s.date()} ~ {e.date()}")
    return out


def fetch_daily(symbol, start="20190101", end="20241231",
                source=None, adjust="qfq"):
    """
    获取日线数据的统一入口

    参数:
        symbol: 股票代码（纯数字即可）
        start/end: 日期，支持 20220101 或 2022-01-01
        source: "akshare"(默认,走新浪) | "eastmoney" | "yfinance"
        adjust: qfq前复权 / hfq后复权 / ""不复权
    """
    source = source or DEFAULT_DATA_SOURCE
    sym = str(symbol).split(".")[0]

    if source in ("akshare", "sina"):
        # 1) 全量缓存优先
        full = _load_full_cache(sym, adjust)
        if full is None:
            # 2) 缓存无效 → 新浪全量拉取
            try:
                full = _fetch_all_via_sina(sym, adjust)
                _save_full_cache(full, sym, adjust)
            except Exception as e_sina:
                # 3) 新浪失败 → 东财备用
                try:
                    return _fetch_via_eastmoney(sym, start, end, adjust)
                except Exception:
                    raise RuntimeError(f"{sym} 数据获取失败。新浪: {e_sina}")
        return _slice(full, start, end)

    elif source == "eastmoney":
        return _fetch_via_eastmoney(sym, start, end, adjust)

    elif source == "yfinance":
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("未安装 yfinance，请先 pip install yfinance")
        ys = sym if sym.endswith((".SS", ".SZ")) else (
            f"{sym}.SS" if sym.startswith(("6", "9")) else f"{sym}.SZ")
        s, e = _norm_date(start), _norm_date(end)
        s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        e = f"{e[:4]}-{e[4:6]}-{e[6:8]}"
        df = yf.download(ys, start=s, end=e, auto_adjust=(adjust == "qfq"),
                         progress=False, multi_level_index=False)
        if df.empty:
            raise ValueError(f"yfinance 空数据: {ys}（可能限流）")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = df.index.tz_localize(None)
        return df.sort_index()

    else:
        raise ValueError(f"未知数据源: {source}")


def handle_missing(df):
    """缺失值处理：前向填充（停牌价格不变）+ 首尾兜底"""
    return df.copy().ffill().bfill()
