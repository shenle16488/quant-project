"""全 A 股代码表

来源：akshare stock_info_a_code_name（沪深京全部 A 股）
缓存：本地 CSV，7 天有效（新股上市、改名不频繁，无需天天拉）
"""
import datetime
import time
import pandas as pd

from config import OUTPUT_DIR

CACHE_FILE = OUTPUT_DIR / "cache" / "stock_list.csv"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_stock_list(force_refresh=False):
    """
    返回 DataFrame(code, name)，约 5500+ 只全 A 股

    参数:
        force_refresh: 忽略缓存强制重新拉取
    """
    if CACHE_FILE.exists() and not force_refresh:
        age_days = (datetime.date.today() -
                    datetime.date.fromtimestamp(CACHE_FILE.stat().st_mtime)).days
        if age_days <= 7:
            df = pd.read_csv(CACHE_FILE, dtype={"code": str})
            df["code"] = df["code"].str.zfill(6)
            return df

    import akshare as ak

    last_err = None
    for attempt in range(3):
        try:
            df = ak.stock_info_a_code_name()
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"全市场代码表获取失败（已重试 3 次）: {last_err}")

    df.columns = ["code", "name"]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str).str.replace(" ", "", regex=False)
    df = df.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    return df
