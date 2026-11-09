"""模拟盘自动交易（仅虚拟记账，无任何下单能力）

本模块不连接券商、不发送委托、不包含任何下单函数。
它只做三件事：按规则判断、按最新收盘价虚拟成交、把结果写进本地 JSON 账本。

每日流程（15:40 由计划任务触发，也可在页面手动点一次）：
1. 扫描指数成分股，得到「≥2 个策略共振 + 走势健康」的候选
2. 先卖：
   - 现价 ≤ 止损价  → 止损
   - 现价 ≥ 目标价  → 止盈
   - 不再出现在今日候选（策略转空）→ 离场
3. 后买：按 共振数 → 距年线 排序，最多持 5 只，单只 ≤10% 总资产，总仓位 ≤60%
4. 按最新收盘价虚拟成交，扣佣金（万3，最低 5 元）与印花税（卖出万5）
5. 记录持仓、成交明细、每日总资产
"""
import datetime
import json

import pandas as pd

from config import OUTPUT_DIR, COMMISSION_RATE, STAMP_TAX_RATE
from core.screener import screen, INDEX_POOLS
from data.fetcher import fetch_daily, handle_missing

STATE_DIR = OUTPUT_DIR / "paper"
STATE_FILE = STATE_DIR / "state.json"

MAX_POSITIONS = 5          # 最多同时持有几只
MAX_TOTAL_POSITION = 0.60  # 总仓位上限
MAX_SINGLE = 0.10          # 单只上限
MIN_COMMISSION = 5.0       # 最低佣金
LOOKBACK_DAYS = 400


def load_state(capital):
    """读取虚拟账户；不存在则按本金开户"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("positions", {})
        state.setdefault("trades", [])
        state.setdefault("equity", [])
        return state

    return {
        "capital": float(capital),
        "cash": float(capital),
        "positions": {},
        "trades": [],
        "equity": [],
        "created": datetime.date.today().strftime("%Y-%m-%d"),
    }


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _buy_fee(amount):
    return max(MIN_COMMISSION, amount * COMMISSION_RATE)


def _sell_fee(amount):
    return max(MIN_COMMISSION, amount * COMMISSION_RATE) + amount * STAMP_TAX_RATE


def _last_close(code):
    """取最近一根收盘价；失败返回 None（宁可不动作，也不乱记账）"""
    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() -
             datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    try:
        df = handle_missing(fetch_daily(code, start, end, source="akshare"))
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def run_once(pool_code="000300", capital=100000, progress_cb=None):
    """跑一次完整流程：卖出 → 买入 → 记账（全部为虚拟成交）"""
    state = load_state(capital)
    today = datetime.date.today().strftime("%Y-%m-%d")
    trades = []

    res = screen(pool_code, progress_cb=progress_cb, capital=state["capital"])
    candidates = res["result"]
    cand_map = ({str(r["代码"]): r for _, r in candidates.iterrows()}
                if len(candidates) else {})
    # 候选为空可能是全市场无信号，也可能是数据源抖动——后者不应误杀持仓
    can_rebalance = len(candidates) > 0

    # ---------- 1. 卖出 ----------
    for code in list(state["positions"].keys()):
        pos = state["positions"][code]
        price = _last_close(code)
        if price is None:
            continue
        pos["last_price"] = round(price, 2)
        pos["last_date"] = today

        reason = None
        if price <= pos["stop"]:
            reason = f"跌破止损价 {pos['stop']:.2f}"
        elif price >= pos["target"]:
            reason = f"到达目标价 {pos['target']:.2f}"
        elif can_rebalance and code not in cand_map:
            reason = "策略转空，不再入选"

        if not reason:
            continue

        amount = pos["shares"] * price
        fee = _sell_fee(amount)
        state["cash"] += amount - fee
        trades.append({
            "日期": today, "代码": code, "名称": pos["name"], "方向": "卖出",
            "股数": pos["shares"], "价格": round(price, 2),
            "金额": round(amount, 2), "费用": round(fee, 2),
            "原因": reason,
            "盈亏": round(amount - fee - pos["shares"] * pos["cost"], 2),
        })
        del state["positions"][code]

    # ---------- 2. 买入 ----------
    def _market_value():
        return sum(p["shares"] * p.get("last_price", p["buy_price"])
                   for p in state["positions"].values())

    slots = MAX_POSITIONS - len(state["positions"])
    for _, row in candidates.iterrows():
        if slots <= 0:
            break
        code = str(row["代码"])
        if code in state["positions"]:
            continue

        price = float(row["收盘价"])
        # 只在页面给出的「建议买入区间」内成交，避免追高或提前抄底
        try:
            zone_lo, zone_hi = [float(v) for v in
                                str(row["建议买入"]).split("~")]
        except (ValueError, AttributeError):
            continue
        if not (zone_lo <= price <= zone_hi):
            continue

        total_now = state["cash"] + _market_value()
        room = total_now * MAX_TOTAL_POSITION - _market_value()
        budget = min(total_now * MAX_SINGLE, state["cash"], room)
        lots = int(budget // (price * 100))
        if lots <= 0:
            continue

        amount = lots * 100 * price
        fee = _buy_fee(amount)
        while lots > 0 and amount + fee > state["cash"]:
            lots -= 1
            amount = lots * 100 * price
            fee = _buy_fee(amount)
        if lots <= 0:
            continue

        state["cash"] -= amount + fee
        state["positions"][code] = {
            "name": row["名称"],
            "shares": lots * 100,
            "cost": round((amount + fee) / (lots * 100), 3),
            "buy_price": round(price, 2),
            "buy_date": today,
            "stop": float(row["止损价"]),
            "target": float(row["目标价"]),
            "hits": int(row["共振数"]),
            "last_price": round(price, 2),
            "last_date": today,
        }
        trades.append({
            "日期": today, "代码": code, "名称": row["名称"], "方向": "买入",
            "股数": lots * 100, "价格": round(price, 2),
            "金额": round(amount, 2), "费用": round(fee, 2),
            "原因": f"{int(row['共振数'])}策略共振", "盈亏": 0.0,
        })
        slots -= 1

    # ---------- 3. 记账 ----------
    market_value = _market_value()
    total = state["cash"] + market_value
    today_row = {"date": today, "total": round(total, 2)}
    state["equity"] = [e for e in state["equity"] if e["date"] != today] + [today_row]
    state["equity"].sort(key=lambda e: e["date"])

    state["trades"].extend(trades)
    state["trades"] = state["trades"][-200:]
    save_state(state)

    message = res["message"] or (
        f"扫描 {res['scanned']} 只，{len(candidates)} 只入选；"
        f"本次虚拟成交 {len(trades)} 笔")

    return {
        "date": today,
        "pool": INDEX_POOLS.get(pool_code, pool_code),
        "cash": round(state["cash"], 2),
        "market_value": round(market_value, 2),
        "total": round(total, 2),
        "pnl_pct": total / state["capital"] - 1,
        "trades": trades,
        "positions": [{"代码": c, **p} for c, p in state["positions"].items()],
        "equity": state["equity"],
        "message": message,
        "state": state,
    }


def read_state():
    """供页面展示当前虚拟账户"""
    return load_state(0)


def positions_frame(state):
    """持仓转成表格（含浮动盈亏）"""
    rows = []
    for code, p in state.get("positions", {}).items():
        price = p.get("last_price", p["buy_price"])
        mv = p["shares"] * price
        rows.append({
            "代码": code,
            "名称": p["name"],
            "股数": p["shares"],
            "成本价": p["cost"],
            "现价": price,
            "市值": round(mv, 2),
            "浮动盈亏": round(mv - p["shares"] * p["cost"], 2),
            "浮盈%": round(price / p["cost"] - 1, 4) if p["cost"] else 0.0,
            "止损价": p["stop"],
            "目标价": p["target"],
            "买入日": p["buy_date"],
        })
    if not rows:
        return pd.DataFrame(columns=["代码", "名称", "股数", "成本价", "现价",
                                     "市值", "浮动盈亏", "浮盈%",
                                     "止损价", "目标价", "买入日"])
    return pd.DataFrame(rows)


def trades_frame(state, limit=30):
    """最近成交"""
    trades = state.get("trades", [])[-limit:][::-1]
    if not trades:
        return pd.DataFrame(columns=["日期", "代码", "名称", "方向", "股数",
                                     "价格", "金额", "费用", "原因", "盈亏"])
    return pd.DataFrame(trades)
