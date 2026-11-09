"""Streamlit 量化策略回测看板

运行方式: streamlit run app.py
"""
import sys
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from config import (
    STOCK_POOL, DEFAULT_START_DATE, DEFAULT_END_DATE,
    STRATEGY_PARAMS, PAGE_TITLE, PAGE_ICON,
)
from data.fetcher import fetch_daily, handle_missing
from data.stock_list import get_stock_list
from core.strategies import get_strategy, list_strategies
from core.screener import screen, format_for_display, order_plan, INDEX_POOLS
from core.paper_trader import (
    run_once as paper_run, read_state, positions_frame, trades_frame,
)
from core.holdings_review import review_all
from backtest.vectorized import backtest_from_signal
from backtest.metrics import format_metrics


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON,
                   layout="wide", initial_sidebar_state="expanded")


# ========== 错误翻译 + 输入校验 ==========

def explain_error(e, symbol, start, end):
    """把底层异常翻译成用户能看懂的原因，并给出针对性排查步骤"""
    name = type(e).__name__
    msg = str(e)

    # 1. 文件权限（缓存写入被拒）
    if isinstance(e, PermissionError) or "Errno 13" in msg or "Permission denied" in msg:
        return ("缓存文件写入被拒绝", [
            "这不是行情数据的问题，而是写 `output/cache` 目录没有权限",
            "关掉杀毒软件的实时防护后再试（常见于 360/火绒锁定 CSV）",
            "确认启动 Streamlit 的终端是普通命令行，而非受限环境",
            "临时方案：把项目复制到 `C:\\Users\\你的用户名\\quant-project` 下运行",
        ])

    # 2. 网络中断 / 限流
    net_keys = ("RemoteDisconnected", "ConnectionError", "ConnectionReset",
                "Timeout", "timed out", "urlopen error", "ChunkedEncoding",
                "IncompleteRead", "Max retries", "SSLError", "ProxyError")
    if any(k in name or k in msg for k in net_keys):
        return ("网络请求被中断（数据源限流或连接被重置）", [
            "等 1~3 分钟再试，数据源有短时频率限制，会自动恢复",
            "确认浏览器能正常打开 finance.sina.com.cn",
            "如果开着代理或 VPN，先关掉——代理会导致连接被对端重置",
            "频繁连续点击「运行回测」也会触发限流，间隔几秒再点",
        ])

    # 3. 区间内无数据
    if "区间内无数据" in msg:
        return (f"{symbol} 在 {start} ~ {end} 这个区间查不到交易日数据", [
            f"确认该股票在 {start[:4]} 年时已经上市（次新股上市前的区间是空的）",
            "把起始日期往后调，比如换成该股票上市后的年份",
            "确认结束日期没早于起始日期",
        ])

    # 4. 代码错误 / 空数据
    if "空数据" in msg or "empty" in msg.lower() or "未找到" in msg:
        return (f"股票代码 {symbol} 没有取到任何数据", [
            "核对是否为 6 位纯数字 A 股代码（600519 而不是 600519.SS）",
            "本接口只覆盖沪深京 A 股，港股（00700）、美股（AAPL）、基金（510300）取不到",
            "已退市的股票同样取不到数据",
        ])

    # 5. 全数据源失败
    if "数据获取失败" in msg:
        return ("新浪与东方财富两个数据源都请求失败", [
            "最常见原因是数据源短时限流，等 2 分钟再试",
            "确认网络可访问 finance.sina.com.cn 和 push2his.eastmoney.com",
            "换一只已缓存的股票（如 600519）先确认程序本身正常",
        ])

    # 6. 兜底
    return (f"未预期的错误：{name}", [msg])


def validate_dates(start, end):
    """返回错误说明字符串；None 表示校验通过"""
    for label, v in [("起始", start), ("结束", end)]:
        s = str(v).strip()
        if not (s.isdigit() and len(s) == 8):
            return f"{label}日期格式不对：应为 8 位数字 YYYYMMDD，当前是「{v}」"
        year = int(s[:4])
        if year < 1990 or year > 2100:
            return f"{label}日期年份不合理：{s[:4]}"
    if str(start).strip() > str(end).strip():
        return f"起始日期 {start} 晚于结束日期 {end}，请调换"
    return None


def render_screen_result(res, pool_label):
    """渲染选股结果（含买入区间 / 止损 / 目标价 / 建议股数）"""
    if res["message"]:
        st.warning(f"⚠️ {res['message']}")
        return

    df = res["result"]
    st.success(f"✅ 扫描 {res['scanned']} 只 {pool_label} 成分股，"
               f"{len(df)} 只通过「多策略共振 + 走势健康」筛选")

    action_cols = ["代码", "名称", "昨开", "昨收", "今开", "跳空%", "收盘价",
                   "建议买入", "止损价", "目标价", "建议股数", "操作建议"]
    st.dataframe(format_for_display(df[action_cols]), use_container_width=True,
                 hide_index=True, height=460)

    detail_cols = ["代码", "名称", "共振数", "共振策略", "距半年高点",
                   "20日涨幅", "年化波动", "距年线", "近1年最大回撤"]
    with st.expander("📊 共振策略与风险明细", expanded=False):
        st.dataframe(format_for_display(df[detail_cols]), use_container_width=True,
                     hide_index=True, height=460)

    st.download_button(
        "⬇️ 下载 CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"screen_{datetime.date.today():%Y%m%d}.csv",
        mime="text/csv",
    )

    with st.expander("📖 价位是怎么算出来的（点开看规则）", expanded=False):
        st.markdown("""
| 列 | 规则 | 怎么看 |
|---|---|---|
| **建议买入** | 收盘价 ±1.5% | 在这个区间内的价格算同一个买点，不必为几分钱错过 |
| **止损价** | 近20日最低价 -1% 与 收盘价 -8% 中较近的一个，最紧不超过 -3% | 跌破就离场，先保本金 |
| **目标价** | 收盘价 + 2 ×（收盘价 - 止损价），且不越过近半年最高价 | 2:1 盈亏比，到压力位先减仓 |
| **建议股数** | 本金 ×10% ÷ 价格，按 100 股一手取整 | 单只不超过总资金 10%，避免一次亏光 |
| **操作建议** | 高于20日线 8% 视为追高 → 等回踩；低于20日线 → 等站上 | 一句话操作提示 |
| **昨开 / 昨收 / 今开** | 最近三根日K的开盘、收盘、开盘价 | 对比昨收和今开，看当天有没有跳空 |
| **跳空%** | 今开 ÷ 昨收 - 1 | 正数是高开，追进去容易买在日内高点；负数是低开，可能直接跌破买点 |
| **距半年高点** | 现价 / 近半年最高 - 1 | 越接近 0 说明越在高位，追高风险越大 |
""")

    st.caption("「共振策略(天数)」= 哪些策略看多 + 已持有几个交易日。天数越小，信号越新鲜。")
    st.caption("「昨开/昨收/今开」取的是数据里最近三根日K。收盘后跑就是当天的真实值，"
               "盘中或盘前跑则「今开」是最新一根未走完的K线。")
    st.warning("以上价位是按固定规则机械计算，不是预测。历史信号不代表未来，"
               "建议单只不超过总资金 10%，跌破止损价无条件离场。")


def render_paper_account(state, message=None):
    """渲染模拟账户：总资产、持仓、成交、资金曲线"""
    pos = state.get("positions", {})
    mv = sum(p["shares"] * p.get("last_price", p["buy_price"]) for p in pos.values())
    total = state.get("cash", 0) + mv
    capital = state.get("capital", 0) or 1

    if message:
        st.caption(message)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总资产（虚拟）", f"{total:,.0f}", f"{total / capital - 1:+.2%}")
    c2.metric("现金", f"{state.get('cash', 0):,.0f}")
    c3.metric("持仓市值", f"{mv:,.0f}")
    c4.metric("持仓数量", f"{len(pos)} 只")

    st.subheader("📌 当前持仓")
    pf = positions_frame(state)
    if pf.empty:
        st.info("当前空仓——没有任何股票满足「≥2 策略共振 + 走势健康」。")
    else:
        show = pf.copy()
        show["浮盈%"] = (show["浮盈%"] * 100).round(2).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)

    eq = state.get("equity", [])
    if len(eq) >= 2:
        st.subheader("📈 模拟账户资金曲线")
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot([e["date"] for e in eq], [e["total"] for e in eq],
                color="#14805e", linewidth=1.8, label="总资产")
        ax.axhline(capital, color="#8a8f98", linewidth=1, linestyle="--",
                   label=f"本金 {capital:,.0f}")
        ax.set_ylabel("元"); ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig, clear_figure=True)

    st.subheader("🧾 最近成交")
    tf = trades_frame(state)
    if tf.empty:
        st.caption("还没有成交记录")
    else:
        st.dataframe(tf, use_container_width=True, hide_index=True)

    st.warning("这是模拟盘：资金与成交全是虚拟的，没有向任何券商发送委托。"
               "A 股散户没有官方下单 API，真金白银的全自动下单需要券商 QMT/Ptrade 权限。")


def render_order_plan(plan, capital):
    """渲染今日操作单：今天买什么、什么价买、买多少、什么价卖"""
    if plan.empty:
        st.warning("⚠️ 今天没有满足条件的标的，建议空仓等待")
        return

    buy = plan[plan["可买"]].copy()
    wait = plan[~plan["可买"]].copy()

    if buy.empty:
        st.warning("⚠️ 今天没有可直接买的标的（或价格未到买点），下面列出等待中的条件单")
    else:
        total = buy["金额"].sum()
        st.success(f"✅ 今日可挂单 {len(buy)} 只，合计 {total:,.0f} 元，"
                   f"占本金 {total / capital:.1%}，剩余现金 {capital - total:,.0f} 元")

        buy_show = buy[["代码", "名称", "买入价", "股数", "金额",
                        "止损价", "目标价", "盈亏比", "共振数"]].copy()
        buy_show["金额"] = buy_show["金额"].round(0).astype(int)
        st.dataframe(buy_show, use_container_width=True, hide_index=True)

        st.caption("卖出规则：① 到目标价止盈　② 跌破止损价无条件卖　"
                   "③ 每天 15:40 自动扫描，掉出候选（不再有 2 个策略看多）就卖。"
                   "A 股 T+1，今天买的最快明天才能卖。")

    if not wait.empty:
        st.subheader("⏳ 等待买点的条件单")
        wait_show = format_for_display(
            wait[["代码", "名称", "现价", "买入区间", "买入时机", "距买点",
                  "止损价", "目标价", "盈亏比", "备注"]])
        st.dataframe(wait_show, use_container_width=True, hide_index=True)
        st.caption("「距买点」= 现价离买点还差多少。正数表示还要跌下来，负数表示还要涨上去。")

    state = read_state()
    pf = positions_frame(state)
    if not pf.empty:
        st.subheader("📤 现有模拟持仓的卖出价位")
        show = pf[["代码", "名称", "股数", "成本价", "现价", "浮盈%",
                   "止损价", "目标价"]].copy()
        show["浮盈%"] = (show["浮盈%"] * 100).round(2).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ 下载今日操作单 CSV",
        plan.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"orders_{datetime.date.today():%Y%m%d}.csv",
        mime="text/csv",
    )
    st.warning("操作单是按固定规则算出的机械清单，不是收益预测。"
               "目标价未必会到，止损价必须执行。")


def render_review(rdf):
    """渲染持仓体检结论"""
    if rdf.empty:
        st.info("当前空仓，没有需要体检的持仓。")
        return

    show = rdf.copy()
    for col in ["浮盈%", "20日涨幅", "距年线"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    st.dataframe(
        show[["代码", "名称", "股数", "现价", "浮盈%", "20日涨幅",
              "距年线", "共振数", "结论", "建议动作"]],
        use_container_width=True, hide_index=True)

    counts = rdf["结论"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("该卖出", int(counts.get("卖出", 0)))
    c2.metric("该减仓", int(counts.get("减仓", 0)))
    c3.metric("可继续持有", int(counts.get("持有", 0)))

    for _, r in rdf.iterrows():
        show_fn = {"卖出": st.error, "减仓": st.warning, "持有": st.success}[r["结论"]]
        show_fn(f"{r['代码']} {r['名称']}：{r['结论']}（{r['建议动作']}）— {r['理由']}")

    st.download_button(
        "⬇️ 下载体检结果 CSV",
        rdf.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"review_{datetime.date.today():%Y%m%d}.csv",
        mime="text/csv",
    )
    st.caption("判定顺序：硬止损 → 止盈 → 策略转空 → 跌破 20 日线且 20 日跌幅超 8% → "
               "距年线低于 -5% → 单日跌超 6%。任一触发就执行，不主观改价。")


# ========== 券商条件单 ==========

BROKER_HOWTO = {
    "同花顺": (
        "同花顺 App 首页点 **交易 → 持仓 → 长按某只股票 → 设置止盈止损**，"
        "止损价填我给你的数值，委托方式选 **市价**；止盈价填我给你的数值，"
        "委托方式选 **限价**。有效期设 **长期有效**，触发一边自动撤另一边。"
    ),
    "东方财富": (
        "东方财富 App 底部 **交易 → 持仓 → 某只股票右侧「条件单」按钮**，"
        "先挂止损条件单（最新价 ≤ X → 市价卖出全部），再 **开启止盈止损联动**，"
        "联动里填止盈条件（最新价 ≥ X → 限价卖出全部）。有效期设 **90 天**。"
    ),
}


def render_condition_orders(state, broker="同花顺"):
    """把模拟盘持仓转成券商条件单清单（止损市价、止盈限价，盈亏比标注）"""
    pos = state.get("positions", {})
    if not pos:
        st.info("当前空仓，没有需要挂条件单的持仓。先点「🤖 立即自动交易一次」建仓。")
        return

    rows = []
    for code, p in pos.items():
        buy = p.get("buy_price", p["cost"])
        stop, target = float(p["stop"]), float(p["target"])
        risk = buy - stop
        ratio = (target - buy) / risk if risk > 0 else 0.0
        rows.append({
            "代码": code, "名称": p["name"],
            "持仓股数": p["shares"],
            "买入价": buy,
            "止损触发条件": f"最新价 ≤ {stop:.2f}",
            "止损委托方式": f"市价卖 {p['shares']} 股",
            "止盈触发条件": f"最新价 ≥ {target:.2f}",
            "止盈委托方式": f"限价卖 {p['shares']} 股 @ {target:.2f}",
            "盈亏比": f"{ratio:.1f}:1",
            "建议": "照挂" if ratio >= 1.5 else "盈亏比偏低，建议减仓或跳过",
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    # 股数修正：让用户把模拟盘股数改成自己真实持仓数
    st.subheader("📝 券商条件单清单")
    st.caption("股数默认取模拟盘持仓，改成你**真实账户里的持仓股数**（必须精确，多挂会因融券不够被拒）")

    with st.expander("✏️ 修正为真实持仓股数", expanded=False):
        real_rows = []
        for r in rows:
            real_shares = st.number_input(
                f"{r['代码']} {r['名称']}",
                min_value=0, max_value=1000000, step=100,
                value=r["持仓股数"], key=f"real_{r['代码']}")
            real_rows.append({
                **r,
                "止损委托方式": f"市价卖 {real_shares} 股",
                "止盈委托方式": f"限价卖 {real_shares} 股 @ {float(r['止盈触发条件'].split('≥')[1].strip())}",
            })
        df = pd.DataFrame(real_rows)

    st.dataframe(
        df[["代码", "名称", "持仓股数", "买入价",
            "止损触发条件", "止损委托方式",
            "止盈触发条件", "止盈委托方式",
            "盈亏比", "建议"]],
        use_container_width=True, hide_index=True, height=300)

    st.download_button(
        f"⬇️ 导出 CSV（含股数、止损价、止盈价）",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"condition_orders_{datetime.date.today():%Y%m%d}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader(f"📱 {broker} 操作指引")
    st.info(BROKER_HOWTO.get(broker, BROKER_HOWTO["同花顺"]))

    st.subheader("⚠️ 三条纪律（别违反）")
    st.error("""
    1. **止损委托必须选「市价」**。委托方如果用限价，跌停板上你挂的价根本不会成交。
    2. **止盈委托必须选「限价」**。用市价会被瞬间砸盘，到手的利润剩一半。
    3. **不要改条件单里的价格**。止损和止盈是成套算的，改一边盈亏比就坏了。
    """)


# ========== 侧边栏 ==========
with st.sidebar:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("基于 runoob.com Python 量化教程")
    st.divider()

    pick_mode = st.radio("选股方式", ["热门精选", "全市场搜索", "自定义代码"],
                         horizontal=False)

    if pick_mode == "热门精选":
        stock_options = {f"{code} · {name}": (code, name) for code, name in STOCK_POOL.items()}
        stock_label = st.selectbox("📌 股票", list(stock_options.keys()), index=0)
        symbol, stock_name = stock_options[stock_label]

    elif pick_mode == "全市场搜索":
        try:
            all_df = get_stock_list()
            choices = [f"{r.code} · {r.name}" for r in all_df.itertuples()]
            picked = st.selectbox(
                "📌 股票（可输入代码或名称搜索）", choices, index=None,
                placeholder="输入 519 搜茅台，或输入 600519",
                help="沪深京全部 A 股 5500+ 只，在框内输入代码或中文名即可筛选",
            )
            if picked:
                symbol, stock_name = picked.split(" · ")
            else:
                symbol, stock_name = "600519", "贵州茅台"
        except Exception as e:
            st.warning(f"全市场列表加载失败，改用热门池：{e}")
            stock_options = {f"{code} · {name}": (code, name) for code, name in STOCK_POOL.items()}
            stock_label = st.selectbox("📌 股票", list(stock_options.keys()), index=0)
            symbol, stock_name = stock_options[stock_label]

    else:
        symbol = st.text_input("股票代码（6 位数字）", "600519").strip()
        stock_name = ""
        if symbol:
            try:
                lst = get_stock_list()
                hit = lst[lst["code"] == symbol.zfill(6)]
                stock_name = hit.iloc[0]["name"] if not hit.empty else ""
            except Exception:
                pass
        st.caption(f"识别为：{stock_name or '未知代码（仍会尝试拉取）'}")
        if not symbol:
            symbol = "600519"

    stock_label = f"{symbol} {stock_name}".strip()

    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.text_input("起始 (YYYYMMDD)", DEFAULT_START_DATE)
    end_date = col_d2.text_input("结束 (YYYYMMDD)", DEFAULT_END_DATE)

    source = st.radio("🔗 数据源", ["akshare", "yfinance"], index=0,
                      help="大陆用户推荐 akshare（不限流）")

    st.subheader("⚙️ 策略选择")
    strat_info = list_strategies()
    strat_labels = [s["name"] for s in strat_info]
    strat_key_map = {s["name"]: s["key"] for s in strat_info}
    selected = st.selectbox("策略", strat_labels)
    strat_key = strat_key_map[selected]
    strategy_params = STRATEGY_PARAMS.get(strat_key, {})

    st.subheader("🎛️ 参数")
    if strat_key == "dual_ma":
        fast = st.slider("快线 fast", 5, 120, strategy_params.get("fast", 50), step=5)
        slow = st.slider("慢线 slow", 20, 300, strategy_params.get("slow", 200), step=10)
        params = {"fast": fast, "slow": slow}
    elif strat_key == "turtle":
        entry = st.slider("入场 entry", 5, 60, strategy_params.get("entry", 20), step=1)
        exit_ = st.slider("离场 exit", 3, 30, strategy_params.get("exit", 10), step=1)
        params = {"entry": entry, "exit_": exit_}
    elif strat_key == "bollinger":
        period = st.slider("周期 period", 10, 60, strategy_params.get("period", 20), step=2)
        std_dev = st.slider("标准差 std_dev", 1.0, 3.0, float(strategy_params.get("std_dev", 2.0)), step=0.1)
        params = {"period": period, "std_dev": std_dev}
    else:
        params = {}

    st.divider()
    run_btn = st.button("▶️ 运行回测", type="primary", use_container_width=True)

    st.divider()
    st.subheader("🔍 一键选股")
    st.caption("扫指数成分股，只挑「多个策略同时看多 + 走势没崩」的")
    scan_pool_label = st.selectbox("选股范围", list(INDEX_POOLS.values()),
                                   index=0, key="scan_pool")
    scan_capital = st.number_input("本金（元）", min_value=10000, max_value=10000000,
                                   value=100000, step=10000,
                                   help="用于算「建议股数」，单只最多占总资金 10%")
    scan_btn = st.button("🔍 判断现在该买哪些", type="secondary",
                         use_container_width=True)
    order_btn = st.button("📋 生成今日操作单", type="primary",
                          use_container_width=True,
                          help="直接给出：买哪只、什么价买、买多少股、什么价卖")

    st.divider()
    st.subheader("🤖 自动交易（模拟盘）")
    st.caption("每天 15:40 自动判断买卖，只记虚拟成交，不接券商、不下真单")
    paper_btn = st.button("🤖 立即自动交易一次", type="secondary",
                          use_container_width=True)
    show_paper = st.checkbox("查看当前模拟账户", value=False)
    review_btn = st.button("🩺 持仓体检（该不该继续拿）", type="secondary",
                           use_container_width=True)

    st.divider()
    st.subheader("📝 券商条件单")
    st.caption("把模拟盘止盈止损价同步到券商 App，一次挂好长期有效")
    broker = st.selectbox("你用的券商 App", ["同花顺", "东方财富"], key="broker")
    broker_btn = st.button("📝 生成券商条件单清单", type="secondary",
                           use_container_width=True)


# ========== 主区域 ==========
st.title(f"{PAGE_ICON} {PAGE_TITLE}")

pool_code = {v: k for k, v in INDEX_POOLS.items()}[scan_pool_label]

if scan_btn:
    st.subheader(f"🔍 扫描 {scan_pool_label}")
    prog = st.progress(0.0)
    status = st.empty()
    status.caption("准备中，首次扫描需要下载成分股数据，约 1~3 分钟")

    def _on_progress(done, total):
        prog.progress(done / total)
        status.caption(f"已扫描 {done}/{total}")

    try:
        res = screen(pool_code, progress_cb=_on_progress, capital=scan_capital)
        st.session_state["scan_data"] = res
        st.session_state["scan_pool_label"] = scan_pool_label
    except Exception as e:
        prog.empty()
        status.empty()
        title, tips = explain_error(e, "指数成分股", "", "")
        st.error(f"❌ {title}")
        st.markdown("**排查步骤：**")
        for i, tip in enumerate(tips, 1):
            st.markdown(f"{i}. {tip}")
        with st.expander("🔧 原始异常信息（反馈问题时可复制）", expanded=False):
            st.code(f"{type(e).__name__}: {e}")
        st.stop()
    prog.empty()
    status.empty()

if order_btn:
    cached = st.session_state.get("scan_data")
    same_pool = st.session_state.get("scan_pool_label") == scan_pool_label
    if cached is None or not same_pool:
        o_prog = st.progress(0.0)
        o_status = st.empty()
        o_status.caption(f"正在扫描 {scan_pool_label} 成分股，约 1~3 分钟")

        def _o_progress(done, total):
            o_prog.progress(done / total)
            o_status.caption(f"已扫描 {done}/{total}")

        try:
            cached = screen(pool_code, progress_cb=_o_progress, capital=scan_capital)
            st.session_state["scan_data"] = cached
            st.session_state["scan_pool_label"] = scan_pool_label
        except Exception as e:
            o_prog.empty()
            o_status.empty()
            title, tips = explain_error(e, "指数成分股", "", "")
            st.error(f"❌ {title}")
            st.markdown("**排查步骤：**")
            for i, tip in enumerate(tips, 1):
                st.markdown(f"{i}. {tip}")
            with st.expander("🔧 原始异常信息（反馈问题时可复制）", expanded=False):
                st.code(f"{type(e).__name__}: {e}")
            st.stop()
        o_prog.empty()
        o_status.empty()

    st.subheader(f"📋 今日操作单 · {scan_pool_label}")
    if cached["message"]:
        st.warning(f"⚠️ {cached['message']}")
    else:
        render_order_plan(order_plan(cached["result"], scan_capital), scan_capital)

if paper_btn:
    st.subheader("🤖 模拟盘自动交易")
    p_prog = st.progress(0.0)
    p_status = st.empty()
    p_status.caption(f"按 {scan_pool_label} 扫描并结合当前持仓做调仓，约 1~3 分钟")

    def _p_progress(done, total):
        p_prog.progress(done / total)
        p_status.caption(f"已扫描 {done}/{total}")

    try:
        p_res = paper_run(pool_code, capital=scan_capital, progress_cb=_p_progress)
    except Exception as e:
        p_prog.empty()
        p_status.empty()
        title, tips = explain_error(e, "指数成分股", "", "")
        st.error(f"❌ {title}")
        st.markdown("**排查步骤：**")
        for i, tip in enumerate(tips, 1):
            st.markdown(f"{i}. {tip}")
        with st.expander("🔧 原始异常信息（反馈问题时可复制）", expanded=False):
            st.code(f"{type(e).__name__}: {e}")
        st.stop()
    p_prog.empty()
    p_status.empty()

    if p_res["trades"]:
        st.success(f"本次虚拟成交 {len(p_res['trades'])} 笔（"
                   + "、".join(f"{t['方向']}{t['名称']}" for t in p_res["trades"]) + "）")
    else:
        st.info("本次没有触发任何买卖，持仓原样保持。")
    render_paper_account(p_res["state"], message=p_res["message"])

if review_btn:
    st.subheader("🩺 持仓体检")
    with st.spinner("正在逐只检查持仓..."):
        rdf = review_all(read_state())
    render_review(rdf)

if run_btn:
    st.session_state.pop("scan_data", None)
    date_err = validate_dates(start_date, end_date)
    if date_err:
        st.error(f"❌ {date_err}")
        st.stop()

    with st.spinner("🔄 获取数据并运行回测..."):
        try:
            df = fetch_daily(symbol, start_date, end_date, source=source)
            df = handle_missing(df)
            close = df["Close"]
            st.success(f"✅ {stock_label} 数据 {len(df)} 条")

            strategy = get_strategy(strat_key)
            signal = strategy.generate_signal(close, **params)
            result = backtest_from_signal(close, signal)
            st.divider()

            m = result["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("累计收益", f"{m['累计收益']:.2%}", f"买入持有 {m.get('买入持有累计收益', 0):.2%}")
            c2.metric("年化收益率", f"{m['年化收益率']:.2%}", f"买入持有 {m.get('买入持有年化', 0):.2%}")
            c3.metric("夏普比率", f"{m['夏普比率']:.3f}")
            c4.metric("最大回撤", f"{m['最大回撤']:.2%}")

            with st.expander("📋 完整绩效指标", expanded=False):
                st.dataframe(format_metrics(m), use_container_width=True, hide_index=True)

            st.subheader("📊 累计净值曲线")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(result["cumulative_return"].index, result["cumulative_return"].values,
                    label=f"{strategy.name}", linewidth=1.8, color="#c0563f")
            ax.plot(result["cumulative_benchmark"].index, result["cumulative_benchmark"].values,
                    label="买入持有", linewidth=1.2, color="#14805e", alpha=0.7)
            ax.set_title(f"{stock_label} | {strategy.name} | {params}")
            ax.set_xlabel("日期"); ax.set_ylabel("累计净值")
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig, clear_figure=True)

            st.subheader("📈 买卖点标记")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(close.index, close.values, label="收盘价", linewidth=1.0, color="#1f2933", alpha=0.8)
            pos_chg = signal.diff().fillna(0)
            buy_idx = close.index[pos_chg == 1]
            sell_idx = close.index[pos_chg == -1]
            if len(buy_idx) > 0:
                ax2.scatter(buy_idx, close.loc[buy_idx], marker="^", color="#14805e", s=100, zorder=3, label="买入")
            if len(sell_idx) > 0:
                ax2.scatter(sell_idx, close.loc[sell_idx], marker="v", color="#c0563f", s=100, zorder=3, label="卖出")
            ax2.set_title(f"买卖点（买入 {len(buy_idx)} 次 / 卖出 {len(sell_idx)} 次）")
            ax2.set_xlabel("日期"); ax2.set_ylabel("价格")
            ax2.legend(); ax2.grid(alpha=0.3)
            st.pyplot(fig2, clear_figure=True)

            with st.expander("🧾 每笔交易", expanded=False):
                trades = result["trades"]
                if not trades.empty:
                    td = trades.copy()
                    td["Date"] = td["Date"].dt.strftime("%Y-%m-%d")
                    td["Price"] = td["Price"].round(2)
                    st.dataframe(td, use_container_width=True, hide_index=True)
                else:
                    st.info("期间无交易")

        except Exception as e:
            title, tips = explain_error(e, symbol, start_date, end_date)
            st.error(f"❌ {title}")
            st.markdown("**排查步骤：**")
            for i, tip in enumerate(tips, 1):
                st.markdown(f"{i}. {tip}")
            with st.expander("🔧 原始异常信息（反馈问题时可复制）", expanded=False):
                st.code(f"{type(e).__name__}: {e}")

if broker_btn:
    st.subheader(f"📝 券商条件单清单 · {broker}")
    render_condition_orders(read_state(), broker)

elif order_btn or paper_btn or review_btn:
    pass  # 操作单 / 模拟账户 / 持仓体检已在上面渲染，这里不重复展示
elif show_paper:
    st.subheader("🤖 当前模拟账户")
    render_paper_account(read_state(),
                         message="数据为最近一次自动交易时的收盘价，重新勾选或点「立即自动交易一次」可刷新")
elif st.session_state.get("scan_data") is not None:
    render_screen_result(st.session_state["scan_data"],
                         st.session_state.get("scan_pool_label", ""))
else:
    st.info("👈 左侧选股票/策略/参数 → 点 **▶️ 运行回测**")
    st.markdown("""
    **支持的策略：**
    - **双均线策略**：金叉买入、死叉卖出
    - **海龟交易法则**：20日突破入场、10日跌破离场
    - **布林带回归**：下轨超卖买入、中轨回归卖出

    **数据来源**：默认 akshare（免费、不限流）

    **想让它自己选、自己买卖**：左侧底部点 **🤖 立即自动交易一次**，
    程序会按规则虚拟成交并记账（模拟盘，不接券商）。
    """)
    st.caption("⚠️ 仅供学习研究，不构成投资建议")
