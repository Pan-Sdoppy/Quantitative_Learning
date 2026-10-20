"""
预测入口文件
============
功能：
1. 自动判断当前时间：交易日9:30-15:00自动更新最新实时数据，其余时间用训练文件的最新数据
2. 计算最新因子
3. 加载已训练的预测模型，对全市场股票预测
4. 模型预测次日收益率 pred（标签口径：T日尾盘买、T+1动态卖，训练标签用开盘0.5%/盘中4.5%以聚焦选股）
5. 先按 pred 粗筛 Top10，再做 composite 二次排序（5基础因子+K线形态+规模/量能/资金流，与回测引擎逐因子同口径）选 Top3
6. 二次排序后 Top1 的 pred<MIN_PRED(1.0%) 提示空仓；并打印 T+1 兑现提示（开盘止盈按 pred 分段 1.5%/3%，盘中 4%，不止损）
7. 打印本次预测基于的数据时间范围（综合排序 Top10 打印列：排名/代码/名称/收盘价/当天涨幅/pred/所属板块）

用法：
    python predict.py
"""
import json
import sys
import warnings
from datetime import datetime, time

import numpy as np
import pandas as pd

from core.backtest_engine import (
    KLINE_SHADOW_W, KLINE_KFLAG_W,
    RERANK_MV_W, RERANK_VOL10_W, RERANK_MF_W,
)
from core.data_loader import load_all_data, get_stock_list, get_stock_industry
from core.factors import compute_factors
from core.model import EnsembleModel, compute_confidence
from core.utils import RESULT_DIR, MODEL_DIR, get_logger

warnings.filterwarnings("ignore")
logger = get_logger("predict")

PREDICT_TAG = "latest/predict"  # 预测自动加载预测模型（100%全量数据训练）
TOP_N = 5000
HORIZONS = ["1d_close"]
MIN_STOCKS_PER_DAY = 1000  # 最新交易日最少股票数，避免盘中数据不完整
MIN_PRED = 0.010  # 信号强度过滤：综合排序Top1预测次日收益<1.0%提示空仓（与 backtest_engine/train 守卫一致，2022起夏普最高阈值）
# 分段动态开盘止盈（与backtest.py一致）：pred<1%开盘止盈1.5%、pred≥1%开盘止盈3%，盘中止盈4%
DYN_TP_BOUNDARY = 0.01
DYN_TP_LOW = 0.015
DYN_TP_HIGH = 0.03
INTRADAY_TP = 0.04


def _disp_width(s: str) -> int:
    """计算字符串的终端显示宽度（中文字符算2，英文算1）"""
    width = 0
    for ch in str(s):
        if ord(ch) > 127:
            width += 2
        else:
            width += 1
    return width


def _pad_str(s: str, width: int, align: str = "left") -> str:
    """按显示宽度填充字符串到指定宽度"""
    s = str(s)
    cur_width = _disp_width(s)
    if cur_width >= width:
        return s
    pad = width - cur_width
    if align == "left":
        return s + " " * pad
    elif align == "right":
        return " " * pad + s
    else:
        left = pad // 2
        right = pad - left
        return " " * left + s + " " * right


# 需要右对齐的数字列
NUMERIC_COLS = {"收盘价", "最新涨跌幅"}


def _format_val(col: str, val) -> str:
    """根据列名格式化值，确保数字精度正确"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if col == "收盘价":
        return f"{float(val):.2f}"
    elif col == "动态卖出置信度":
        return f"{float(val):.4f}"
    else:
        return str(val)


def print_table(df: pd.DataFrame):
    """自定义表格打印，正确处理中文字符宽度，确保列和表头对齐"""
    if df.empty:
        print("(空表)")
        return

    cols = df.columns.tolist()
    # 计算每列的最大显示宽度（使用格式化后的值）
    col_widths = []
    for col in cols:
        max_w = _disp_width(str(col))
        for val in df[col].values:
            formatted = _format_val(col, val)
            w = _disp_width(formatted)
            if w > max_w:
                max_w = w
        col_widths.append(max_w + 2)  # 加2个空格的边距

    # 判断每列对齐方式
    aligns = []
    for col in cols:
        if col in NUMERIC_COLS:
            aligns.append("right")
        else:
            aligns.append("left")

    # 打印表头
    header_parts = []
    for i, col in enumerate(cols):
        header_parts.append(_pad_str(col, col_widths[i], aligns[i]))
    print("  ".join(header_parts))

    # 打印分隔线
    sep_parts = []
    for w in col_widths:
        sep_parts.append("-" * w)
    print("--".join(sep_parts))

    # 打印数据行（使用格式化后的值）
    for _, row in df.iterrows():
        row_parts = []
        for i, col in enumerate(cols):
            val = _format_val(col, row[col])
            row_parts.append(_pad_str(val, col_widths[i], aligns[i]))
        print("  ".join(row_parts))


# ============================================================
# 盘中实时数据处理
# ============================================================
def is_trading_hours(now: datetime) -> bool:
    """判断当前是否为A股交易时间：周一到周五 9:30-15:00（节假日由数据接口自然处理）"""
    if now.weekday() >= 5:  # 周六周日
        return False
    t = now.time()
    return time(9, 30) <= t <= time(15, 0)


def _fetch_one_spot() -> pd.DataFrame:
    """单次抓取新浪全市场实时行情并清洗为标准格式（仅主板、close>0）。"""
    import akshare as ak
    spot = ak.stock_zh_a_spot()
    # 列名中文映射：代码、名称、最新价、涨跌幅、今开、最高、最低、成交量、成交额、换手率
    col_map = {}
    for c in spot.columns:
        cl = str(c)
        if "代码" in cl:
            col_map[c] = "code"
        elif "最新价" in cl or cl == "最新":
            col_map[c] = "close"
        elif "今开" in cl:
            col_map[c] = "open"
        elif "最高" in cl:
            col_map[c] = "high"
        elif "最低" in cl:
            col_map[c] = "low"
        elif "成交量" in cl:
            col_map[c] = "volume"
        elif "成交额" in cl:
            col_map[c] = "amount"
        elif "换手率" in cl:
            col_map[c] = "turnover"
        elif "涨跌幅" in cl:
            col_map[c] = "pct_chg"
    spot = spot.rename(columns=col_map)
    needed = ["code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]
    have = [c for c in needed if c in spot.columns]
    spot = spot[have].copy()
    spot["code"] = spot["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    # 仅主板
    spot = spot[spot["code"].str.match(r"^(60|00)")]
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]:
        if c in spot.columns:
            spot[c] = pd.to_numeric(spot[c], errors="coerce")
    # 剔除无效数据（最新价为空或0）
    spot = spot[(spot["close"] > 0) & spot["close"].notna()]
    return spot.reset_index(drop=True)


def fetch_intraday_data() -> pd.DataFrame:
    """用akshare新浪接口获取全市场实时行情，转换为与历史数据一致的格式。

    新浪接口按约70个板块分页抓取，单次调用常有板块超时漏抓（主板在2950~3115间波动，
    且每次漏抓的票不同）。因此多轮抓取并按 code 取并集补全，避免单次截断漏股；
    多轮并集后仍低于下限则抛异常，禁止拿残缺数据出预测。
    """
    logger.info("获取盘中实时行情（新浪接口，多轮并集补全）...")
    MIN_BOARD = 2800  # 并集后主板下限，低于此判定接口异常
    TARGET_BOARD = 3050  # 达到该数量即提前停抓（较全）
    MAX_ROUNDS = 3
    merged = pd.DataFrame()
    for r in range(1, MAX_ROUNDS + 1):
        try:
            one = _fetch_one_spot()
        except Exception as e:
            logger.warning(f"第{r}轮实时行情抓取失败({type(e).__name__}: {e})，继续下一轮")
            continue
        # 并集：同一 code 后到覆盖先到（同一时刻附近价格差异可忽略）
        merged = pd.concat([merged, one], ignore_index=True)
        merged = merged.drop_duplicates(subset="code", keep="last")
        logger.info(f"第{r}轮抓取主板 {len(one)} 只，累计并集 {len(merged)} 只")
        if len(merged) >= TARGET_BOARD:
            break
    # 合理性校验：主板正常 3000+ 只，多轮并集仍过少说明接口抖动/错误页/截断，直接抛异常
    if len(merged) < MIN_BOARD:
        raise RuntimeError(f"实时行情多轮并集仅 {len(merged)} 只主板（正常应3000+），疑似接口异常，拒绝使用")
    logger.info(f"盘中实时行情获取完成: {len(merged)} 只主板股票（多轮并集补全）")
    return merged.reset_index(drop=True)


def merge_intraday_data(historical: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    """将盘中实时数据作为最新一个交易日合并到历史数据（替换当日已有记录）"""
    now = datetime.now()
    # 实时快照作为当天这一个交易日（盘中重复运行时以最新快照整体替换当日行）
    intraday = intraday.copy()
    intraday["date"] = pd.Timestamp(now.date())
    # 历史数据中剔除当天已有记录（盘中重复运行时用最新实时数据替换）
    today = pd.Timestamp(now.date())
    hist = historical[historical["date"] < today].copy()

    # 防御1：剔除停牌/无成交的实时行（实时快照成交量为0=停牌，价格多为昨收，尾盘买不进）
    if "volume" in intraday.columns:
        n_before = len(intraday)
        intraday = intraday[intraday["volume"].fillna(0) > 0]
        n_stop = n_before - len(intraday)
        if n_stop > 0:
            logger.warning(f"盘中剔除停牌/无成交股票 {n_stop} 只")

    # 补齐历史有但实时缺失的列
    for c in hist.columns:
        if c not in intraday.columns:
            intraday[c] = np.nan
    intraday = intraday[hist.columns]

    # 防御2：实时价 vs 本地qfq昨收 涨幅异常(|跳空|>15%)，视为复权错位/脏数据，剔除该股今日行
    # （主板单日涨跌停±10%，qfq基准下正常不可能超15%；600520/600521曾因qfq串行出现-98%/-20%脏值）
    yesterday = (hist.sort_values("date").groupby("code", sort=False).tail(1)
                 [["code", "close"]].rename(columns={"close": "_prev_close"}))
    intraday = intraday.merge(yesterday, on="code", how="left")
    jump = (intraday["close"] - intraday["_prev_close"]) / intraday["_prev_close"]
    bad_mask = intraday["_prev_close"].notna() & (jump.abs() > 0.15)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        bad_codes = intraday.loc[bad_mask, "code"].tolist()
        bad_jumps = [f"{c}:{v * 100:+.1f}%" for c, v in zip(bad_codes, jump[bad_mask].tolist())]
        logger.warning(f"盘中剔除实时价与qfq昨收涨幅异常股票 {n_bad} 只: {bad_jumps}")
        intraday = intraday[~bad_mask]
    if "_prev_close" in intraday.columns:
        intraday = intraday.drop(columns=["_prev_close"])

    merged = pd.concat([hist, intraday], ignore_index=True)
    merged["code"] = merged["code"].astype("category")
    return merged.sort_values(["date", "code"]).reset_index(drop=True)


def build_market_advice(latest_factor_data: pd.DataFrame) -> str:
    """根据沪深300相对20日均线的位置给出市场操作建议"""
    trend_col = "hs300_trend"
    if trend_col not in latest_factor_data.columns:
        return ""
    try:
        trend = float(latest_factor_data[trend_col].iloc[-1])
        pct = trend * 100
        if trend < -0.05:
            return f"⚠️ 空仓避险：沪深300低于20日均线{pct:.1f}%（极端熊市，建议空仓）"
        elif trend < -0.02:
            return f"⚠️ 谨慎操作：沪深300低于20日均线{pct:.1f}%（弱势，降低仓位）"
        else:
            return f"✅ 正常持仓：沪深300高于20日均线{pct:.1f}%（可正常买Top3）"
    except Exception:
        return ""


def main(pred_=["排名", "股票代码", "股票名称", "收盘价", "当天涨幅", "pred", "所属板块"]):
    now = datetime.now()
    logger.info("=" * 70)
    logger.info("A股量化预测开始")
    logger.info(f"当前时间: {now:%Y-%m-%d %H:%M:%S}")

    # ========== Step 1: 判断交易时间，决定数据来源 ==========
    logger.info("\n" + "=" * 70)
    use_intraday = is_trading_hours(now)
    intraday_df = None
    if use_intraday:
        # 铁律：盘中预测永远使用当前时刻的实时行情，什么时候运行就什么时候重新拉取，
        # 没有任何时间间隔/节流，禁止复用任何缓存或旧快照；拉取失败直接退出，不降级到历史数据。
        logger.info("Step 1: 交易时间内，每次运行强制拉取盘中实时数据（不复用缓存）")
        try:
            intraday_df = fetch_intraday_data()
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"盘中实时数据获取失败({e})：预测已停止。")
            logger.error("盘中预测必须基于当日实时行情，禁止复用缓存/历史数据冒充实时，请稍后重试。")
            logger.error("=" * 70)
            sys.exit(1)
    else:
        logger.info("Step 1: 非交易时间，使用训练文件的最新存量数据（不更新）")

    # ========== Step 2: 加载数据并计算最新因子 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 加载数据并计算最新因子")
    all_data = load_all_data()
    if use_intraday and intraday_df is not None and not intraday_df.empty:
        all_data = merge_intraday_data(all_data, intraday_df)
        logger.info(f"本次预测基于实时数据: {all_data['date'].min():%Y-%m-%d} ~ {all_data['date'].max():%Y-%m-%d}")
    elif use_intraday:
        # 双保险：盘中走到这里说明实时数据缺失（Step 1 已应拦截），直接停止，不允许降级
        logger.error("盘中实时数据缺失：预测已停止，禁止用历史存量数据冒充实时行情。")
        sys.exit(1)
    else:
        logger.info(f"本次预测基于存量数据: {all_data['date'].min():%Y-%m-%d} ~ {all_data['date'].max():%Y-%m-%d}")

    factor_data, factor_cols = compute_factors(all_data, skip_redundancy=True)
    del all_data
    import gc
    gc.collect()

    # 智能识别最新交易日：股票数量≥1000的最新一天，避免盘中数据不完整
    day_counts = factor_data.groupby("date")["code"].nunique()
    valid_days = day_counts[day_counts >= MIN_STOCKS_PER_DAY].index
    if len(valid_days) == 0:
        logger.error("没有找到股票数量足够的交易日，预测终止")
        return
    latest_date = valid_days.max()
    latest_data = factor_data[factor_data["date"] == latest_date].copy()

    # 股票池过滤：只保留主板可买列表内股票，剔除已退市/ST残留等池外异常票（在预测前过滤，保证 pred 与行情同序）
    try:
        _stock_list = get_stock_list()
        _tradable = set(_stock_list["code"].astype(str).str.zfill(6))
        _before = len(latest_data)
        latest_data = latest_data[latest_data["code"].astype(str).str.zfill(6).isin(_tradable)].copy()
        logger.info(f"股票池过滤: {_before} -> {len(latest_data)} 只（剔除池外异常票 {_before - len(latest_data)} 只）")
    except Exception as e:
        logger.warning(f"股票池过滤失败({e})，未过滤")

    logger.info(f"预测基准日: {latest_date:%Y-%m-%d}, 可预测股票数: {len(latest_data)}")

    # 读取预测模型的训练数据范围（checkpoint）
    cp_path = MODEL_DIR / PREDICT_TAG / "_checkpoint.json"
    if cp_path.exists():
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            sig = cp.get("data_signature", "")
            logger.info(f"预测模型训练数据: 截止{cp.get('train_end_date', '?')}（100%全量数据训练，签名{sig}）")
        except Exception:
            pass

    # ========== Step 3: 加载模型并预测 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: 加载模型并预测")
    models = {}
    for h in HORIZONS:
        try:
            model = EnsembleModel(horizon=h)
            model.load(tag=PREDICT_TAG)
            models[h] = model
        except Exception as e:
            logger.error(f"模型加载失败 ({h}): {e}")
    if not models:
        logger.error("没有可用模型，请先运行 python train.py")
        return

    # 强制使用模型训练时的因子列表，避免实时去冗余导致特征数不匹配
    import os as _os
    import json as _json
    _meta_path = _os.path.join(MODEL_DIR, PREDICT_TAG, f"lgb_{HORIZONS[0]}.meta.json")
    try:
        with open(_meta_path, "r", encoding="utf-8") as _f:
            _meta = _json.load(_f)
        model_factor_cols = _meta.get("feature_names", [])
        if model_factor_cols:
            missing = [c for c in model_factor_cols if c not in latest_data.columns]
            if missing:
                logger.error(f"模型因子在最新数据中缺失: {missing}")
                return
            factor_cols = model_factor_cols
            logger.info(f"已对齐模型因子列表: {len(factor_cols)} 个因子")
    except Exception as _e:
        logger.warning(f"读取模型因子列表失败({_e})，使用实时计算因子列表")

    available_horizons = list(models.keys())
    predictions = {}
    confidences = {}
    X_latest = latest_data[factor_cols]
    for h in available_horizons:
        pred = models[h].predict(X_latest)
        predictions[h] = pred
        confidences[h] = compute_confidence(pred)
        logger.info(
            f"1天持仓-动态卖出 ({h}) 预测完成，置信度范围: [{confidences[h].min():.4f}, {confidences[h].max():.4f}]")

    # ========== Step 4: 组装预测结果 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 组装预测结果")
    # 构造输出：除 code/close 外，一次性取齐二次排序所需的原始价格与因子列。
    # 必须包含 open/high/low（否则K线形态因子 issubset 恒为 False，回测有而实盘不生效的历史bug）。
    # 列清单与 backtest_engine.composite 完全对齐：基础5因子 + K线(open/high/low) + 规模/量能/资金流3因子。
    _rerank_cols = ["code", "open", "high", "low", "close",
                    "amplitude_1d", "vol_ratio_5", "gap_ratio", "close_position_5d", "ret_3d",
                    "mv_log_yi", "vol_ratio_10", "mf_intraday"]
    _take = [c for c in _rerank_cols if c in latest_data.columns]
    result = latest_data[_take].copy()
    result["code"] = result["code"].astype(str).str.zfill(6)

    # 加载股票名称映射
    try:
        stock_list = get_stock_list()
        name_map = dict(zip(stock_list["code"].astype(str).str.zfill(6), stock_list["name"]))
        result["name"] = result["code"].map(name_map).fillna("")
        logger.info(f"加载股票名称映射: {len(name_map)} 只")
    except Exception as e:
        logger.warning(f"股票名称加载失败({e})，名称留空")
        result["name"] = ""

    # 加载股票行业（板块）映射
    try:
        industry_df = get_stock_industry()
        industry_map = dict(zip(industry_df["code"].astype(str).str.zfill(6), industry_df["industry"]))
        result["industry"] = result["code"].map(industry_map).fillna("")
        logger.info(f"加载股票行业映射: {len(industry_map)} 只")
    except Exception as e:
        logger.warning(f"股票行业加载失败({e})，板块留空")
        result["industry"] = ""

    # 最新涨跌幅：优先用ret_1d_raw（原始收益率，未标准化）
    if "ret_1d_raw" in latest_data.columns:
        result["latest_pct_chg"] = latest_data["ret_1d_raw"].values
    else:
        result["latest_pct_chg"] = 0.0

    # 涨停标记：主板涨停10%，>=9.9%视为涨停（考虑四舍五入）
    result["is_limit_up"] = result["latest_pct_chg"] >= 0.095
    result["limit_status"] = result["is_limit_up"].map({True: "涨停", False: "可买入"})

    for h in available_horizons:
        result[f"pred_ret_{h}"] = np.round(predictions[h], 6)
        result[f"confidence_{h}"] = confidences[h]

    # 排序：动态卖出置信度降序
    sort_cols = [f"confidence_{h}" for h in available_horizons]
    result = result.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    # 市场操作建议
    market_advice = build_market_advice(latest_data)
    data_end = factor_data["date"].max()

    # ========== Step 5: 输出Top N ==========
    logger.info("\n" + "=" * 70)
    sort_desc = "动态卖出置信度"
    logger.info(f"Step 5: Top {TOP_N} 推荐（排序优先级: {sort_desc}）")

    top = result.head(TOP_N).copy()
    top.index = range(1, len(top) + 1)

    h0 = available_horizons[0]
    display_cols = ["code", "name", "close", "latest_pct_chg", "limit_status", f"pred_ret_{h0}"]
    display_names = ["股票代码", "股票名称", "收盘价", "最新涨跌幅", "状态", "pred"]
    # 所属板块放最后一列（排序仍按动态卖出置信度，仅不显示该列）
    display_cols.append("industry")
    display_names.append("所属板块")

    display = top[display_cols].copy()
    display.columns = display_names
    # 在最前面插入排名列（1,2,3...）
    display.insert(0, "排名", range(1, len(display) + 1))
    # 收盘价保留2位小数
    display["收盘价"] = display["收盘价"].round(2)
    # 动态卖出置信度保留4位小数
    if "动态卖出置信度" in display.columns:
        display["动态卖出置信度"] = display["动态卖出置信度"].round(4)
    # 涨跌幅显示为百分比
    display["最新涨跌幅"] = (display["最新涨跌幅"] * 100).round(2).astype(str) + "%"
    display["pred"] = display["pred"].round(4)  # pred=预测次日收益率，保留4位小数

    print("\n" + "=" * 100)
    buy_strategy = "今日尾盘买入" if use_intraday else "下一交易日尾盘买入"
    if use_intraday:
        data_time_str = f"{now:%Y-%m-%d %H:%M}（实时快照）"
    else:
        data_time_str = f"{data_end:%Y-%m-%d}（收盘）"
    print(
        f"预测基准日: {latest_date:%Y-%m-%d}  |  数据时间: {data_time_str}  |  策略: {buy_strategy}, 排序优先级: {sort_desc}")
    print("=" * 100)
    print_table(display.head(4000))  # 终端只打印前50，完整结果存CSV
    print("=" * 100)

    # ========== 综合排序Top10（模型粗筛Top10→composite二次排序，口径与 backtest_engine 完全一致） ==========
    try:
        buyable = result[~result["is_limit_up"]].copy()
        buyable = buyable[buyable["latest_pct_chg"] <= 0.090]  # 涨幅过滤：无下限，上限9.0%（避免追高，与回测一致）
        pred_col = f"pred_ret_{available_horizons[0]}"
        # 粗筛：按 pred 选 Top10 候选（与回测 nlargest(pred) 口径一致；置信度是pred的单调映射，集合相同）
        candidates = buyable.nlargest(10, pred_col).copy()
        candidates["pred"] = candidates[pred_col]
        # composite 打分，与 backtest_engine._backtest_single_mode 逐因子对齐（score 越小越靠前）
        # 基础5因子：[振幅2, 5日量比2, 跳空1, 5日位置1, 3日动量1]
        w = [2, 2, 1, 1, 1]
        score = pd.Series(0.0, index=candidates.index)
        if "amplitude_1d" in candidates.columns:
            score += w[0] * candidates["amplitude_1d"].rank(ascending=True)  # 振幅小→靠前
        if "vol_ratio_5" in candidates.columns:
            score += w[1] * candidates["vol_ratio_5"].rank(ascending=True)  # 缩量→靠前
        if "gap_ratio" in candidates.columns:
            score -= w[2] * candidates["gap_ratio"].rank(ascending=True)  # 高开→靠前
        if "close_position_5d" in candidates.columns:
            score -= w[3] * candidates["close_position_5d"].rank(ascending=True)  # 5日高位→靠前
        if "ret_3d" in candidates.columns:
            score -= w[4] * candidates["ret_3d"].rank(ascending=True)  # 3日动量强→靠前
        # K线形态（open/high/low 现已随 result 取齐；上影短/光头阳权重由常量单一来源）
        if (KLINE_SHADOW_W or KLINE_KFLAG_W) and \
                {'open', 'high', 'low', 'close'}.issubset(candidates.columns):
            _eps = 1e-9
            _rng = (candidates['high'] - candidates['low']).clip(lower=_eps)
            _upper = (candidates['high'] - np.maximum(candidates['open'], candidates['close'])) / _rng
            _bull = ((candidates['close'] >= candidates['high'] - 0.01 * _rng) & (
                    candidates['close'] > candidates['open']))
            _bear = ((candidates['close'] <= candidates['low'] + 0.01 * _rng) & (
                    candidates['close'] < candidates['open']))
            if KLINE_SHADOW_W:
                score += KLINE_SHADOW_W * _upper.rank(ascending=True)  # 上影线短→rank小→靠前
            if KLINE_KFLAG_W:
                _morph = _bull.astype(int) - _bear.astype(int)
                score -= KLINE_KFLAG_W * _morph.rank(ascending=True)  # 光头阳→靠前；光头阴→靠后
        # 规模/量能/资金流三因子（小值优先，各权重1，与回测一致）
        if RERANK_MV_W and "mv_log_yi" in candidates.columns:
            score += RERANK_MV_W * candidates["mv_log_yi"].rank(ascending=True)  # 小市值→靠前
        if RERANK_VOL10_W and "vol_ratio_10" in candidates.columns:
            score += RERANK_VOL10_W * candidates["vol_ratio_10"].rank(ascending=True)  # 10日缩量→靠前
        if RERANK_MF_W and "mf_intraday" in candidates.columns:
            score += RERANK_MF_W * candidates["mf_intraday"].rank(ascending=True)  # 低位收盘/非放量→靠前
        candidates["综合排序分"] = score
        candidates = candidates.sort_values("综合排序分")
        rerank_desc = "composite二次排序（5基础因子+K线形态+规模/量能/资金流）"
        rerank_display = candidates[["code", "name", "close", "latest_pct_chg", "pred", "industry"]].copy()
        rerank_display.insert(0, "排名", range(1, len(rerank_display) + 1))
        rerank_display.columns = pred_
        rerank_display["收盘价"] = rerank_display["收盘价"].round(2)
        rerank_display["当天涨幅"] = (rerank_display["当天涨幅"] * 100).round(2).astype(str) + "%"
        rerank_display["pred"] = rerank_display["pred"].map(lambda x: f"{x:.4f}")  # 统一4位小数对齐
        print()
        print("=" * 100)
        print(f"综合排序Top10（模型粗筛Top10→{rerank_desc}）")
        print("=" * 100)
        print_table(rerank_display)
        top1_pred = float(candidates["pred"].iloc[0])
        if top1_pred < MIN_PRED:
            print(
                f"  [弱信号-建议空仓] 综合排序Top1预测次日收益仅 {top1_pred * 100:.2f}%(<{MIN_PRED * 100:.1f}%)，历史弱信号胜率约58%，当日不买入")
        else:
            print(
                f"  [信号有效-可买入] 综合排序Top1预测次日收益 {top1_pred * 100:.2f}%(>={MIN_PRED * 100:.1f}%)，可尾盘买入")
            _tp = DYN_TP_LOW if top1_pred < DYN_TP_BOUNDARY else DYN_TP_HIGH
            print(
                f"  [次日卖出] 开盘止盈{_tp * 100:.1f}%(pred{"<" if top1_pred < DYN_TP_BOUNDARY else "≥"}1%)，盘中止盈{INTRADAY_TP * 100:.0f}%，均未触及则收盘卖出，不止损")
        print("=" * 100)
    except Exception as e:
        logger.warning(f"综合排序打印失败（已跳过）: {e}")

    # 保存完整预测结果到Excel（单文件多sheet，每天一个sheet，同日重跑覆盖旧数据）
    # 盘中模式跳过Excel更新；任何异常只警告不中断
    if not use_intraday:
        try:
            from openpyxl.styles import Font
            from openpyxl.utils import get_column_letter
            excel_path = RESULT_DIR / PREDICT_TAG / "predictions.xlsx"
            excel_path.parent.mkdir(parents=True, exist_ok=True)
            sheet_name = latest_date.strftime("%Y-%m-%d")

            excel_cols = ["code", "name", "close", "latest_pct_chg", "limit_status",
                          f"pred_ret_{available_horizons[0]}", "industry",
                          f"confidence_{available_horizons[0]}"]
            excel_df = result[excel_cols].copy()
            excel_df.insert(0, "排名", range(1, len(excel_df) + 1))
            excel_df.columns = ["排名", "股票代码", "股票名称", "收盘价", "当天涨幅", "状态", "pred", "所属板块",
                                "动态卖出置信度"]
            excel_df["收盘价"] = excel_df["收盘价"].round(2)
            excel_df["当天涨幅"] = (excel_df["当天涨幅"] * 100).round(2).map(lambda x: f"{x:.2f}%")
            excel_df["pred"] = excel_df["pred"].round(4)  # pred=预测次日收益率，4位小数

            file_exists = excel_path.exists()
            excel_mode = "a" if file_exists else "w"
            with pd.ExcelWriter(excel_path, engine="openpyxl", mode=excel_mode,
                                if_sheet_exists="replace" if file_exists else None) as writer:
                excel_df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                for row in ws.iter_rows():
                    for cell in row:
                        cell.font = Font(name="微软雅黑", size=11)
                # 收盘价列（第4列）强制两位小数显示
                for cell in ws["D"]:
                    if cell.row > 1 and cell.value is not None:
                        cell.number_format = "0.00"
                for cell in ws["G"]:
                    if cell.row > 1 and cell.value is not None:
                        cell.number_format = "0.0000"
                for col_idx, col in enumerate(ws.columns, 1):
                    max_len = 0
                    for cell in col:
                        if cell.value is not None:
                            # D列(收盘价)按两位小数显示格式计算宽度
                            if col_idx == 4 and cell.row > 1:
                                v = f"{cell.value:.2f}"
                            elif col_idx == 7 and cell.row > 1:
                                v = f"{cell.value:.4f}"
                            else:
                                v = str(cell.value)
                            w = sum(2 if ord(ch) > 127 else 1 for ch in v)
                            if w > max_len:
                                max_len = w
                    ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 2
            logger.info(f"预测结果已保存: {excel_path} [sheet={sheet_name}, {len(excel_df)}行]")
        except Exception as e:
            logger.warning(f"Excel保存失败（已跳过）: {e}")

    # 市场操作建议
    if market_advice:
        logger.info("\n" + "=" * 70)
        logger.info(f"市场操作建议: {market_advice}")

    logger.info("\n预测完成！")


if __name__ == "__main__":
    main()
