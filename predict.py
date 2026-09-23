"""
预测入口文件
============
功能：
1. 自动判断当前时间：交易日9:30-15:00自动更新最新实时数据，其余时间用训练文件的最新数据
2. 计算最新因子
3. 加载已训练的预测模型，对全市场股票预测
4. 给出1天持仓动态卖出的置信度（保留4位小数）：
      - 1d_close: T日尾盘买，T+1尾盘收盘卖
5. 按动态卖出置信度优先排序，输出Top股票推荐
6. 打印本次预测基于的数据时间范围

用法：
    python predict.py
"""
import json
import warnings
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_loader import load_all_data, get_stock_list, get_stock_industry
from core.factors import compute_factors
from core.model import EnsembleModel, compute_confidence
from core.utils import RESULT_DIR, MODEL_DIR, get_logger, RAW_DIR

warnings.filterwarnings("ignore")
logger = get_logger("predict")

PREDICT_TAG = "latest/predict"  # 预测自动加载预测模型（100%全量数据训练）
TOP_N = 5000
HORIZONS = ["1d_close"]
INTRADAY_CACHE_FILE = Path(RAW_DIR) / "_intraday_last_update.json"
INTRADAY_DATA_CACHE = Path(RAW_DIR) / "_intraday_cache.parquet"
INTRADAY_MIN_INTERVAL = 0  # 盘中每次运行都获取最新实时数据，不缓存（0=不节流）
MIN_STOCKS_PER_DAY = 1000  # 最新交易日最少股票数，避免盘中数据不完整


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


# 数字列（右对齐）和文字列（左对齐）的判断
NUMERIC_COLS = {"收盘价", "最新涨跌幅"}
TEXT_COLS = {"股票代码", "股票名称", "状态", "所属板块"}


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


def should_update_intraday(now: datetime) -> bool:
    """距上次盘中更新是否已超过最小间隔（15分钟）"""
    if not INTRADAY_CACHE_FILE.exists():
        return True
    try:
        with open(INTRADAY_CACHE_FILE, "r", encoding="utf-8") as f:
            last = json.load(f).get("last_update")
        if not last:
            return True
        last_dt = datetime.fromisoformat(last)
        elapsed = (now - last_dt).total_seconds() / 60
        return elapsed >= INTRADAY_MIN_INTERVAL
    except Exception:
        return True


def save_intraday_update_time(now: datetime):
    """记录本次盘中数据更新时间"""
    try:
        with open(INTRADAY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_update": now.isoformat()}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"盘中更新时间保存失败: {e}")


def load_intraday_cache(now: datetime):
    """加载盘中实时数据缓存（距上次更新不足15分钟时复用）"""
    if INTRADAY_DATA_CACHE.exists():
        try:
            return pd.read_parquet(INTRADAY_DATA_CACHE)
        except Exception:
            return None
    return None


def save_intraday_cache(df: pd.DataFrame):
    """缓存盘中实时数据"""
    try:
        df.to_parquet(INTRADAY_DATA_CACHE, index=False)
    except Exception as e:
        logger.warning(f"盘中数据缓存失败: {e}")


def fetch_intraday_data() -> pd.DataFrame:
    """用akshare新浪接口获取全市场实时行情，转换为与历史数据一致的格式"""
    import akshare as ak
    logger.info("获取盘中实时行情（新浪接口）...")
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
    logger.info(f"盘中实时行情获取完成: {len(spot)} 只主板股票")
    return spot.reset_index(drop=True)


def merge_intraday_data(historical: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    """将盘中实时数据作为最新一个交易日合并到历史数据（替换当日已有记录）"""
    now = datetime.now()
    # 盘中数据以15分钟为间隔向下取整，日期取当天
    intraday = intraday.copy()
    intraday["date"] = pd.Timestamp(now.date())
    # 历史数据中剔除当天已有记录（盘中重复运行时用最新实时数据替换）
    today = pd.Timestamp(now.date())
    hist = historical[historical["date"] < today].copy()
    # 补齐历史有但实时缺失的列
    for c in hist.columns:
        if c not in intraday.columns:
            intraday[c] = np.nan
    intraday = intraday[hist.columns]
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


def main():
    now = datetime.now()
    logger.info("=" * 70)
    logger.info("A股量化预测开始")
    logger.info(f"当前时间: {now:%Y-%m-%d %H:%M:%S}")

    # ========== Step 1: 判断交易时间，决定数据来源 ==========
    logger.info("\n" + "=" * 70)
    use_intraday = is_trading_hours(now)
    intraday_df = None
    if use_intraday:
        if should_update_intraday(now):
            logger.info("Step 1: 交易时间内，更新盘中实时数据")
            try:
                intraday_df = fetch_intraday_data()
                save_intraday_cache(intraday_df)
                save_intraday_update_time(now)
            except Exception as e:
                logger.warning(f"盘中实时数据获取失败({e})，尝试使用缓存")
                intraday_df = load_intraday_cache(now)
                if intraday_df is None:
                    use_intraday = False
        else:
            logger.info("Step 1: 交易时间内，但距上次更新不足15分钟，使用缓存")
            intraday_df = load_intraday_cache(now)
            if intraday_df is None:
                use_intraday = False
    else:
        logger.info("Step 1: 非交易时间，使用训练文件的最新存量数据（不更新）")

    # ========== Step 2: 加载数据并计算最新因子 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 加载数据并计算最新因子")
    all_data = load_all_data()
    if use_intraday and intraday_df is not None and not intraday_df.empty:
        all_data = merge_intraday_data(all_data, intraday_df)
        logger.info(f"本次预测基于实时数据: {all_data['date'].min():%Y-%m-%d} ~ {all_data['date'].max():%Y-%m-%d}")
    else:
        use_intraday = False
        logger.info(f"本次预测基于存量数据: {all_data['date'].min():%Y-%m-%d} ~ {all_data['date'].max():%Y-%m-%d}")

    factor_data, factor_cols = compute_factors(all_data)
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
    result = latest_data[["code", "close"]].copy()
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

    # 二次排序所需因子（7因子：振幅小+量比低+跳空高+换手率低+成交额低+5日位置高+3日动量高）
    for rc in ["amplitude_1d", "vol_ratio_5", "gap_ratio"]:
        if rc in latest_data.columns:
            result[rc] = latest_data[rc].values
        else:
            result[rc] = np.nan

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

    display_cols = ["code", "name", "close", "latest_pct_chg", "limit_status"]
    display_names = ["股票代码", "股票名称", "收盘价", "最新涨跌幅", "状态"]
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

    # ========== 综合排序Top10（二次排序：模型粗筛Top10→7因子综合rank重排） ==========
    try:
        buyable = result[~result["is_limit_up"]].copy()
        buyable = buyable[(buyable["latest_pct_chg"] >= -0.03) & (buyable["latest_pct_chg"] <= 0.07)]
        conf_col = f"confidence_{available_horizons[0]}"
        candidates = buyable.nlargest(10, conf_col).copy()
        # 7因子综合rank排序（权重与回测一致）
        # [振幅小2, 量比低1, 跳空高1, 换手率低1, 成交额低1, 5日位置高1, 3日动量高1]
        for rc in ["turnover_ratio", "amount_ratio", "close_position_5d", "ret_3d"]:
            if rc not in candidates.columns and rc in latest_data.columns:
                candidates[rc] = latest_data.set_index("code").reindex(candidates["code"])[rc].values
        score = pd.Series(0.0, index=candidates.index)
        if "amplitude_1d" in candidates.columns:
            score += 2 * candidates["amplitude_1d"].rank(ascending=True)
        if "vol_ratio_5" in candidates.columns:
            score += 1 * candidates["vol_ratio_5"].rank(ascending=True)
        if "gap_ratio" in candidates.columns:
            score -= 1 * candidates["gap_ratio"].rank(ascending=True)
        if "turnover_ratio" in candidates.columns:
            score += 1 * candidates["turnover_ratio"].rank(ascending=True)
        if "amount_ratio" in candidates.columns:
            score += 1 * candidates["amount_ratio"].rank(ascending=True)
        if "close_position_5d" in candidates.columns:
            score -= 1 * candidates["close_position_5d"].rank(ascending=True)
        if "ret_3d" in candidates.columns:
            score -= 1 * candidates["ret_3d"].rank(ascending=True)
        candidates["综合排序分"] = score
        candidates = candidates.sort_values("综合排序分")
        rerank_display = candidates[["code", "name", "close", "latest_pct_chg", "industry", conf_col]].copy()
        rerank_display.insert(0, "排名", range(1, len(rerank_display) + 1))
        rerank_display.columns = ["排名", "股票代码", "股票名称", "收盘价", "当天涨幅", "所属板块", "模型置信度"]
        rerank_display["收盘价"] = rerank_display["收盘价"].round(2)
        rerank_display["当天涨幅"] = (rerank_display["当天涨幅"] * 100).round(2).astype(str) + "%"
        rerank_display["模型置信度"] = rerank_display["模型置信度"].round(4)
        print()
        print("=" * 100)
        print("综合排序Top10（模型粗筛Top10→7因子重排：振幅小2+量比低1+跳空高1+换手率低1+成交额低1+5日位置高1+3日动量高1，回测收益24763万%）")
        print("=" * 100)
        print_table(rerank_display)
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

            excel_cols = ["code", "name", "close", "latest_pct_chg", "limit_status", "industry",
                          f"confidence_{available_horizons[0]}"]
            excel_df = result[excel_cols].copy()
            excel_df.insert(0, "排名", range(1, len(excel_df) + 1))
            excel_df.columns = ["排名", "股票代码", "股票名称", "收盘价", "当天涨幅", "状态", "所属板块", "动态卖出置信度"]
            excel_df["收盘价"] = excel_df["收盘价"].round(2)
            excel_df["当天涨幅"] = (excel_df["当天涨幅"] * 100).round(2).map(lambda x: f"{x:.2f}%")

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
                for col_idx, col in enumerate(ws.columns, 1):
                    max_len = 0
                    for cell in col:
                        if cell.value is not None:
                            # D列(收盘价)按两位小数显示格式计算宽度
                            if col_idx == 4 and cell.row > 1:
                                v = f"{cell.value:.2f}"
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
