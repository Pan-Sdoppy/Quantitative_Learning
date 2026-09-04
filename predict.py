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
7. 打印本次预测基于的数据时间范围

用法：
    python predict.py
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_loader import load_all_data
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
INTRADAY_MIN_INTERVAL = 15  # 盘中最小更新间隔（分钟）


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
NUMERIC_COLS = {"收盘价", "最新涨跌幅", "动态卖出置信度", "动态卖出预测收益"}
TEXT_COLS = {"股票代码", "股票名称", "状态"}


def _format_val(col: str, val) -> str:
    """根据列名格式化值，确保数字精度正确"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if col == "收盘价":
        return f"{float(val):.2f}"
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


    for h in available_horizons:
        result[f"pred_ret_{h}"] = np.round(predictions[h], 6)
        result[f"confidence_{h}"] = confidences[h]

    result = result.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    # ========== Step 5: 输出Top N ==========
    logger.info("\n" + "=" * 70)
    sort_desc = "动态卖出置信度"
    logger.info(f"Step 5: Top {TOP_N} 推荐（排序优先级: {sort_desc}）")

    top = result.head(TOP_N).copy()
    top.index = range(1, len(top) + 1)

    display_cols = ["code", "name", "close", "latest_pct_chg", "limit_status"]
    display_names = ["股票代码", "股票名称", "收盘价", "最新涨跌幅", "状态"]
    for h in available_horizons:
        display_cols.extend([f"confidence_{h}"])
        mode_name = "动态卖出置信度"
        display_names.extend([mode_name])

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
        minute_floor = (now.minute // 15) * 15
        data_time_str = f"{now:%Y-%m-%d} {now.hour:02d}:{minute_floor:02d}"
    else:
        data_time_str = f"{data_end:%Y-%m-%d}（收盘）"
    print(
        f"预测基准日: {latest_date:%Y-%m-%d}  |  数据时间: {data_time_str}  |  策略: {buy_strategy}, 排序优先级: {sort_desc}")
    print("=" * 100)
    print_table(display)
    print("=" * 100)

    # 保存完整预测结果
    out_path = RESULT_DIR / PREDICT_TAG / f"prediction_{latest_date:%Y%m%d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"完整预测结果已保存: {out_path}")

    # 推荐买入Top3
    logger.info("\n" + "=" * 70)
    logger.info(f"推荐买入 Top 3（排序优先级: {sort_desc}）:")
    for i, (_, row) in enumerate(top.head(3).iterrows(), 1):
        conf_str = "  ".join([
            f"动态卖出:{row[f'confidence_{h}']:.4f}"
            for h in available_horizons
        ])
        pct_str = f"{row['latest_pct_chg'] * 100:+.2f}%"
        limit_str = f"  状态:{row['limit_status']}"
        logger.info(
            f"  {i}. {row['code']} {row['name']}  收盘价:{row['close']:.2f}  涨跌幅:{pct_str}{limit_str}  {conf_str}")

    # 市场操作建议（在股票推荐之后给出）
    if market_advice:
        logger.info("\n" + "=" * 70)
        logger.info(f"市场操作建议: {market_advice}")

    logger.info("\n预测完成！")


if __name__ == "__main__":
    main()
