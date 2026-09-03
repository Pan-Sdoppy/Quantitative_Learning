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
6. 计算主力出货置信度（6大信号加权综合），辅助判断买入风险
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
    elif col in ("出货置信度", "动态卖出置信度", "动态卖出预测收益"):
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


def compute_distribution_risk(all_data: pd.DataFrame, latest_date) -> pd.DataFrame:
    """专业6维度主力出货置信度计算（0-1，越高越可能出货）
    维度1: 位置判断(20%) - 看高位
    维度2: 成交量判断(25%) - 看爆量
    维度3: K线形态判断(25%) - 看形态
    维度4: 技术指标判断(15%) - MACD/RSI/KDJ/布林
    维度5: 资金行为判断(15%) - 尾盘跳水/冲高回落/收盘价vs均价
    """
    df = all_data.sort_values(["code", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # 保留近120天数据用于计算
    cutoff = pd.to_datetime(latest_date) - pd.Timedelta(days=180)
    df = df[df["date"] >= cutoff]

    results = []
    for code, grp in df.groupby("code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < 20:
            results.append({"code": code, "distribution_risk": 0.0})
            continue

        latest = grp.iloc[-1]
        close = float(latest["close"])
        open_p = float(latest["open"])
        high = float(latest["high"])
        low = float(latest["low"])
        volume = float(latest["volume"])
        turnover = float(latest.get("turnover", 0))
        pct_chg = float(latest.get("pct_chg", 0))

        # ===== 预计算技术指标 =====
        # 均线
        for w in [5, 10, 20, 60]:
            grp[f"ma{w}"] = grp["close"].rolling(w).mean()
            grp[f"vma{w}"] = grp["volume"].rolling(w).mean()

        # 布林带(20,2)
        grp["boll_mid"] = grp["close"].rolling(20).mean()
        grp["boll_std"] = grp["close"].rolling(20).std()
        grp["boll_upper"] = grp["boll_mid"] + 2 * grp["boll_std"]
        grp["boll_lower"] = grp["boll_mid"] - 2 * grp["boll_std"]
        grp["boll_width"] = (grp["boll_upper"] - grp["boll_lower"]) / grp["boll_mid"]

        # ATR(14)
        tr = pd.concat([
            grp["high"] - grp["low"],
            (grp["high"] - grp["close"].shift()).abs(),
            (grp["low"] - grp["close"].shift()).abs()
        ], axis=1).max(axis=1)
        grp["atr14"] = tr.rolling(14).mean()

        # RSI(14)
        delta = grp["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        grp["rsi14"] = 100 - 100 / (1 + rs)

        # KDJ(9,3,3)
        low_n = grp["low"].rolling(9).min()
        high_n = grp["high"].rolling(9).max()
        rsv = (grp["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        grp["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
        grp["kdj_d"] = grp["kdj_k"].ewm(alpha=1/3, adjust=False).mean()
        grp["kdj_j"] = 3 * grp["kdj_k"] - 2 * grp["kdj_d"]

        # MACD(12,26,9)
        ema12 = grp["close"].ewm(span=12, adjust=False).mean()
        ema26 = grp["close"].ewm(span=26, adjust=False).mean()
        grp["macd_dif"] = ema12 - ema26
        grp["macd_dea"] = grp["macd_dif"].ewm(span=9, adjust=False).mean()
        grp["macd_hist"] = (grp["macd_dif"] - grp["macd_dea"]) * 2

        # 累计涨幅
        grp["ret_5d"] = grp["close"].pct_change(5)
        grp["ret_10d"] = grp["close"].pct_change(10)
        grp["ret_20d"] = grp["close"].pct_change(20)
        grp["ret_60d"] = grp["close"].pct_change(60)

        # 高低点
        grp["high_20d"] = grp["high"].rolling(20).max()
        grp["high_60d"] = grp["high"].rolling(60).max()
        grp["low_20d"] = grp["low"].rolling(20).min()

        last = grp.iloc[-1]
        prev = grp.iloc[-2] if len(grp) >= 2 else last
        prev2 = grp.iloc[-3] if len(grp) >= 3 else prev

        # 安全取值
        def safe(val, default=0):
            return float(val) if not pd.isna(val) else default

        ma5 = safe(last["ma5"])
        ma10 = safe(last["ma10"])
        ma20 = safe(last["ma20"])
        ma60 = safe(last["ma60"])
        vma5 = safe(last["vma5"], volume)
        vma20 = safe(last["vma20"], volume)
        boll_upper = safe(last["boll_upper"], high)
        boll_mid = safe(last["boll_mid"], close)
        boll_width = safe(last["boll_width"])
        prev_boll_width = safe(prev["boll_width"])
        rsi = safe(last["rsi14"], 50)
        kdj_k = safe(last["kdj_k"], 50)
        kdj_d = safe(last["kdj_d"], 50)
        kdj_j = safe(last["kdj_j"], 50)
        prev_kdj_k = safe(prev["kdj_k"], 50)
        prev_kdj_d = safe(prev["kdj_d"], 50)
        macd_dif = safe(last["macd_dif"])
        macd_dea = safe(last["macd_dea"])
        macd_hist = safe(last["macd_hist"])
        prev_macd_hist = safe(prev["macd_hist"])
        ret_5d = safe(last["ret_5d"])
        ret_10d = safe(last["ret_10d"])
        ret_20d = safe(last["ret_20d"])
        ret_60d = safe(last["ret_60d"])
        high_60d = safe(last["high_60d"], high)
        high_20d = safe(last["high_20d"], high)
        low_20d = safe(last["low_20d"], low)
        atr14 = safe(last["atr14"])

        # ===== 维度1: 位置判断（20%）看高位 =====
        pos_score = 0.0
        # 1a. 相对位置（在近20日高低点中的位置）
        if high_20d > low_20d:
            relative_pos = (close - low_20d) / (high_20d - low_20d)
            if relative_pos > 0.9:
                pos_score += 0.06
            elif relative_pos > 0.8:
                pos_score += 0.04
            elif relative_pos > 0.7:
                pos_score += 0.02

        # 1b. 距60日高点距离
        if high_60d > 0:
            dist_from_high = (high_60d - close) / high_60d
            if dist_from_high < 0.02:
                pos_score += 0.05
            elif dist_from_high < 0.05:
                pos_score += 0.03

        # 1c. 累计涨幅过大
        if ret_20d > 0.50 or ret_60d > 1.0:
            pos_score += 0.05
        elif ret_20d > 0.30 or ret_60d > 0.60:
            pos_score += 0.03
        elif ret_10d > 0.25:
            pos_score += 0.02

        # 1d. 均线多头排列 + 价格在所有均线上方
        if ma5 > ma10 > ma20 > ma60 and close > ma5:
            pos_score += 0.02

        # 1e. 触及布林带上轨
        if close >= boll_upper * 0.98:
            pos_score += 0.02

        pos_score = min(pos_score, 0.20)

        # ===== 维度2: 成交量判断（25%）看爆量 =====
        vol_score = 0.0
        # 2a. 量比（当日量/5日均量）
        if vma5 > 0:
            vol_ratio = volume / vma5
            if vol_ratio > 3.0:
                vol_score += 0.08
            elif vol_ratio > 2.5:
                vol_score += 0.06
            elif vol_ratio > 2.0:
                vol_score += 0.04
            elif vol_ratio > 1.5:
                vol_score += 0.02

        # 2b. 换手率
        if turnover > 20:
            vol_score += 0.06
        elif turnover > 15:
            vol_score += 0.04
        elif turnover > 10:
            vol_score += 0.02

        # 2c. 天量（创近20日新高）
        vol_20_max = safe(grp["volume"].tail(20).max(), volume)
        if volume >= vol_20_max * 0.95:
            vol_score += 0.04

        # 2d. 持续放量（连续3日量>5日均量1.2倍）
        if len(grp) >= 5:
            recent_3 = grp.tail(3)
            if all(recent_3["volume"] > recent_3["vma5"] * 1.2):
                vol_score += 0.03

        # 2e. 量价背离（价创新高但量萎缩）
        if close >= high_20d * 0.99 and volume < vma5 * 0.8:
            vol_score += 0.04

        vol_score = min(vol_score, 0.25)

        # ===== 维度3: K线形态判断（25%）看形态 =====
        shape_score = 0.0
        body = abs(close - open_p)
        upper_shadow = high - max(close, open_p)
        lower_shadow = min(close, open_p) - low
        total_range = high - low if high > low else 0.01

        # 3a. 长上影线（高位）
        if upper_shadow > body * 2 and upper_shadow / total_range > 0.5 and pos_score > 0.05:
            shape_score += 0.06
        elif upper_shadow > body * 1.5 and upper_shadow / total_range > 0.4 and pos_score > 0.05:
            shape_score += 0.03

        # 3b. 射击之星（高位小实体+长上影+小下影）
        if (body / total_range < 0.3 and upper_shadow > body * 2
                and lower_shadow < body * 0.5 and pos_score > 0.05):
            shape_score += 0.04

        # 3c. 吊颈线（高位小实体+长下影）
        if (body / total_range < 0.3 and lower_shadow > body * 2
                and upper_shadow < body * 0.5 and pos_score > 0.05 and close < open_p):
            shape_score += 0.04

        # 3d. 高位十字星
        if body / total_range < 0.1 and total_range > atr14 * 0.8 and pos_score > 0.05:
            shape_score += 0.04

        # 3e. 看跌吞没（高位阴线完全吞没前一日阳线）
        prev_open = safe(prev["open"])
        prev_close = safe(prev["close"])
        if (prev_close > prev_open and close < open_p  # 前阳后阴
                and open_p > prev_close and close < prev_open  # 完全吞没
                and pos_score > 0.05):
            shape_score += 0.05

        # 3f. 黄昏之星（3根K线：阳+十字星/小实体+阴）
        prev2_open = safe(prev2["open"])
        prev2_close = safe(prev2["close"])
        if (prev2_close > prev2_open  # 第一根阳
                and abs(safe(prev["close"]) - safe(prev["open"])) / max(safe(prev["high"]) - safe(prev["low"]), 0.01) < 0.3  # 第二根小实体
                and close < open_p and close < safe(prev["close"])  # 第三根阴
                and pos_score > 0.05):
            shape_score += 0.04

        # 3g. 连续阴线（高位连续2根阴线）
        if close < open_p and safe(prev["close"]) < safe(prev["open"]) and pos_score > 0.05:
            shape_score += 0.03

        # 3h. 跳空低开低走（高位）
        if open_p < safe(prev["low"]) and close < open_p and pos_score > 0.05:
            shape_score += 0.03

        shape_score = min(shape_score, 0.25)

        # ===== 维度4: 技术指标判断（15%）=====
        tech_score = 0.0
        # 4a. RSI超买
        if rsi > 85:
            tech_score += 0.04
        elif rsi > 80:
            tech_score += 0.03
        elif rsi > 75:
            tech_score += 0.02
        elif rsi > 70:
            tech_score += 0.01

        # 4b. KDJ超买/死叉
        if kdj_j > 110:
            tech_score += 0.02
        elif kdj_j > 100:
            tech_score += 0.01
        # KDJ死叉
        if kdj_k < kdj_d and prev_kdj_k >= prev_kdj_d:
            tech_score += 0.03

        # 4c. MACD顶背离（价格创新高，MACD不创新高）
        if len(grp) >= 30:
            # 找近30日价格高点和对应MACD
            recent_30 = grp.tail(30)
            price_high_idx = recent_30["close"].idxmax()
            if price_high_idx == len(grp) - 1:  # 当前是价格高点
                # 找前一个价格高点
                before = grp.iloc[:-5]
                if len(before) > 10:
                    prev_high_idx = before["close"].idxmax()
                    if (safe(grp.iloc[prev_high_idx]["macd_dif"]) > macd_dif
                            and close > safe(grp.iloc[prev_high_idx]["close"])):
                        tech_score += 0.04  # 顶背离

        # 4d. MACD死叉 + 红柱缩短
        if macd_dif < macd_dea and safe(prev["macd_dif"]) >= safe(prev["macd_dea"]):
            tech_score += 0.02
        if macd_hist < prev_macd_hist and macd_hist > 0:
            tech_score += 0.01

        # 4e. 布林带收口（高位收口=变盘信号）
        if pos_score > 0.05 and boll_width < prev_boll_width * 0.9:
            tech_score += 0.02

        tech_score = min(tech_score, 0.15)

        # ===== 维度5: 资金行为判断（15%）=====
        money_score = 0.0
        # 5a. 尾盘跳水（收盘价接近当日最低价）
        if total_range > 0:
            close_pos = (close - low) / total_range
            if close_pos < 0.2 and pct_chg < 0:
                money_score += 0.04
            elif close_pos < 0.3 and pct_chg < 0:
                money_score += 0.02

        # 5b. 冲高回落（最高价远高于收盘价）
        if high > close * 1.03 and pct_chg < 0.02:
            money_score += 0.03

        # 5c. 收盘价低于均价（用(high+low+close)/3模拟均价）
        avg_price = (high + low + close) / 3
        if close < avg_price and volume > vma5:
            money_score += 0.03

        # 5d. 连续资金流出（连续2日收盘价低于均价）
        if len(grp) >= 2:
            prev_avg = (safe(prev["high"]) + safe(prev["low"]) + safe(prev["close"])) / 3
            if safe(prev["close"]) < prev_avg and close < avg_price:
                money_score += 0.02

        # 5e. 放量收阴（量能放大+收阴线）
        if close < open_p and volume > vma5 * 1.3:
            money_score += 0.03

        money_score = min(money_score, 0.15)

        # ===== 综合评分 =====
        total_score = pos_score + vol_score + shape_score + tech_score + money_score
        total_score = min(total_score, 1.0)

        results.append({
            "code": code,
            "distribution_risk": round(total_score, 4),
            "dist_pos": round(pos_score, 4),
            "dist_vol": round(vol_score, 4),
            "dist_shape": round(shape_score, 4),
            "dist_tech": round(tech_score, 4),
            "dist_money": round(money_score, 4),
        })

    return pd.DataFrame(results)


def is_trading_hours(now: datetime) -> bool:
    """判断当前是否为A股交易时间：周一到周五 9:30-15:00（节假日由数据接口自然处理）"""
    if now.weekday() >= 5:  # 周六周日
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 15 * 60


def should_update_intraday(now: datetime) -> bool:
    """判断是否需要更新盘中实时数据（交易时间内，且距上次更新>=15分钟）"""
    if not is_trading_hours(now):
        return False
    if INTRADAY_CACHE_FILE.exists():
        try:
            with open(INTRADAY_CACHE_FILE, "r", encoding="utf-8") as f:
                last = datetime.fromisoformat(json.load(f)["last_update"])
            if (now - last).total_seconds() < INTRADAY_MIN_INTERVAL * 60:
                return False
        except Exception:
            pass
    return True


def save_intraday_update_time(now: datetime):
    """保存盘中实时数据最后更新时间"""
    try:
        INTRADAY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INTRADAY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_update": now.isoformat()}, f)
    except Exception:
        pass


def load_intraday_cache(now: datetime):
    """从文件加载盘中实时数据缓存，15分钟内有效。返回(DataFrame, is_valid)"""
    if not INTRADAY_DATA_CACHE.exists():
        return pd.DataFrame(), False
    try:
        mtime = datetime.fromtimestamp(INTRADAY_DATA_CACHE.stat().st_mtime)
        if (now - mtime).total_seconds() > INTRADAY_MIN_INTERVAL * 60:
            return pd.DataFrame(), False
        df = pd.read_parquet(INTRADAY_DATA_CACHE)
        if df.empty:
            return pd.DataFrame(), False
        return df, True
    except Exception:
        return pd.DataFrame(), False


def save_intraday_cache(df: pd.DataFrame):
    """保存盘中实时数据到文件缓存"""
    try:
        INTRADAY_DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(INTRADAY_DATA_CACHE, index=False)
    except Exception:
        pass


def fetch_intraday_data() -> pd.DataFrame:
    """获取当天实时行情（akshare新浪接口 stock_zh_a_spot，稳定可用）"""
    import akshare as ak
    import time as _time
    logger.info("获取当天实时行情数据(新浪接口)...")
    today = pd.Timestamp.now().normalize()

    for attempt in range(3):  # 重试3次
        try:
            spot = ak.stock_zh_a_spot()
            if spot is None or spot.empty:
                logger.warning(f"实时行情为空(第{attempt + 1}次)")
                if attempt < 2:
                    _time.sleep(1)
                    continue
                return pd.DataFrame()
            # 新浪接口列名映射（中文列名）
            col_map = {}
            for c in spot.columns:
                cl = str(c).strip()
                if cl == "代码":
                    col_map[c] = "code"
                elif cl == "今开":
                    col_map[c] = "open"
                elif cl == "最高":
                    col_map[c] = "high"
                elif cl == "最低":
                    col_map[c] = "low"
                elif cl == "最新价":
                    col_map[c] = "close"
                elif cl == "成交量":
                    col_map[c] = "volume"
                elif cl == "成交额":
                    col_map[c] = "amount"
                elif cl == "涨跌幅":
                    col_map[c] = "pct_chg"
            spot = spot.rename(columns=col_map)
            needed = ["code", "open", "high", "low", "close", "volume", "amount"]
            if not all(c in spot.columns for c in needed):
                logger.warning(f"列缺失，需要{needed}，实际{list(spot.columns)}")
                return pd.DataFrame()
            keep_cols = needed + (["pct_chg"] if "pct_chg" in spot.columns else [])
            spot = spot[keep_cols].copy()
            spot["date"] = today
            # 新浪接口代码格式为 sh600000/sz000001/bj920000，去掉交易所前缀，保留6位数字
            spot["code"] = spot["code"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                spot[c] = pd.to_numeric(spot[c], errors="coerce")
            spot = spot.dropna(subset=["close"])
            spot = spot[spot["close"] > 0]
            logger.info(f"获取到 {len(spot)} 只股票的当天实时行情")
            return spot
        except Exception as e:
            logger.warning(f"获取实时行情失败(第{attempt + 1}次): {e}")
            if attempt < 2:
                _time.sleep(1)

    logger.warning("实时行情获取失败，使用本地存量数据")
    return pd.DataFrame()


def merge_intraday_data(historical: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    """将当天实时行情合并到历史数据中（替换当天已有数据，追加新数据）"""
    if intraday.empty:
        return historical
    today = intraday["date"].iloc[0]
    # 移除历史数据中今天的记录
    hist = historical[historical["date"] != today].copy()
    # 追加今天的实时数据
    merged = pd.concat([hist, intraday], ignore_index=True)
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    logger.info(f"合并实时数据完成: 历史{len(historical)}行 + 当天{len(intraday)}行 = {len(merged)}行")
    return merged


def main():
    now = datetime.now()
    logger.info("=" * 70)
    logger.info("A股量化预测开始")
    logger.info(f"当前时间: {now:%Y-%m-%d %H:%M:%S}")

    # ========== Step 1: 数据更新（自动判断交易时间） ==========
    logger.info("\n" + "=" * 70)
    use_intraday = False
    intraday = pd.DataFrame()  # 缓存Step1获取的实时数据，避免Step2重复请求
    if is_trading_hours(now):
        if should_update_intraday(now):
            logger.info("Step 1: 交易时间内，更新最新实时数据")
            try:
                intraday = fetch_intraday_data()
                if not intraday.empty:
                    use_intraday = True
                    save_intraday_update_time(now)
                    save_intraday_cache(intraday)
                    # 向下取整到15分钟
                    minute_floor = (now.minute // 15) * 15
                    data_time_str = f"{now:%Y-%m-%d} {now.hour:02d}:{minute_floor:02d}"
                    logger.info(f"实时数据更新完成，数据时间: {data_time_str}")
                else:
                    logger.warning("实时行情为空，尝试加载缓存...")
                    intraday, cache_valid = load_intraday_cache(now)
                    if cache_valid:
                        use_intraday = True
                        logger.info(f"使用缓存的实时数据（{len(intraday)}只）")
            except Exception as e:
                logger.error(f"实时数据更新失败({e})，尝试加载缓存...")
                intraday, cache_valid = load_intraday_cache(now)
                if cache_valid:
                    use_intraday = True
                    logger.info(f"使用缓存的实时数据（{len(intraday)}只）")
        else:
            logger.info("Step 1: 交易时间内，距上次更新不足15分钟，从文件加载缓存")
            intraday, cache_valid = load_intraday_cache(now)
            if cache_valid:
                use_intraday = True
                logger.info(f"缓存有效，使用缓存的实时数据（{len(intraday)}只）")
            else:
                logger.warning("缓存失效，将在Step2重新获取")
                use_intraday = True
    else:
        logger.info("Step 1: 非交易时间，使用训练文件的最新存量数据（不更新）")

    # ========== Step 2: 加载数据并计算因子 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 加载数据并计算最新因子")
    try:
        all_data = load_all_data()
        data_start = all_data["date"].min()
        data_end = all_data["date"].max()
        if use_intraday:
            # 复用Step1已获取的实时数据，避免重复请求被限流
            if intraday.empty:
                logger.info("重新获取实时数据（Step1未缓存）...")
                intraday = fetch_intraday_data()
            if not intraday.empty:
                all_data = merge_intraday_data(all_data, intraday)
                data_end = intraday["date"].iloc[0]
                minute_floor = (now.minute // 15) * 15
                data_time_str = f"{now:%Y-%m-%d} {now.hour:02d}:{minute_floor:02d}"
                logger.info(f"本次预测基于实时数据: {data_start:%Y-%m-%d} ~ {data_time_str}")
            else:
                logger.info(f"本次预测基于存量数据: {data_start:%Y-%m-%d} ~ {data_end:%Y-%m-%d}")
        else:
            logger.info(f"本次预测基于存量数据: {data_start:%Y-%m-%d} ~ {data_end:%Y-%m-%d}")
        factor_data, factor_cols = compute_factors(all_data)
    except Exception as e:
        logger.error(f"数据加载或因子计算失败: {e}")
        return

    # 选择股票数量最多的最新交易日
    date_counts = factor_data.groupby("date")["code"].nunique()
    latest_date = date_counts[date_counts >= 1000].index.max()
    if pd.isna(latest_date):
        latest_date = date_counts.idxmax()
    latest_data = factor_data[factor_data["date"] == latest_date].copy()
    logger.info(f"预测基准日: {latest_date:%Y-%m-%d}, 可预测股票数: {len(latest_data)}")

    # 打印模型训练数据范围
    cp_path = MODEL_DIR / PREDICT_TAG / "_checkpoint.json"
    if cp_path.exists():
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            train_end = cp.get("train_end_date", "未知")
            logger.info(f"预测模型训练数据: 2016-01-01 ~ {train_end}（100%全量数据训练）")
        except Exception:
            pass

    # 市场择时状态（移到预测结果后展示，先给股票推荐，再给操作建议）
    market_advice = ""
    if "hs300_trend" in latest_data.columns:
        market_trend = latest_data["hs300_trend"].iloc[0]
        if market_trend < -0.05:
            market_advice = f"⚠ 空仓建议：沪深300低于20日均线 {market_trend * 100:.1f}%（极端熊市，建议空仓观望）"
        else:
            market_advice = f"✓ 正常持仓：沪深300高于20日均线 {market_trend * 100:.1f}%（可正常买Top3）"

    # ========== Step 3: 加载模型并预测 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: 加载模型并预测")
    X = latest_data[factor_cols].values

    predictions = {}
    confidences = {}
    available_horizons = []
    for h in HORIZONS:
        try:
            model = EnsembleModel(horizon=h)
            model.load(tag=PREDICT_TAG)
            pred = model.predict(X)
            predictions[h] = pred
            confidences[h] = compute_confidence(pred)
            available_horizons.append(h)
            logger.info(
                f"1天持仓-动态卖出 ({h}) 预测完成，置信度范围: [{confidences[h].min():.4f}, {confidences[h].max():.4f}]")
        except Exception as e:
            logger.error(f"1天持仓-动态卖出 ({h}) 模型加载/预测失败: {e}，跳过该模式")

    if not available_horizons:
        logger.error("没有可用的模型，请先运行 python train.py")
        return

    # 排序：按动态卖出(1d_close)置信度降序
    sort_priority = [h for h in ["1d_close"] if h in available_horizons]
    sort_cols = [f"confidence_{h}" for h in sort_priority]

    # ========== Step 4: 组装结果 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 组装预测结果")

    # 加载股票名称映射
    name_map = {}
    try:
        stock_list_path = RAW_DIR / "_stock_list.csv"
        if stock_list_path.exists():
            sl = pd.read_csv(stock_list_path, dtype={"code": str})
            name_map = dict(zip(sl["code"], sl["name"]))
            logger.info(f"加载股票名称映射: {len(name_map)} 只")
    except Exception as e:
        logger.warning(f"股票名称映射加载失败: {e}")

    result = latest_data[["code", "close"]].copy()
    result["name"] = result["code"].map(name_map).fillna("")
    result["date"] = latest_date

    # 最新涨跌幅：盘中用实时数据，非盘中用历史数据
    if use_intraday and "pct_chg" in all_data.columns:
        intraday_today = all_data[all_data["date"] == latest_date][["code", "pct_chg"]].copy()
        result = result.merge(intraday_today, on="code", how="left")
        result["latest_pct_chg"] = result["pct_chg"].fillna(0) / 100.0  # 新浪接口是百分比数值
        result = result.drop(columns=["pct_chg"])
    elif "ret_1d_raw" in latest_data.columns:
        result["latest_pct_chg"] = latest_data["ret_1d_raw"].values
    else:
        result["latest_pct_chg"] = 0.0

    # 涨停标记：主板涨停10%，>=9.9%视为涨停（考虑四舍五入）
    result["is_limit_up"] = result["latest_pct_chg"] >= 0.099
    result["limit_status"] = result["is_limit_up"].map({True: "涨停", False: "可买入"})

    # 主力出货置信度（0-1，越高越可能出货）
    try:
        dist_risk = compute_distribution_risk(all_data, latest_date)
        result = result.merge(dist_risk, on="code", how="left")
        result["distribution_risk"] = result["distribution_risk"].fillna(0.0)
    except Exception as e:
        logger.warning(f"出货置信度计算失败: {e}")
        result["distribution_risk"] = 0.0

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
    # 出货置信度保留4位小数
    if "出货置信度" in display.columns:
        display["出货置信度"] = display["出货置信度"].round(4)
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
        logger.info(f"  {i}. {row['code']} {row['name']}  收盘价:{row['close']:.2f}  涨跌幅:{pct_str}{limit_str}  {conf_str}")

    # 市场操作建议（在股票推荐之后给出）
    if market_advice:
        logger.info("\n" + "=" * 70)
        logger.info(f"市场操作建议: {market_advice}")

    logger.info("\n预测完成！")


if __name__ == "__main__":
    main()
