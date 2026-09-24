"""
因子工程模块：
- 技术指标因子（均线、MACD、RSI、布林带、KDJ）
- 动量因子（多周期收益率）
- 成交量因子（量比、均量）
- 波动率因子
- 横截面标准化
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import get_logger, safe_div

logger = get_logger("factors")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = safe_div(avg_gain, avg_loss)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = _ema(close, fast) - _ema(close, slow)
    dea = _ema(dif, signal)
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def _bollinger(close: pd.Series, period=20, num_std=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pctb = safe_div(close - lower, upper - lower)
    width = safe_div(upper - lower, mid)
    return upper, mid, lower, pctb, width


def _kdj(high, low, close, n=9, m1=3, m2=3):
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = safe_div(close - low_n, high_n - low_n) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def compute_factors_single(df: pd.DataFrame) -> pd.DataFrame:
    """对单只股票计算所有因子，df需按date升序"""
    df = df.copy().sort_values("date").reset_index(drop=True)
    # 显式转为pandas Series（兼容pandas 3.x）
    close = pd.Series(df["close"].values, index=df.index, dtype=float)
    high = pd.Series(df["high"].values, index=df.index, dtype=float)
    low = pd.Series(df["low"].values, index=df.index, dtype=float)
    open_ = pd.Series(df["open"].values, index=df.index, dtype=float)
    volume = pd.Series(df["volume"].values, index=df.index, dtype=float)
    if "amount" in df.columns:
        amount = pd.Series(df["amount"].values, index=df.index, dtype=float)
    else:
        amount = volume * close
    if "turnover" in df.columns:
        turnover = pd.Series(df["turnover"].values, index=df.index, dtype=float)
    else:
        turnover = pd.Series(0.0, index=df.index)

    # ===== 动量/收益率 =====
    for n in [1, 3, 5, 10, 20, 60]:
        df[f"ret_{n}d"] = close.pct_change(n)

    # 对数收益率
    df["log_ret_1d"] = np.log(close / close.shift(1))

    # ===== 均线 =====
    for n in [5, 10, 20, 60]:
        df[f"ma{n}"] = close.rolling(n).mean()
        df[f"close_ma{n}_ratio"] = safe_div(close, df[f"ma{n}"])
    df["ma5_ma20_ratio"] = safe_div(df["ma5"], df["ma20"])
    df["ma10_ma60_ratio"] = safe_div(df["ma10"], df["ma60"])

    # ===== MACD =====
    df["macd_dif"], df["macd_dea"], df["macd_hist"] = _macd(close)

    # ===== RSI =====
    df["rsi_6"] = _rsi(close, 6)
    df["rsi_12"] = _rsi(close, 12)
    df["rsi_24"] = _rsi(close, 24)

    # ===== 布林带 =====
    df["boll_upper"], df["boll_mid"], df["boll_lower"], df["boll_pctb"], df["boll_width"] = _bollinger(close)

    # ===== KDJ =====
    df["kdj_k"], df["kdj_d"], df["kdj_j"] = _kdj(high, low, close)

    # ===== 成交量 =====
    df["vol_ma5"] = volume.rolling(5).mean()
    df["vol_ma10"] = volume.rolling(10).mean()
    df["vol_ratio_5"] = safe_div(volume, df["vol_ma5"])
    df["vol_ratio_10"] = safe_div(volume, df["vol_ma10"])
    df["amount_ma5"] = amount.rolling(5).mean()
    df["amount_ratio"] = safe_div(amount, df["amount_ma5"])

    # ===== 波动率 =====
    for n in [5, 10, 20]:
        df[f"volatility_{n}d"] = df["log_ret_1d"].rolling(n).std() * np.sqrt(252)

    # ===== 振幅与位置 =====
    df["amplitude_1d"] = safe_div(high - low, close.shift(1))
    df["amplitude_5d"] = (high.rolling(5).max() - low.rolling(5).min()) / close.shift(5)
    df["close_position_5d"] = safe_div(close - low.rolling(5).min(),
                                       high.rolling(5).max() - low.rolling(5).min())
    df["close_position_20d"] = safe_div(close - low.rolling(20).min(),
                                        high.rolling(20).max() - low.rolling(20).min())
    df["high_low_ratio"] = safe_div(high, low)

    # ===== 换手率 =====
    df["turnover_ma5"] = turnover.rolling(5).mean()
    df["turnover_ratio"] = safe_div(turnover, df["turnover_ma5"])

    # ===== 跳空 =====
    df["gap_ratio"] = safe_div(open_ - close.shift(1), close.shift(1))

    # ===== 趋势强度 =====
    df["trend_strength_20d"] = safe_div(close - close.rolling(20).min(),
                                        close.rolling(20).max() - close.rolling(20).min())

    # ===== 资金流代理因子（baostock无真实资金流数据，用量价关系构造） =====
    # 1. 资金净流入代理：涨且放量=资金流入，跌且放量=资金流出
    df["mf_net_proxy"] = np.sign(df["ret_1d"]) * df["amount_ratio"]

    # 2. 日内资金流向：收盘在日内区间的位置 × 成交额（收高=买方强势）
    hl_range = high - low
    df["mf_intraday"] = safe_div(close - low, hl_range.replace(0, np.nan)) * amount

    # 3. 量价背离：涨但缩量=假突破风险（正值=量价配合好，负值=背离）
    df["vol_price_div"] = df["ret_1d"] * df["vol_ratio_5"]

    # 4. OBV代理：累计(sign(涨跌) × 成交量)
    obv = (np.sign(df["log_ret_1d"]) * volume).cumsum()
    df["obv_ma5_ratio"] = safe_div(obv, obv.rolling(5).mean())

    # 5. 上攻力度：(close-open)/open × 量比（实体涨幅 × 放量程度）
    df["buying_strength"] = safe_div(close - open_, open_) * df["vol_ratio_5"]

    return df


def compute_factors(all_data: pd.DataFrame) -> tuple:
    """
    对全市场数据计算因子（内存优化版）。
    - float32降精度
    - 丢弃冗余OHLCV列
    - 向量化横截面标准化
    - 分块处理避免内存峰值
    """
    import gc
    logger.info("开始计算因子...")

    # code转为category节省内存
    if all_data["code"].dtype != "category":
        all_data["code"] = all_data["code"].astype("category")

    codes = all_data["code"].unique()
    result_parts = []

    # 分块处理，每500只股票一批，及时释放内存
    chunk_size = 500
    for chunk_start in tqdm(range(0, len(codes), chunk_size), desc="因子分块", ncols=100):
        chunk_codes = codes[chunk_start:chunk_start + chunk_size]
        chunk_parts = []
        for code in chunk_codes:
            stock_df = all_data[all_data["code"] == code]
            if len(stock_df) < 60:
                continue
            try:
                factor_df = compute_factors_single(stock_df)
                # 立即降为float32
                for c in factor_df.columns:
                    if c not in ["date", "code"] and factor_df[c].dtype == "float64":
                        factor_df[c] = factor_df[c].astype("float32")
                # 只保留必要列：date, code, open, high, low, close + 因子（high/low用于回测盘中止盈止损）
                keep = ["date", "code", "open", "high", "low", "close"] + [
                    c for c in factor_df.columns if c not in
                                                    ["date", "code", "open", "high", "low", "close", "volume",
                                                     "amount", "amplitude", "pct_chg", "change", "turnover"]
                ]
                chunk_parts.append(factor_df[keep])
            except Exception as e:
                logger.warning(f"股票 {code} 因子计算失败: {e}")
        if chunk_parts:
            result_parts.append(pd.concat(chunk_parts, ignore_index=True))
        del chunk_parts
        gc.collect()

    if not result_parts:
        raise RuntimeError("因子计算失败，无有效数据")

    result = pd.concat(result_parts, ignore_index=True)
    del result_parts
    gc.collect()

    # 排序（此时已丢弃冗余列，内存占用低）
    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    # 去掉前60行（因子预热期）
    result["_cumcnt"] = result.groupby("code").cumcount()
    result = result[result["_cumcnt"] >= 60].drop(columns=["_cumcnt"]).reset_index(drop=True)

    # 识别因子列（ret_1d_raw保留原始收益率不标准化，用于涨停判断）
    result["ret_1d_raw"] = result["ret_1d"]
    factor_cols = [c for c in result.columns if
                   c not in ["date", "code", "open", "high", "low", "close", "ret_1d_raw", "high_low_ratio"]]
    logger.info(f"共 {len(factor_cols)} 个因子，{len(result)} 行，进行横截面标准化...")

    # 因子去冗余：相关性>0.95的因子只保留第一个（减少过拟合）
    logger.info("因子去冗余：计算因子间相关性矩阵...")
    corr_matrix = result[factor_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop:
        logger.info(f"剔除 {len(to_drop)} 个高相关冗余因子: {to_drop}")
        factor_cols = [c for c in factor_cols if c not in to_drop]
        result = result.drop(columns=to_drop, errors="ignore")
    else:
        logger.info("无高相关冗余因子")

    # 向量化横截面标准化：先算每日mean/std，再merge（比transform省内存）
    date_group = result.groupby("date")[factor_cols]
    means = date_group.transform("mean")
    stds = date_group.transform("std").replace(0, 1e-8)
    result[factor_cols] = (result[factor_cols] - means) / stds
    del means, stds
    gc.collect()

    # 极端值截断
    result[factor_cols] = result[factor_cols].clip(-5, 5)
    # 填充NaN
    result[factor_cols] = result[factor_cols].fillna(0)

    # ========== 增加市场状态特征（沪深300，不参与横截面标准化） ==========
    try:
        from .data_loader import load_hs300
        hs300 = load_hs300()
        if hs300 is not None and not hs300.empty:
            hs300 = hs300.sort_values("date").reset_index(drop=True)
            hs300["hs300_ret_1d"] = hs300["close"].pct_change(1)
            hs300["hs300_ret_5d"] = hs300["close"].pct_change(5)
            hs300["hs300_ret_20d"] = hs300["close"].pct_change(20)
            hs300["hs300_ma20"] = hs300["close"].rolling(20).mean()
            hs300["hs300_trend"] = (hs300["close"] / hs300["hs300_ma20"] - 1)
            hs300["hs300_volatility_5d"] = hs300["hs300_ret_1d"].rolling(5).std()
            market_cols = ["date", "hs300_ret_1d", "hs300_ret_5d", "hs300_ret_20d",
                           "hs300_trend", "hs300_volatility_5d"]
            hs300_feat = hs300[market_cols].copy()
            for c in market_cols[1:]:
                hs300_feat[c] = hs300_feat[c].astype("float32")
            result = result.merge(hs300_feat, on="date", how="left")
            for c in market_cols[1:]:
                result[c] = result[c].fillna(0)
                # 市场状态特征不加入factor_cols（每天所有股票相同，对横截面选股无帮助，仅用于回测择时）
            logger.info(f"已加入市场状态特征(仅择时用): {market_cols[1:]}")
    except Exception as e:
        logger.warning(f"市场状态特征加载失败: {e}")

    # ========== 增加关键因子横截面排名（排名比绝对值更稳定） ==========
    rank_features = ["ret_5d", "ret_20d", "vol_ratio_5", "turnover_ratio", "rsi_12"]
    for feat in rank_features:
        if feat in result.columns:
            rank_col = f"{feat}_rank"
            result[rank_col] = result.groupby("date")[feat].rank(pct=True).astype("float32")
            result[rank_col] = result[rank_col].fillna(0.5)
            factor_cols.append(rank_col)

    logger.info(f"因子计算完成: {len(result)} 行, {len(factor_cols)} 个因子")
    return result, factor_cols


def create_labels(factor_data: pd.DataFrame, horizons: list = None,
                  take_profit: float = 0.005, intraday_take_profit: float = 0.045) -> pd.DataFrame:
    """
    创建标签：T日预测，T日尾盘买入，1天后卖出的收益率。
    卖出逻辑与 backtest.py 完全一致：
      1. T+1开盘高开≥take_profit → 开盘价止盈卖出
      2. T+1盘中最高价≥intraday_take_profit → 按止盈价卖出
      3. 否则 → T+1收盘价卖出（不止损）
    默认参数与 backtest.py 当前写死值一致：开盘止盈0.5%，盘中止盈4.5%。
    """
    if horizons is None:
        horizons = ["1d_close"]
    df = factor_data

    buy_price = df["close"]
    g = df.groupby("code", sort=False)

    # 预计算T+1开盘价、收盘价、最高价
    open_d1 = g["open"].shift(-1)
    close_d1 = g["close"].shift(-1)
    high_d1 = g["high"].shift(-1)

    for mode in horizons:
        open_gap = open_d1 / buy_price - 1
        high_ret = high_d1 / buy_price - 1
        # 默认收盘价卖出
        sell_price = close_d1.copy()
        # 1. 开盘止盈：高开≥0.5%按开盘价卖
        tp_mask = open_gap >= take_profit
        sell_price[tp_mask] = open_d1[tp_mask]
        # 2. 盘中止盈：盘中最高价≥4.5%按4.5%卖（未被开盘止盈触发的情况下）
        itp_mask = (~tp_mask) & (high_ret >= intraday_take_profit)
        sell_price[itp_mask] = buy_price[itp_mask] * (1 + intraday_take_profit)
        total_ret = sell_price / buy_price - 1
        df[f"label_ret_{mode}"] = total_ret.astype("float32")
        df[f"label_abs_ret_{mode}"] = total_ret.astype("float32")

    # 去掉没有未来标签的行（最后1行）- 使用反向cumcount
    df["_rev_cnt"] = df.groupby("code", sort=False).cumcount(ascending=False)
    df = df[df["_rev_cnt"] >= 1].drop(columns=["_rev_cnt"]).reset_index(drop=True)
    df = df.dropna(subset=[f"label_ret_{m}" for m in horizons])

    logger.info(f"标签创建完成，卖出方式 {horizons}，有效样本 {len(df)}")
    return df
