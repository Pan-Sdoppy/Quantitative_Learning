"""
回测引擎（1天持仓策略，动态卖出）：
- 1d_close: T日模型预测→粗筛pred Top10候选→composite二次排序→取Top3（实际Top1全仓 [1,0,0]），T日尾盘买入
- 动态卖出：T+1开盘高开达分段阈值则开盘止盈（pred<1%→1.5%、pred≥1%→3%），否则盘中触及4%止盈，
  均未触及则收盘卖出；不止损（经完整回测验证，止损会切掉强势股的次日修复）
- 每交易日调仓，扣交易成本0.35%/次；剔除T日涨停（尾盘封单买不进）与T日涨幅>9%（追高过滤）
- 弱信号空仓：二次排序后Top1的pred < min_pred(1.0%) 当日空仓（提升夏普、大幅降低回撤）
- 市场择时：沪深300低于20日均线5%以上空仓（极端熊市保护）
- composite二次排序：5基础因子(振幅/5日量比/跳空/5日位置/3日动量) + K线形态(上影线/光头阳)
  + 3因子(流通市值对数/10日量比/日内资金流)，均为rank累加，score越小越靠前
- 计算累计收益、年化、夏普、最大回撤、胜率、年度收益，输出回测报告与净值曲线
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import RESULT_DIR, get_logger

# === K线形态因子（2022-01起完整样本外回测验证，R2简化配置）===
# 仅保留两个有稳定正增益的形态因子；其余形态/连涨/上榜因子经完整回测验证为负向或无效，已删除：
#   - 开盘价位置 OPEN_POS_W 是 2024+ 过拟合（2022起全样本失效），删除
#   - 实体强度 BODY_STR_W 不稳定、连涨 STREAK_UP_W 零作用、连续上榜 STREAK_BONUS 关闭后略好，删除
KLINE_SHADOW_W = 1.0  # 上影线短加分权重（短上影=无冲高回落，回测验证有效）
KLINE_KFLAG_W = 0.5  # 光头阳加分/光头阴减分权重（半权重，全权重扭曲排序）
KLINE_BODY_W = 0.0  # 实体强度：叠加后负向/无增益，保持关闭
KLINE_HIGH20_W = 0.0  # 20日高位回避：负向，保持关闭

# === composite 二次排序新增因子（小值优先，2022-01起严格样本外回测验证为正增益）===
# 三者方向均为"越小越靠前"，与基础缩量/形态因子共线度低，权重各 1（0.5~2 扫描不敏感，1:1:1 最优）
RERANK_MV_W = 1.0  # 流通市值对数 mv_log_yi（小市值次日溢价显著）
RERANK_VOL10_W = 1.0  # 10日量比 vol_ratio_10（缩量企稳，比5日量比更稳）
RERANK_MF_W = 1.0  # 日内资金流 mf_intraday（收盘位置×成交额，规避高位放量收盘）

# === 板块涨停集中度因子（实验开关，默认0=关闭）===
# 候选股所属板块当天涨停数量越多→越靠前（涨停集中板块强势，其所属可买个股次日有望补涨）
# 需行业归属表 data/raw/_stock_industry.csv，仅当权重>0时才加载行业并计算
SECTOR_LIMITUP_W = float(os.environ.get("SECTOR_LIMITUP_W", "0"))

logger = get_logger("backtest")

TRADING_DAYS_PER_YEAR = 252
# A股单次调仓交易成本：佣金万2.5(买卖)+印花税千1(卖出)+滑点千1(买卖) ≈ 0.35%
DEFAULT_TRANSACTION_COST = 0.0035
# 主板涨停阈值（10%，考虑四舍五入用9.5%）
LIMIT_UP_THRESHOLD = 0.095


def run_backtest(factor_data: pd.DataFrame, models: dict,
                 top_n: int = 3, horizons: list = None,
                 start_date: str = None, end_date: str = None,
                 transaction_cost: float = DEFAULT_TRANSACTION_COST,
                 take_profit: float = 0.01,
                 stop_loss: float = -0.05,
                 filter_today_body_ret: float = None,
                 filter_today_day_ret: float = None,
                 filter_today_day_ret_min: float = None,
                 filter_yin_line: bool = False,
                 filter_upper_shadow: float = None,
                 intraday_take_profit: float = None,
                 weights: list = None,
                 filter_min_mv: float = None,
                 only_prev_limit_up: bool = False,
                 prev_limit_up_open_buy: bool = False,
                 skip_if_index_down: bool = False,
                 filter_amplitude_max: float = None,
                 filter_vol_ratio_max: float = None,
                 rerank_candidates: int = None,
                 rerank_method: str = None,
                 rerank_weights: list = None,
                 min_pred: float = None,
                 index_5d_empty: float = None,
                 dynamic_tp: tuple = None) -> dict:
    """运行动态卖出回测（1d_close）
    take_profit: 高开止盈阈值（如0.02=高开2%开盘卖），None=无止盈
    stop_loss: 止损阈值（如-0.02=跌2%止损），None=不止损
    filter_today_body_ret: 方案一，T日实体涨幅(close-open)/open超过此阈值则剔除，None=不过滤
    filter_today_day_ret: 方案二，T日日涨幅(ret_1d_raw)超过此阈值则剔除，None=不过滤
    filter_today_day_ret_min: T日日涨幅(ret_1d_raw)低于此阈值则剔除（剔除暴跌股），None=不过滤
    filter_yin_line: 只买T日阴线收盘股（close<open），False=不过滤
    filter_upper_shadow: T日上影线/close超过此阈值则剔除（冲高回落），None=不过滤
    intraday_take_profit: 盘中最高价达到此阈值则止盈卖出（如0.01=盘中涨1%止盈），None=不启用
    weights: TopN仓位权重列表（如[0.5,0.3,0.2]），None=用默认（Top3全仓Top1）
    filter_min_mv: 剔除流通市值低于此阈值（亿元）的小票，None=不剔除
    only_prev_limit_up: 只买前一天涨停且当天非涨停（开板）的股票（打板次日接力策略）
    prev_limit_up_open_buy: T日涨停Top1→T+1开盘价买入→T+2动态卖出（打板次日开盘接力）
    skip_if_index_down: 大盘（沪深300）当天收阴线则空仓不买
    filter_amplitude_max: 当日振幅(amplitude_1d z-score)超过此值则剔除，None=不过滤
    filter_vol_ratio_max: 5日量比(vol_ratio_5 z-score)超过此值则剔除，None=不过滤
    rerank_candidates: 先从全市场选TopN候选再二次排序；None=不启用
    rerank_method: 二次排序方式 amplitude_asc/vol_ratio_asc/gap_desc/composite
    rerank_weights: composite基础5因子权重 [振幅,5日量比,跳空,5日位置,3日动量]，None=[2,2,1,1,1]
    """
    if horizons is None:
        horizons = ["1d_close"]

    factor_data = factor_data.copy()
    factor_data["date"] = pd.to_datetime(factor_data["date"])

    # 板块涨停集中度因子：仅在启用时加载行业归属并合并（默认不加载，不影响基线内存/性能）
    if SECTOR_LIMITUP_W > 0:
        from pathlib import Path
        industry_path = Path(__file__).parent.parent / "data" / "raw" / "_stock_industry.csv"
        try:
            ind_df = pd.read_csv(industry_path, usecols=["code", "industry"])
            ind_df["code"] = ind_df["code"].astype(str).str.zfill(6)
            ind_df = ind_df[["code", "industry"]].drop_duplicates("code")
            factor_data["code"] = factor_data["code"].astype(str).str.zfill(6)
            factor_data = factor_data.merge(ind_df, on="code", how="left")
            factor_data["industry"] = factor_data["industry"].fillna("")
            del ind_df
            logger.info(f"板块涨停集中度因子启用：行业归属加载完成")
        except Exception as _e:
            logger.warning(f"行业归属加载失败，板块涨停因子不生效: {_e}")

    if start_date:
        factor_data = factor_data[factor_data["date"] >= pd.Timestamp(start_date)]
    if end_date:
        factor_data = factor_data[factor_data["date"] <= pd.Timestamp(end_date)]

    results = {}
    for h in horizons:
        logger.info(f"{'=' * 60}")
        tp_str = f"，止盈{take_profit * 100:.1f}%" if take_profit else ""
        sl_str = f"，止损{stop_loss * 100:.1f}%" if stop_loss else ""
        filt_str = ""
        if filter_today_body_ret is not None:
            filt_str += f"，T日实体涨幅>{filter_today_body_ret * 100:.1f}%剔除"
        if filter_today_day_ret is not None:
            filt_str += f"，T日日涨幅>{filter_today_day_ret * 100:.1f}%剔除"
        if filter_today_day_ret_min is not None:
            filt_str += f"，T日日涨幅<{filter_today_day_ret_min * 100:.1f}%剔除"
        if filter_yin_line:
            filt_str += "，只买阴线"
        if filter_upper_shadow is not None:
            filt_str += f"，上影线>{filter_upper_shadow * 100:.1f}%剔除"
        if intraday_take_profit is not None:
            filt_str += f"，盘中止盈{intraday_take_profit * 100:.1f}%"
        if filter_min_mv is not None:
            filt_str += f"，市值>={filter_min_mv}亿"
        if only_prev_limit_up:
            filt_str += "，仅前一天涨停"
        if prev_limit_up_open_buy:
            filt_str += "，打板次日开盘接力"
        if skip_if_index_down:
            filt_str += "，大盘阴线空仓"
        if filter_amplitude_max is not None:
            filt_str += f"，振幅<={filter_amplitude_max}"
        if filter_vol_ratio_max is not None:
            filt_str += f"，量比<={filter_vol_ratio_max}"
        if rerank_candidates is not None:
            filt_str += f"，二次排序Top{rerank_candidates}/{rerank_method}"
        logger.info(f"开始回测: 1天持仓-动态卖出 ({h}){tp_str}{sl_str}, Top{top_n}{filt_str}")
        res = _backtest_single_mode(factor_data, models[h], h, top_n, transaction_cost,
                                    take_profit, stop_loss,
                                    filter_today_body_ret, filter_today_day_ret,
                                    filter_today_day_ret_min,
                                    filter_yin_line, filter_upper_shadow,
                                    intraday_take_profit,
                                    weights,
                                    filter_min_mv,
                                    only_prev_limit_up,
                                    prev_limit_up_open_buy,
                                    skip_if_index_down,
                                    filter_amplitude_max,
                                    filter_vol_ratio_max,
                                    rerank_candidates,
                                    rerank_method,
                                    rerank_weights,
                                    min_pred,
                                    index_5d_empty,
                                    dynamic_tp)
        results[h] = res
        _print_metrics(h, res["metrics"], take_profit, stop_loss)

    return results


def _backtest_single_mode(factor_data: pd.DataFrame, model, mode: str, top_n: int,
                          transaction_cost: float = DEFAULT_TRANSACTION_COST,
                          take_profit: float = 0.02,
                          stop_loss: float = -0.02,
                          filter_today_body_ret: float = None,
                          filter_today_day_ret: float = None,
                          filter_today_day_ret_min: float = None,
                          filter_yin_line: bool = False,
                          filter_upper_shadow: float = None,
                          intraday_take_profit: float = None,
                          weights: list = None,
                          filter_min_mv: float = None,
                          only_prev_limit_up: bool = False,
                          prev_limit_up_open_buy: bool = False,
                          skip_if_index_down: bool = False,
                          filter_amplitude_max: float = None,
                          filter_vol_ratio_max: float = None,
                          rerank_candidates: int = None,
                          rerank_method: str = None,
                          rerank_weights: list = None,
                          min_pred: float = None,
                          index_5d_empty: float = None,
                          dynamic_tp: tuple = None) -> dict:
    """单模式回测（1d_close 动态止盈+止损）
    take_profit: 高开止盈阈值（如0.02=高开2%开盘卖），None=无止盈
    stop_loss: 止损阈值（如-0.02=跌2%止损），None=不止损
    filter_today_body_ret: 方案一，T日实体涨幅(close-open)/open超过此阈值则剔除
    filter_today_day_ret: 方案二，T日日涨幅(ret_1d_raw)超过此阈值则剔除
    filter_today_day_ret_min: T日日涨幅(ret_1d_raw)低于此阈值则剔除（剔除暴跌股）
    filter_yin_line: 只买T日阴线收盘股（close<open）
    filter_upper_shadow: T日上影线/close超过此阈值则剔除（冲高回落）
    intraday_take_profit: 盘中最高价达到此阈值则止盈卖出，None=不启用
    weights: TopN仓位权重列表（如[0.5,0.3,0.2]），None=用默认（Top3全仓Top1）
    filter_min_mv: 剔除流通市值低于此阈值（亿元）的小票，None=不剔除
    only_prev_limit_up: 只买前一天涨停且当天非涨停（开板）的股票（打板次日接力策略）
    prev_limit_up_open_buy: T日涨停Top1→T+1开盘价买入→T+2动态卖出（打板次日开盘接力）
    skip_if_index_down: 大盘（沪深300）当天收阴线则空仓不买
    filter_amplitude_max: 当日振幅(amplitude_1d z-score)超过此值则剔除（低振幅过滤），None=不过滤
    filter_vol_ratio_max: 5日量比(vol_ratio_5 z-score)超过此值则剔除（缩量过滤），None=不过滤
    rerank_candidates: 先从全市场选TopN候选，再二次排序取TopN；None=不启用二次排序
    rerank_method: 二次排序方式 amplitude_asc/vol_ratio_asc/gap_desc/composite
    rerank_weights: composite基础5因子权重 [振幅,5日量比,跳空,5日位置,3日动量]，None=[2,2,1,1,1]；
                    另有K线形态(KLINE_SHADOW/KFLAG)与规模/量能/资金流3因子(RERANK_MV/VOL10/MF)由常量控制
    """
    feature_cols = [c for c in (model.feature_names or []) if c is not None and c in factor_data.columns]
    if not feature_cols:
        raise ValueError("模型特征列与数据不匹配")

    # 二次排序：手工composite规则（学习型reranker经A/B测试为负向，已删除）

    all_dates = sorted(factor_data["date"].unique())
    # 1天持仓，每个交易日调仓
    rebalance_dates = all_dates
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # 预计算T+1开盘价、收盘价和最低价
    logger.info("预计算T+1价格矩阵（T日尾盘买入）...")
    factor_data = factor_data.sort_values(["code", "date"]).reset_index(drop=True)
    g = factor_data.groupby("code", sort=False)
    factor_data["_open_d1"] = g["open"].shift(-1)
    factor_data["_close_d1"] = g["close"].shift(-1)
    factor_data["_low_d1"] = g["low"].shift(-1)
    factor_data["_high_d1"] = g["high"].shift(-1)
    factor_data["_open_d2"] = g["open"].shift(-2)
    factor_data["_close_d2"] = g["close"].shift(-2)
    factor_data["_low_d2"] = g["low"].shift(-2)
    factor_data["_high_d2"] = g["high"].shift(-2)

    # 前一天涨停标记（T-1日涨停，T日可能开板可买）
    factor_data["_prev_limit_up"] = (g["ret_1d_raw"].shift(1) >= LIMIT_UP_THRESHOLD).astype(float)

    # 市值过滤：从原始数据加载流通市值（亿）= amount / (turnover/100) / 1e8
    if filter_min_mv is not None:
        logger.info(f"加载流通市值数据（剔除<{filter_min_mv}亿小票）...")
        from pathlib import Path
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
        mv_parts = []
        for fp_raw in raw_dir.glob("*.parquet"):
            try:
                d = pd.read_parquet(fp_raw, columns=["date", "amount", "turnover"])
                d["code"] = fp_raw.stem
                d = d[d["turnover"] > 0]
                d["_mv_yi"] = d["amount"] / (d["turnover"] / 100.0) / 1e8
                mv_parts.append(d[["date", "code", "_mv_yi"]])
            except Exception:
                continue
        if mv_parts:
            mv_df = pd.concat(mv_parts, ignore_index=True)
            mv_df["date"] = pd.to_datetime(mv_df["date"])
            mv_df["code"] = mv_df["code"].astype(str).str.zfill(6)
            factor_data["code"] = factor_data["code"].astype(str).str.zfill(6)
            factor_data = factor_data.merge(mv_df, on=["date", "code"], how="left")
            factor_data["_mv_yi"] = factor_data["_mv_yi"].fillna(9999)  # 无市值数据的不剔除
            del mv_df, mv_parts
        else:
            logger.warning("未找到原始市值数据，跳过市值过滤")
            filter_min_mv = None

    # 市场择时：沪深300低于20日均线5%以上空仓（仅极端熊市）
    MARKET_TIMING_EMPTY = -0.05
    has_market_feature = "hs300_trend" in factor_data.columns

    equity = 1.0
    trades = []
    stock_trades = []
    equity_points = []
    skipped_empty = 0
    skipped_filter = 0

    pbar_desc = "回测 动态卖出"

    for rebal_date in tqdm(rebalance_dates, desc=pbar_desc, ncols=100):
        day_data = factor_data[factor_data["date"] == rebal_date].copy()
        if len(day_data) < top_n:
            continue

        # 市场择时：极端熊市空仓
        if has_market_feature:
            market_trend = day_data["hs300_trend"].iloc[0]
            if market_trend < MARKET_TIMING_EMPTY:
                skipped_empty += 1
                equity_points.append({"date": rebal_date, "equity": equity})
                continue

        # 大盘当天收阴线则空仓（沪深300当日收益<0）
        if skip_if_index_down and "hs300_ret_1d" in day_data.columns:
            idx_ret = day_data["hs300_ret_1d"].iloc[0]
            if idx_ret < 0:
                skipped_empty += 1
                equity_points.append({"date": rebal_date, "equity": equity})
                continue
        # 大盘5日收益低于阈值则空仓（市场状态过滤，None=不启用）
        if index_5d_empty is not None and "hs300_ret_5d" in day_data.columns:
            if day_data["hs300_ret_5d"].iloc[0] < index_5d_empty:
                skipped_empty += 1
                equity_points.append({"date": rebal_date, "equity": equity})
                continue

        # 板块涨停集中度：计算当天每个板块涨停股票数（涨停集中板块→强势）
        if SECTOR_LIMITUP_W > 0 and "industry" in day_data.columns and "ret_1d_raw" in day_data.columns:
            _lim_mask = day_data["ret_1d_raw"] >= LIMIT_UP_THRESHOLD
            day_data["sector_limitup_cnt"] = day_data.loc[_lim_mask].groupby("industry")["code"].transform(
                "count").fillna(0)

        X = day_data[feature_cols].values
        preds = model.predict(X)
        day_data["pred"] = preds

        # 剔除T日涨停股（尾盘封单买不进，实盘无法成交）
        # 打板次日开盘接力(prev_limit_up_open_buy)不剔除涨停，反而只选涨停
        if "ret_1d_raw" in day_data.columns and not prev_limit_up_open_buy:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] < LIMIT_UP_THRESHOLD]
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 剔除{before - len(day_data)}只涨停股")
        if prev_limit_up_open_buy and "ret_1d_raw" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] >= LIMIT_UP_THRESHOLD]
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 非涨停过滤剔除{before - len(day_data)}只")

        # 只买前一天涨停的股票（打板次日接力）
        if only_prev_limit_up:
            before = len(day_data)
            day_data = day_data[day_data["_prev_limit_up"] == 1.0]
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 非前涨停过滤剔除{before - len(day_data)}只")

        # 方案一：剔除T日实体大阳线（(close-open)/open > 阈值），避免追高
        if filter_today_body_ret is not None:
            before = len(day_data)
            body_ret = (day_data["close"] - day_data["open"]) / day_data["open"]
            day_data = day_data[body_ret < filter_today_body_ret]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 实体大阳线过滤剔除{before - len(day_data)}只")

        # 方案二：剔除T日日涨幅过大（ret_1d_raw > 阈值），避免追高
        if filter_today_day_ret is not None and "ret_1d_raw" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] < filter_today_day_ret]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 日涨幅过滤剔除{before - len(day_data)}只")

        # 剔除T日暴跌股（ret_1d_raw < 下限阈值），避免接飞刀
        if filter_today_day_ret_min is not None and "ret_1d_raw" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] > filter_today_day_ret_min]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 日跌幅下限过滤剔除{before - len(day_data)}只")

        # 只买T日阴线收盘股（close < open）
        if filter_yin_line:
            before = len(day_data)
            day_data = day_data[day_data["close"] < day_data["open"]]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 阴线过滤剔除{before - len(day_data)}只")

        # 剔除流通市值过小的小票
        if filter_min_mv is not None and "_mv_yi" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["_mv_yi"] >= filter_min_mv]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 市值过滤剔除{before - len(day_data)}只")

        # 低振幅过滤：当日振幅z-score超过阈值则剔除（盈利股特征：振幅小）
        if filter_amplitude_max is not None and "amplitude_1d" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["amplitude_1d"] <= filter_amplitude_max]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 振幅过滤剔除{before - len(day_data)}只")

        # 缩量过滤：5日量比z-score超过阈值则剔除（盈利股特征：缩量）
        if filter_vol_ratio_max is not None and "vol_ratio_5" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["vol_ratio_5"] <= filter_vol_ratio_max]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 量比过滤剔除{before - len(day_data)}只")

        # 剔除T日上影线异常股（冲高回落，上方抛压重）
        if filter_upper_shadow is not None and "high_low_ratio" in day_data.columns:
            before = len(day_data)
            high = day_data["high_low_ratio"] * day_data["low"]
            body_top = day_data[["open", "close"]].max(axis=1)
            upper_shadow = (high - body_top) / day_data["close"]
            day_data = day_data[upper_shadow <= filter_upper_shadow]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 上影线过滤剔除{before - len(day_data)}只")

        if len(day_data) < top_n:
            continue

        # 二次排序：先选Top rerank_candidates 候选，再按规则重排
        if rerank_candidates is not None and rerank_method is not None:
            candidates = day_data.nlargest(min(rerank_candidates, len(day_data)), "pred")
            if rerank_method == "amplitude_asc" and "amplitude_1d" in candidates.columns:
                candidates = candidates.sort_values("amplitude_1d", ascending=True)
            elif rerank_method == "vol_ratio_asc" and "vol_ratio_5" in candidates.columns:
                candidates = candidates.sort_values("vol_ratio_5", ascending=True)
            elif rerank_method == "gap_desc" and "gap_ratio" in candidates.columns:
                candidates = candidates.sort_values("gap_ratio", ascending=False)
            elif rerank_method == "composite":
                # 手工composite：5基础因子 + K线形态(上影/光头) + 规模/量能/资金流3因子，rank累加排序
                # score 越小越靠前：小值偏好 score += w*rank，大值偏好 score -= w*rank
                # rerank_weights 默认 5 元素 = [振幅, 5日量比, 跳空, 5日位置, 3日动量]
                w = rerank_weights or [2, 2, 1, 1, 1]
                score = pd.Series(0.0, index=candidates.index)
                # --- 基础因子（当日日内形态，z-score 标准化值，rank 对标准化不敏感）---
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
                # --- K线形态因子（上影线短/光头形态；body/high20 默认关闭）---
                if (KLINE_SHADOW_W or KLINE_KFLAG_W or KLINE_BODY_W or KLINE_HIGH20_W) and {'open', 'high', 'low',
                                                                                            'close'}.issubset(
                        candidates.columns):
                    _eps = 1e-9
                    _rng = (candidates['high'] - candidates['low']).clip(lower=_eps)
                    _upper = (candidates['high'] - np.maximum(candidates['open'], candidates['close'])) / _rng
                    _body = (candidates['close'] - candidates['open']).abs() / _rng
                    _bull = ((candidates['close'] >= candidates['high'] - 0.01 * _rng) & (
                                candidates['close'] > candidates['open']))
                    _bear = ((candidates['close'] <= candidates['low'] + 0.01 * _rng) & (
                                candidates['close'] < candidates['open']))
                    if KLINE_SHADOW_W:
                        score += KLINE_SHADOW_W * _upper.rank(ascending=True)  # 上影线短→rank小→靠前
                    if KLINE_KFLAG_W:
                        _morph = _bull.astype(int) - _bear.astype(int)
                        score -= KLINE_KFLAG_W * _morph.rank(ascending=True)  # 光头阳(+1)→rank大→靠前；光头阴(-1)→靠后
                    if KLINE_BODY_W:
                        score -= KLINE_BODY_W * _body.rank(ascending=True)  # 实体大→rank大→靠前
                    if KLINE_HIGH20_W and 'ret_20d' in candidates.columns:
                        score += KLINE_HIGH20_W * candidates['ret_20d'].rank(ascending=True)  # 20日涨幅高→rank大→靠后（回避高位）
                # --- 规模/量能/资金流三因子（小值优先，2022起完整回测验证为正增益）---
                if RERANK_MV_W and "mv_log_yi" in candidates.columns:
                    score += RERANK_MV_W * candidates["mv_log_yi"].rank(ascending=True)  # 小市值→靠前
                if RERANK_VOL10_W and "vol_ratio_10" in candidates.columns:
                    score += RERANK_VOL10_W * candidates["vol_ratio_10"].rank(ascending=True)  # 10日缩量→靠前
                if RERANK_MF_W and "mf_intraday" in candidates.columns:
                    score += RERANK_MF_W * candidates["mf_intraday"].rank(ascending=True)  # 低位收盘/非放量→靠前
                # === 板块涨停集中度因子（实验开关，默认关闭；涨停集中板块强势，其所属可买个股高优先）===
                if SECTOR_LIMITUP_W > 0 and "sector_limitup_cnt" in candidates.columns:
                    score -= SECTOR_LIMITUP_W * candidates["sector_limitup_cnt"].rank(ascending=True)
                candidates = candidates.assign(_rerank_score=score).sort_values("_rerank_score")
            selected = candidates.head(top_n)
        else:
            selected = day_data.nlargest(top_n, "pred")
        selected_codes = selected["code"].tolist()
        selected_preds = selected["pred"].tolist()
        # 信号强度过滤：Top1预测收益率低于min_pred则空仓（弱信号胜率显著低，回测验证可提升夏普/降低回撤）
        if min_pred is not None and selected_preds:
            if selected_preds[0] < min_pred:
                skipped_empty += 1
                equity_points.append({"date": rebal_date, "equity": equity})
                continue
        # 分段动态开盘止盈：按当日Top1 pred选tp（弱信号小赚快跑、强信号给上涨空间）
        tp_eff = take_profit
        if dynamic_tp is not None and selected_preds:
            _tp_bnd, _tp_lo, _tp_hi = dynamic_tp
            tp_eff = _tp_lo if selected_preds[0] < _tp_bnd else _tp_hi
        if weights is not None:
            weights = list(weights)[:top_n] + [0.0] * max(0, top_n - len(weights))
        elif top_n == 3:
            weights = [1.0, 0.0, 0.0]
        elif top_n == 2:
            weights = [1.0, 0.0]
        else:
            weights = [1.0 / top_n] * top_n

        period_rets = []
        actual_hold_days = []

        for _, row in selected.iterrows():
            code = row["code"]
            if prev_limit_up_open_buy:
                buy_price = row.get("_open_d1", np.nan)  # T+1开盘价买入
                hold_days = 2
            else:
                buy_price = row["close"]  # T日尾盘买入价
                hold_days = 1
            if pd.isna(buy_price) or buy_price <= 0:
                continue

            is_stop_loss = False

            # 动态卖出：动态止盈+止损
            # 1. 开盘高开≥2% → 开盘价止盈
            # 2. 开盘低开≤-2% → 开盘价止损（开盘已跌超止损线，只能按实际开盘价成交）
            # 3. 盘中最低价跌破-2%止损线 → 按止损价成交（即使尾盘拉回止损线以上，
            #    实盘止损单在盘中触及-2%时已触发成交，回测必须按止损价计，不能按收盘价高估收益）
            # 4. 否则 → 收盘价卖出
            if prev_limit_up_open_buy:
                open_d1 = row.get("_open_d2", np.nan)
                close_d1 = row.get("_close_d2", np.nan)
                low_d1 = row.get("_low_d2", np.nan)
                high_d1_sell = row.get("_high_d2", np.nan)
            else:
                open_d1 = row["_open_d1"]
                close_d1 = row["_close_d1"]
                low_d1 = row.get("_low_d1", np.nan)
                high_d1_sell = row.get("_high_d1", np.nan)
            if pd.isna(close_d1):
                continue
            sell_price = None
            if not pd.isna(open_d1):
                open_gap = open_d1 / buy_price - 1
                if tp_eff and open_gap >= tp_eff:
                    # 高开达到当日动态止盈阈值，开盘价止盈卖出
                    sell_price = open_d1
                    is_stop_loss = False  # 止盈，非止损
                elif stop_loss and open_gap <= stop_loss:
                    # 开盘已跌超止损阈值，开盘价止损卖出（只能按实际开盘价成交）
                    sell_price = open_d1
                    is_stop_loss = True
            if sell_price is None and stop_loss and not pd.isna(low_d1):
                # 盘中最低价跌破止损线 → 按止损价成交（即使尾盘收回止损线以上）
                low_ret = low_d1 / buy_price - 1
                if low_ret <= stop_loss:
                    sell_price = buy_price * (1 + stop_loss)
                    is_stop_loss = True
            # 盘中最高价达到止盈线 → 按止盈价成交（同时触及止盈止损时已在上方按止损保守处理）
            if sell_price is None and intraday_take_profit is not None:
                if not pd.isna(high_d1_sell):
                    high_ret = high_d1_sell / buy_price - 1
                    if high_ret >= intraday_take_profit:
                        sell_price = buy_price * (1 + intraday_take_profit)
                        is_stop_loss = False
            if sell_price is None:
                # 兜底逻辑：
                # - low数据缺失(NaN)时回退旧逻辑：收盘价跌超止损线按止损价成交
                # - low有效且未破止损线：盘中最低价都没破，收盘价必高于止损线，按收盘价卖出
                if stop_loss and not pd.isna(close_d1):
                    cum_ret = close_d1 / buy_price - 1
                    if cum_ret <= stop_loss:
                        sell_price = buy_price * (1 + stop_loss)
                        is_stop_loss = True
                    else:
                        sell_price = close_d1
                else:
                    sell_price = close_d1

            if pd.isna(sell_price):
                continue
            stock_ret = sell_price / buy_price - 1
            period_rets.append(stock_ret)
            actual_hold_days.append(hold_days)

            # 计算卖出日期
            buy_idx = date_to_idx.get(rebal_date)
            sell_date = all_dates[buy_idx + hold_days] if buy_idx is not None and buy_idx + hold_days < len(
                all_dates) else None
            stock_trades.append({
                "buy_date": rebal_date,
                "sell_date": sell_date,
                "code": code,
                "buy_price": round(buy_price, 4),
                "sell_price": round(sell_price, 4),
                "hold_days": hold_days,
                "is_stop_loss": is_stop_loss,
                "return": round(stock_ret, 6),
                "return_pct": round(stock_ret * 100, 4),
            })

        if not period_rets:
            continue

        # 加权收益
        weighted_ret = sum(w * r for w, r in zip(weights[:len(period_rets)], period_rets))
        total_w = sum(weights[:len(period_rets)])
        weighted_ret = weighted_ret / total_w if total_w > 0 else 0
        weighted_ret -= transaction_cost  # 扣除佣金+印花税+滑点
        equity *= (1 + weighted_ret)

        avg_hold = np.mean(actual_hold_days) if actual_hold_days else 1
        trades.append({
            "date": rebal_date,
            "codes": ",".join(selected_codes),
            "preds": ",".join([f"{p:.6f}" for p in selected_preds]),
            "return": round(weighted_ret, 6),
            "equity": round(equity, 6),
            "avg_hold_days": round(avg_hold, 2),
        })
        equity_points.append({"date": rebal_date, "equity": equity})

    if skipped_empty > 0:
        logger.info(f"市场择时空仓次数: {skipped_empty} (大盘低于20日均线5%以上，极端熊市)")
    if skipped_filter > 0:
        logger.info(f"当天大阳线过滤累计剔除: {skipped_filter}只候选股")

    equity_df = pd.DataFrame(equity_points)
    if equity_df.empty:
        equity_df = pd.DataFrame([{"date": all_dates[0], "equity": 1.0}])
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    metrics = _calc_metrics(equity_df, trades)
    # 添加仓位比例信息
    metrics["weights"] = weights
    trades_df = pd.DataFrame(trades)
    stock_trades_df = pd.DataFrame(stock_trades)

    return {"metrics": metrics, "trades": trades_df, "stock_trades": stock_trades_df, "equity_curve": equity_df}


def _calc_metrics(equity_df: pd.DataFrame, trades: list) -> dict:
    """计算回测指标（1天持仓，每天调仓）"""
    horizon = 1  # 1天持仓
    if equity_df.empty or len(trades) == 0:
        return {"total_return": 0, "annual_return": 0, "sharpe": 0,
                "max_drawdown": 0, "win_rate": 0, "profit_loss_ratio": 0,
                "num_trades": 0}

    total_return = equity_df["equity"].iloc[-1] - 1

    # 年化收益
    days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days
    years = max(days / 365, 0.01)
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1

    # 夏普比率（基于日收益年化）
    equity_df["period_ret"] = equity_df["equity"].pct_change().fillna(0)
    period_rets = equity_df["period_ret"].values
    if len(period_rets) > 1 and np.std(period_rets) > 0:
        periods_per_year = TRADING_DAYS_PER_YEAR / horizon
        sharpe = np.mean(period_rets) / np.std(period_rets) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    # 最大回撤
    cummax = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - cummax) / cummax
    max_drawdown = float(drawdown.min())

    # 胜率 & 盈亏比
    rets = [t["return"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    win_rate = len(wins) / len(rets) if rets else 0
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = abs(float(np.mean(losses))) if losses else 1e-8
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 年度收益统计
    yearly = {}
    for t in trades:
        year = pd.Timestamp(t["date"]).year
        if year not in yearly:
            yearly[year] = []
        yearly[year].append(t["return"])
    yearly_returns = {}
    for year in sorted(yearly.keys()):
        year_rets = yearly[year]
        cum = 1.0
        for r in year_rets:
            cum *= (1 + r)
        yearly_returns[year] = {
            "return": round(cum - 1, 4),
            "trades": len(year_rets),
            "win_rate": round(sum(1 for r in year_rets if r > 0) / len(year_rets), 4) if year_rets else 0,
            "avg_return": round(float(np.mean(year_rets)), 6) if year_rets else 0,
        }

    # 月度收益统计
    monthly = {}
    for t in trades:
        ts = pd.Timestamp(t["date"])
        month_key = f"{ts.year}-{ts.month:02d}"
        if month_key not in monthly:
            monthly[month_key] = []
        monthly[month_key].append(t["return"])
    # 补全回测区间内所有月份（含 0 笔交易的空仓月），避免空仓月被漏报
    monthly_returns = {}
    if "date" in equity_df.columns and len(equity_df) > 0:
        m_start = pd.Timestamp(equity_df["date"].min()).to_period("M")
        m_end = pd.Timestamp(equity_df["date"].max()).to_period("M")
        all_months = pd.period_range(m_start, m_end, freq="M")
    else:
        all_months = [pd.Period(k) for k in sorted(monthly.keys())]
    for p in all_months:
        month_key = f"{p.year}-{p.month:02d}"
        month_rets = monthly.get(month_key, [])
        cum = 1.0
        for r in month_rets:
            cum *= (1 + r)
        monthly_returns[month_key] = {
            "return": round(cum - 1, 4),
            "trades": len(month_rets),
            "win_rate": round(sum(1 for r in month_rets if r > 0) / len(month_rets), 4) if month_rets else 0,
            "avg_return": round(float(np.mean(month_rets)), 6) if month_rets else 0,
        }

    return {
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "profit_loss_ratio": round(profit_loss_ratio, 4),
        "avg_return_per_trade": round(float(np.mean(rets)) if rets else 0, 6),
        "num_trades": len(trades),
        "yearly_returns": yearly_returns,
        "monthly_returns": monthly_returns,
    }


def _print_metrics(mode: str, m: dict, take_profit: float = 0.02, stop_loss: float = -0.02):
    mode_desc = "T日尾盘买/T+1动态卖"
    tp_str = f"止盈{take_profit * 100:.1f}%" if take_profit else "无止盈"
    sl_str = f"止损{stop_loss * 100:.1f}%" if stop_loss else "不止损"
    stop_desc = f"动态{tp_str}+{sl_str}"
    weights = m.get("weights", [1.0, 0.0, 0.0])
    weight_str = ":".join([f"{w * 100:.0f}" for w in weights])
    logger.info(
        f"  1天持仓-{mode_desc} 回测结果 (Top3仓位{weight_str}, {stop_desc}, 已扣交易成本{DEFAULT_TRANSACTION_COST * 100:.2f}%/次, 剔除涨停股):")
    logger.info(f"    累计收益率: {m['total_return'] * 100:.2f}%")
    logger.info(f"    年化收益率: {m['annual_return'] * 100:.2f}%")
    logger.info(f"    夏普比率:   {m['sharpe']:.4f}")
    logger.info(f"    最大回撤:   {m['max_drawdown'] * 100:.2f}%")
    logger.info(f"    胜率:       {m['win_rate'] * 100:.2f}%")
    logger.info(f"    盈亏比:     {m['profit_loss_ratio']:.4f}")
    logger.info(f"    单次平均收益: {m.get('avg_return_per_trade', 0) * 100:.4f}%")
    logger.info(f"    交易次数:   {m['num_trades']}")
    # 年度收益
    yearly = m.get("yearly_returns", {})
    if yearly:
        logger.info(f"    --- 年度收益 ---")
        for year, ym in sorted(yearly.items()):
            logger.info(
                f"    {year}年: 收益={ym['return'] * 100:+.2f}%, 交易{ym['trades']}次, 胜率={ym['win_rate'] * 100:.1f}%, 单次均收={ym['avg_return'] * 100:.4f}%")

    # 最新年度月度收益（含空仓月，0 笔交易也打印，避免缺月）
    monthly = m.get("monthly_returns", {})
    latest_year = max((k.split("-")[0] for k in monthly.keys()), default=str(pd.Timestamp.now().year))
    monthly_latest = {k: v for k, v in monthly.items() if k.startswith(latest_year)}
    if monthly_latest:
        logger.info(f"    --- {latest_year}年月度收益（空仓月也列出）---")
        for month_key, mm in sorted(monthly_latest.items()):
            if mm["trades"] == 0:
                logger.info(f"    {month_key}: 空仓（0笔交易，大盘择时或二次排序Top1 pred<min_pred，收益0%）")
            else:
                logger.info(
                    f"    {month_key}: 收益={mm['return'] * 100:+.2f}%, 交易{mm['trades']}次, 胜率={mm['win_rate'] * 100:.1f}%, 单次均收={mm['avg_return'] * 100:.4f}%")


def save_backtest_results(results: dict, tag: str = "latest"):
    """保存回测结果到文件"""
    out_dir = RESULT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    yearly_all = []
    monthly_all = []
    for h, res in results.items():
        m = res["metrics"]
        summary.append({"sell_mode": h, **{k: v for k, v in m.items() if k != "yearly_returns"}})
        res["trades"].to_csv(out_dir / f"trades_{h}.csv", index=False, encoding="utf-8-sig")
        if "stock_trades" in res and not res["stock_trades"].empty:
            res["stock_trades"].to_csv(out_dir / f"stock_trades_{h}.csv", index=False, encoding="utf-8-sig")
        res["equity_curve"].to_csv(out_dir / f"equity_{h}.csv", index=False, encoding="utf-8-sig")
        # 年度收益
        for year, ym in m.get("yearly_returns", {}).items():
            yearly_all.append({"sell_mode": h, "year": year, **ym})
        # 月度收益
        for month_key, mm in m.get("monthly_returns", {}).items():
            monthly_all.append({"sell_mode": h, "month": month_key, **mm})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    if yearly_all:
        pd.DataFrame(yearly_all).to_csv(out_dir / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    if monthly_all:
        pd.DataFrame(monthly_all).to_csv(out_dir / "monthly_returns.csv", index=False, encoding="utf-8-sig")
    logger.info(f"回测结果已保存到 {out_dir}")
    return summary_df
