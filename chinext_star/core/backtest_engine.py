"""
回测引擎（1天持仓策略，动态卖出）：
- 1d_close: T日模型预测→粗筛 Top8 候选→池内剔除极端放量票→composite 二次排序→取 Top3（实际 Top1 全仓 [1,0,0]），T日尾盘买入
- 动态卖出：T+1 开盘高开达分段阈值则开盘止盈（pred<1%→1.2%、pred≥1%→2.5%）；未触发则进入移动止盈，
  盘中冲高≥0.75% 激活、从当日高点回撤 0.25% 卖出；未激活或收盘在回撤线上方则收盘卖出；不止损
  （回测验证止损会切掉强势股的次日修复；固定盘中止盈 7% 仅作移动止盈关闭时的兼容兜底）
- 每交易日调仓，扣交易成本 0.35%/次；剔除 T 日涨停（尾盘封单买不进）与 T 日涨幅超出 [-6%, 12%] 区间的追高/暴跌票
- 弱信号空仓：二次排序后 Top1 的 pred < min_pred(1.0%) 当日空仓（提升夏普、降低回撤）
- 大盘择时：沪深300当日收阴则空仓；极端熊市择时由 market_timing_empty 控制，生产口径关闭（-1.0）
- composite 二次排序：5 基础因子（权重 [4,2,1,0,1]=振幅/5日量比/跳空/5日位置/3日动量，5日位置项权重0）
  + K线形态（上影线/光头阳）+ 3 因子（流通市值对数/10日量比/日内资金流，资金流权重0.5），rank 累加，score 越小越靠前
- 输出累计收益、年化、夏普、最大回撤、胜率、年度/月度收益与净值曲线
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import board_profile
from .utils import RESULT_DIR, get_logger

# === composite 二次排序 K 线形态因子（2022-01 起样本外回测验证为正增益）===
KLINE_SHADOW_W = 1.0  # 上影线短加分权重（短上影=无冲高回落）
KLINE_KFLAG_W = 0.5  # 光头阳加分/光头阴减分权重（半权重）

# === composite 二次排序规模/量能/资金流因子（小值优先，样本外回测验证为正增益）===
RERANK_MV_W = 1.0  # 流通市值对数 mv_log_yi（小市值次日溢价显著）
RERANK_VOL10_W = 1.0  # 10日量比 vol_ratio_10（缩量企稳，比5日量比更稳）
RERANK_MF_W = 0.5  # 日内资金流 mf_intraday（半权重，规避高位放量收盘）

logger = get_logger("backtest")

TRADING_DAYS_PER_YEAR = 252
# A股单次调仓交易成本：佣金万2.5(买卖)+印花税千1(卖出)+滑点千1(买卖) ≈ 0.35%
DEFAULT_TRANSACTION_COST = 0.0035
# 涨停阈值由 board_profile 按个股 code+date 逐行计算（主板±10%、双创±20%，创业板按注册制日期分界）


def run_backtest(factor_data: pd.DataFrame, models: dict,
                 top_n: int = 3, horizons: list = None,
                 start_date: str = None, end_date: str = None,
                 transaction_cost: float = DEFAULT_TRANSACTION_COST,
                 take_profit: float = 0.020,
                 stop_loss: float = None,
                 filter_today_day_ret: float = None,
                 filter_today_day_ret_min: float = None,
                 intraday_take_profit: float = None,
                 weights: list = None,
                 skip_if_index_down: bool = False,
                 rerank_candidates: int = None,
                 rerank_method: str = None,
                 rerank_weights: list = None,
                 min_pred: float = None,
                 dynamic_tp: tuple = None,
                 candidate_factors: list = None,
                 market_timing_empty: float = -1.0,
                 trailing_start: float = None,
                 trailing_dd: float = None) -> dict:
    """运行动态卖出回测（1d_close）
    take_profit: 开盘止盈基准阈值，被 dynamic_tp 分段动态覆盖；None=无止盈
    stop_loss: 止损阈值，None=不止损（生产口径）
    filter_today_day_ret: T日日涨幅(ret_1d_raw)超过此阈值则剔除（追高），None=不过滤
    filter_today_day_ret_min: T日日涨幅低于此阈值则剔除（暴跌股），None=不过滤
    intraday_take_profit: 盘中最高价达到此阈值则止盈卖出，None=不启用
    weights: TopN 仓位权重列表，None=默认 Top3 全仓 Top1 [1,0,0]
    skip_if_index_down: 大盘（沪深300）当天收阴线则空仓不买
    rerank_candidates: 先从当日预测全集选 TopN 候选再二次排序；None=不启用
    rerank_method: 二次排序方式，生产固定 composite
    rerank_weights: composite 基础 5 因子权重 [振幅,5日量比,跳空,5日位置,3日动量]，None=[4,2,1,0,1]
    min_pred: 二次排序后 Top1 pred 低于此阈值当日空仓，None=不启用
    dynamic_tp: 分段动态开盘止盈 tuple=(pred 分界, 弱信号 tp, 强信号 tp)，None=用 take_profit
    candidate_factors: 候选池内横截面分位过滤 list of (col, direction, q)，None=不过滤
    market_timing_empty: 极端熊市择时阈值（沪深300 趋势低于该值空仓），-1.0=关闭
    trailing_start/trailing_dd: 移动止盈参数（默认 None=关闭，退回固定盘中止盈 intraday_take_profit）；
        非 None 时替代固定盘中止盈：开盘动态止盈优先，盘中冲高达 trailing_start 激活移动止盈，
        从高点回撤 trailing_dd 卖出；未激活或收盘仍在回撤线上方则收盘卖出。B16 生产由 backtest.py/train.py 传入
    """
    if horizons is None:
        horizons = ["1d_close"]

    factor_data = factor_data.copy()
    factor_data["date"] = pd.to_datetime(factor_data["date"])

    if start_date:
        factor_data = factor_data[factor_data["date"] >= pd.Timestamp(start_date)]
    if end_date:
        factor_data = factor_data[factor_data["date"] <= pd.Timestamp(end_date)]

    results = {}
    for h in horizons:
        logger.info(f"{'=' * 60}")
        tp_str = f"，止盈{take_profit * 100:.1f}%" if take_profit else ""
        sl_str = "，不止损" if stop_loss is None else f"，止损{stop_loss * 100:.1f}%"
        filt_str = ""
        if filter_today_day_ret is not None:
            filt_str += f"，T日日涨幅>{filter_today_day_ret * 100:.1f}%剔除"
        if filter_today_day_ret_min is not None:
            filt_str += f"，T日日涨幅<{filter_today_day_ret_min * 100:.1f}%剔除"
        if intraday_take_profit is not None:
            filt_str += f"，盘中止盈{intraday_take_profit * 100:.1f}%"
        if skip_if_index_down:
            filt_str += "，大盘阴线空仓"
        if rerank_candidates is not None:
            filt_str += f"，二次排序Top{rerank_candidates}/{rerank_method}"
        logger.info(f"开始回测: 1天持仓-动态卖出 ({h}){tp_str}{sl_str}, Top{top_n}{filt_str}")
        res = _backtest_single_mode(factor_data, models[h], h, top_n, transaction_cost,
                                    take_profit, stop_loss,
                                    filter_today_day_ret,
                                    filter_today_day_ret_min,
                                    intraday_take_profit,
                                    weights,
                                    skip_if_index_down,
                                    rerank_candidates,
                                    rerank_method,
                                    rerank_weights,
                                    min_pred,
                                    dynamic_tp,
                                    candidate_factors=candidate_factors,
                                    market_timing_empty=market_timing_empty,
                                    trailing_start=trailing_start,
                                    trailing_dd=trailing_dd)
        results[h] = res
        _print_metrics(h, res["metrics"], take_profit, stop_loss)

    return results


def _backtest_single_mode(factor_data: pd.DataFrame, model, mode: str, top_n: int,
                          transaction_cost: float = DEFAULT_TRANSACTION_COST,
                          take_profit: float = 0.020,
                          stop_loss: float = None,
                          filter_today_day_ret: float = None,
                          filter_today_day_ret_min: float = None,
                          intraday_take_profit: float = None,
                          weights: list = None,
                          skip_if_index_down: bool = False,
                          rerank_candidates: int = None,
                          rerank_method: str = None,
                          rerank_weights: list = None,
                          min_pred: float = None,
                          dynamic_tp: tuple = None,
                          candidate_factors: list = None,
                          market_timing_empty: float = -1.0,
                          trailing_start: float = None,
                          trailing_dd: float = None) -> dict:
    """单模式回测（1d_close 动态止盈，不止损）。参数口径见 run_backtest docstring。"""
    feature_cols = [c for c in (model.feature_names or []) if c is not None and c in factor_data.columns]
    if not feature_cols:
        raise ValueError("模型特征列与数据不匹配")

    # 二次排序：composite 手工规则（候选池内因子过滤后按 rank 累加打分，score 越小越靠前）

    all_dates = sorted(factor_data["date"].unique())
    # 1天持仓，每个交易日调仓
    rebalance_dates = all_dates
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # 预计算 T+1 开盘价、收盘价、最低价、最高价（T日尾盘买入）
    logger.info("预计算T+1价格矩阵（T日尾盘买入）...")
    factor_data = factor_data.sort_values(["code", "date"]).reset_index(drop=True)
    # 板块自适应：逐行计算个股当日涨停阈值 _limit_thr（主板10%/双创20%、创业板按注册制日期分界），供涨停剔除使用
    factor_data = board_profile.add_limit_threshold(factor_data)
    g = factor_data.groupby("code", sort=False)
    factor_data["_open_d1"] = g["open"].shift(-1)
    factor_data["_close_d1"] = g["close"].shift(-1)
    factor_data["_low_d1"] = g["low"].shift(-1)
    factor_data["_high_d1"] = g["high"].shift(-1)

    # 极端熊市择时：沪深300趋势低于 market_timing_empty 空仓；-1.0 表示关闭
    MARKET_TIMING_EMPTY = market_timing_empty
    has_market_feature = "hs300_trend" in factor_data.columns

    equity = 1.0
    trades = []
    stock_trades = []
    equity_points = []
    empty_days = []  # 每个空仓日 {"date", "reason"}，保证每个交易日都有“开仓/空仓”交代
    skipped_empty = 0
    skipped_filter = 0

    def _mark_empty(dt, reason):
        """记录一个空仓日：净值不变化（追加equity点），并登记空仓原因。"""
        nonlocal skipped_empty
        skipped_empty += 1
        empty_days.append({"date": dt, "reason": reason})
        equity_points.append({"date": dt, "equity": equity})

    pbar_desc = "回测 动态卖出"

    for rebal_date in tqdm(rebalance_dates, desc=pbar_desc, ncols=100):
        day_data = factor_data[factor_data["date"] == rebal_date].copy()
        if len(day_data) < top_n:
            _mark_empty(rebal_date, "当日有效样本不足（可买股票少于TopN）")
            continue

        # 极端熊市择时：沪深300 趋势低于阈值空仓（MARKET_TIMING_EMPTY=-1.0 关闭）
        if has_market_feature and MARKET_TIMING_EMPTY > -0.9:
            market_trend = day_data["hs300_trend"].iloc[0]
            if market_trend < MARKET_TIMING_EMPTY:
                _mark_empty(rebal_date, f"大盘择时（沪深300低于20日均线{abs(MARKET_TIMING_EMPTY) * 100:.0f}%以上）")
                continue

        # 大盘当天收阴线则空仓（沪深300当日收益<0）
        if skip_if_index_down and "hs300_ret_1d" in day_data.columns:
            idx_ret = day_data["hs300_ret_1d"].iloc[0]
            if idx_ret < 0:
                _mark_empty(rebal_date, "大盘当日收阴（沪深300当日收益<0）")
                continue

        X = day_data[feature_cols].values
        preds = model.predict(X)
        day_data["pred"] = preds

        # 剔除 T 日涨停股（尾盘封单买不进，实盘无法成交）
        if "ret_1d_raw" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] < day_data["_limit_thr"]]
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 剔除{before - len(day_data)}只涨停股")

        # 剔除 T 日日涨幅过大（追高）与跌幅过大（接飞刀）
        if filter_today_day_ret is not None and "ret_1d_raw" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] < filter_today_day_ret]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 日涨幅过滤剔除{before - len(day_data)}只")

        if filter_today_day_ret_min is not None and "ret_1d_raw" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] > filter_today_day_ret_min]
            skipped_filter += before - len(day_data)
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 日跌幅下限过滤剔除{before - len(day_data)}只")

        if len(day_data) < top_n:
            _mark_empty(rebal_date, "涨停/涨跌幅过滤后可买标的不足TopN")
            continue

        # 二次排序：先选Top rerank_candidates 候选，再按规则重排
        if rerank_candidates is not None and rerank_method is not None:
            candidates = day_data.nlargest(min(rerank_candidates, len(day_data)), "pred")
            # 候选池内因子过滤（与盈亏组分析口径一致：只在 pred TopK 候选内按当日横截面 rank 剔除，
            # 不作用于当日全集，避免误删模型高 pred 候选；dir='hi' 剔除分位过高、'lo' 剔除过低）。
            # 若过滤后候选为空则该因子跳过（保证至少可买）。
            if candidate_factors and len(candidates) > 0:
                for _col, _dir, _q in candidate_factors:
                    if _col not in candidates.columns:
                        continue
                    _v = pd.to_numeric(candidates[_col], errors="coerce")
                    _pct = _v.rank(pct=True, method="average")
                    if _dir == "hi":
                        _keep = _pct <= _q
                    elif _dir == "lo":
                        _keep = _pct >= _q
                    else:
                        continue
                    _kept = candidates[_keep.fillna(True)]
                    if len(_kept) > 0:
                        candidates = _kept
            # composite：5 基础因子 + K线形态(上影/光头) + 规模/量能/资金流 3 因子，rank 累加排序
            # score 越小越靠前：小值偏好 score += w*rank，大值偏好 score -= w*rank
            w = rerank_weights or [4, 2, 1, 0, 1]
            score = pd.Series(0.0, index=candidates.index)
            # --- 基础因子（当日日内形态，rank 对标准化不敏感）---
            if "amplitude_1d" in candidates.columns:
                score += w[0] * candidates["amplitude_1d"].rank(ascending=True)  # 振幅小→靠前
            if "vol_ratio_5" in candidates.columns:
                score += w[1] * candidates["vol_ratio_5"].rank(ascending=True)  # 缩量→靠前
            if "gap_ratio" in candidates.columns:
                score -= w[2] * candidates["gap_ratio"].rank(ascending=True)  # 高开→靠前
            if "close_position_5d" in candidates.columns and w[3] > 0:
                score -= w[3] * candidates["close_position_5d"].rank(ascending=True)  # 5日高位→靠前（权重0时关闭）
            if "ret_3d" in candidates.columns:
                score -= w[4] * candidates["ret_3d"].rank(ascending=True)  # 3日动量强→靠前
            # --- K线形态因子（上影线短 + 光头阳/阴形态）---
            if (KLINE_SHADOW_W or KLINE_KFLAG_W) and {'open', 'high', 'low',
                                                      'close'}.issubset(candidates.columns):
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
                    score -= KLINE_KFLAG_W * _morph.rank(ascending=True)  # 光头阳(+1)→rank大→靠前；光头阴(-1)→靠后
            # --- 规模/量能/资金流三因子（小值优先；流通市值/10日量比权重1，资金流半权重0.5）---
            if RERANK_MV_W and "mv_log_yi" in candidates.columns:
                score += RERANK_MV_W * candidates["mv_log_yi"].rank(ascending=True)  # 小市值→靠前
            if RERANK_VOL10_W and "vol_ratio_10" in candidates.columns:
                score += RERANK_VOL10_W * candidates["vol_ratio_10"].rank(ascending=True)  # 10日缩量→靠前
            if RERANK_MF_W and "mf_intraday" in candidates.columns:
                score += RERANK_MF_W * candidates["mf_intraday"].rank(ascending=True)  # 低位收盘/非放量→靠前
            candidates = candidates.assign(_rerank_score=score).sort_values("_rerank_score")
            selected = candidates.head(top_n)
        else:
            selected = day_data.nlargest(top_n, "pred")

        selected_codes = selected["code"].tolist()
        selected_preds = selected["pred"].tolist()
        # 信号强度过滤：Top1预测收益率低于min_pred则空仓（弱信号胜率显著低，回测验证可提升夏普/降低回撤）
        if min_pred is not None and selected_preds:
            if selected_preds[0] < min_pred:
                _mark_empty(rebal_date,
                            f"弱信号空仓（二次排序后Top1 pred={selected_preds[0] * 100:.2f}%低于阈值{min_pred * 100:.2f}%）")
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
            buy_price = row["close"]  # T日尾盘买入价，持仓 1 天
            hold_days = 1
            if pd.isna(buy_price) or buy_price <= 0:
                continue

            is_stop_loss = False

            # T+1 动态卖出：
            # 1. 开盘高开 ≥ 动态止盈阈值 → 开盘价止盈
            # 2. stop_loss 非 None 时按止损线处理（生产口径 stop_loss=None，不止损，分支为防御保留）
            # 3. 盘中移动止盈：冲高达 trailing_start 激活，回撤 trailing_dd 成交；移动止盈关闭时退回固定盘中止盈线
            # 4. 均未触及 → 收盘价卖出
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
            # 盘中卖出：trailing_start/trailing_dd 非 None（B16 生产口径）走移动止盈；
            # 两者均 None 时退回固定盘中止盈线 intraday_take_profit（互斥短路，不重复执行）
            if sell_price is None and trailing_start is not None and trailing_dd is not None:
                if not pd.isna(high_d1_sell):
                    high_ret = high_d1_sell / buy_price - 1
                    if high_ret >= trailing_start:
                        trail_ret = high_ret - trailing_dd
                        # 收盘仍在回撤线上方 → 落兜底按收盘卖；否则按回撤线成交
                        if pd.isna(close_d1) or close_d1 / buy_price - 1 < trail_ret:
                            sell_price = buy_price * (1 + trail_ret)
                            is_stop_loss = False
            elif sell_price is None and intraday_take_profit is not None:
                # 盘中最高价达到止盈线 → 按止盈价成交（同时触及止盈止损时已在上方按止损保守处理）
                if not pd.isna(high_d1_sell):
                    high_ret = high_d1_sell / buy_price - 1
                    if high_ret >= intraday_take_profit:
                        sell_price = buy_price * (1 + intraday_take_profit)
                        is_stop_loss = False
            if sell_price is None:
                # 兜底：stop_loss 非 None 时收盘价跌破止损线按止损价成交；否则按收盘价卖出
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
            _mark_empty(rebal_date, "入选股无有效T+1成交价格（停牌/数据缺失）")
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

    total_days = len(rebalance_dates)
    open_days = len(trades)
    logger.info("=" * 60)
    logger.info(f"交易日合计: {total_days} 天 = 开仓 {open_days} 天 + 空仓 {len(empty_days)} 天")
    if open_days + len(empty_days) != total_days:
        logger.warning(
            f"开仓+空仓({open_days + len(empty_days)})与交易日总数({total_days})不一致，存在未归类交易日，请检查")

    # 空仓日按原因归组统计：INFO 仅输出分组计数；逐条日期+原因走 DEBUG，明细另存 empty_days csv
    if empty_days:
        from collections import defaultdict
        groups = defaultdict(list)
        for ed in empty_days:
            groups[ed["reason"].split("（")[0]].append(ed)
        logger.info(f"--- 空仓日按原因分组（共 {len(empty_days)} 天）---")
        for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            logger.info(f"[{key}] 共 {len(items)} 天")
            for ed in items:
                logger.debug(f"    空仓 {pd.Timestamp(ed['date']):%Y-%m-%d}  {ed['reason']}")
    if skipped_filter > 0:
        logger.info(f"涨幅/暴跌过滤累计剔除: {skipped_filter}只候选股")

    equity_df = pd.DataFrame(equity_points)
    if equity_df.empty:
        equity_df = pd.DataFrame([{"date": all_dates[0], "equity": 1.0}])
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    metrics = _calc_metrics(equity_df, trades, rebalance_dates, empty_days)
    # 添加仓位比例信息
    metrics["weights"] = weights
    trades_df = pd.DataFrame(trades)
    stock_trades_df = pd.DataFrame(stock_trades)
    empty_days_df = pd.DataFrame(empty_days)

    return {"metrics": metrics, "trades": trades_df, "stock_trades": stock_trades_df,
            "empty_days": empty_days_df, "equity_curve": equity_df}


def _calc_metrics(equity_df: pd.DataFrame, trades: list, trading_dates=None, empty_days=None) -> dict:
    """计算回测指标（1天持仓，每天调仓）。
    trading_dates: 回测区间内全部交易日（用于补全0交易的空仓月/空仓年份）；
    empty_days: 空仓日记录 [{"date","reason"}]，用于统计开仓/空仓天数。"""
    horizon = 1  # 1天持仓
    if equity_df.empty:
        return {"total_return": 0, "annual_return": 0, "sharpe": 0,
                "max_drawdown": 0, "win_rate": 0, "profit_loss_ratio": 0,
                "num_trades": 0, "yearly_returns": {}, "monthly_returns": {}}

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

    # 交易日历 / 开仓日 / 空仓日集合（全部归一化到日期）
    open_dates = [pd.Timestamp(t["date"]).normalize() for t in trades]
    empty_dates = [pd.Timestamp(e["date"]).normalize() for e in (empty_days or [])]
    if trading_dates is not None:
        cal_dates = pd.to_datetime(pd.Series(trading_dates)).dt.normalize().tolist()
    else:
        cal_dates = pd.to_datetime(equity_df["date"]).dt.normalize().tolist()
    open_set, empty_set = set(open_dates), set(empty_dates)

    # 年度收益统计（按交易日历覆盖每一年，即使某年0交易也列出；含开仓/空仓天数）
    yearly_rets = {}
    for t in trades:
        yearly_rets.setdefault(pd.Timestamp(t["date"]).year, []).append(t["return"])
    yearly_cal, yearly_open, yearly_empty = {}, {}, {}
    for d in cal_dates:
        yearly_cal[d.year] = yearly_cal.get(d.year, 0) + 1
    for d in open_set:
        yearly_open[d.year] = yearly_open.get(d.year, 0) + 1
    for d in empty_set:
        yearly_empty[d.year] = yearly_empty.get(d.year, 0) + 1
    yearly_returns = {}
    for year in sorted(yearly_cal.keys()):
        year_rets = yearly_rets.get(year, [])
        cum = 1.0
        for r in year_rets:
            cum *= (1 + r)
        yearly_returns[year] = {
            "return": round(cum - 1, 4),
            "trades": len(year_rets),
            "open_days": yearly_open.get(year, 0),
            "empty_days": yearly_empty.get(year, 0),
            "total_days": yearly_cal.get(year, 0),
            "win_rate": round(sum(1 for r in year_rets if r > 0) / len(year_rets), 4) if year_rets else 0,
            "avg_return": round(float(np.mean(year_rets)), 6) if year_rets else 0,
        }

    # 月度收益统计（按交易日历补全回测区间内每个月份，含0交易空仓月；含开仓/空仓天数）
    monthly_rets = {}
    for t in trades:
        ts = pd.Timestamp(t["date"])
        monthly_rets.setdefault(f"{ts.year}-{ts.month:02d}", []).append(t["return"])
    monthly_cal, monthly_open, monthly_empty = {}, {}, {}
    for d in cal_dates:
        mk = f"{d.year}-{d.month:02d}"
        monthly_cal[mk] = monthly_cal.get(mk, 0) + 1
    for d in open_set:
        mk = f"{d.year}-{d.month:02d}"
        monthly_open[mk] = monthly_open.get(mk, 0) + 1
    for d in empty_set:
        mk = f"{d.year}-{d.month:02d}"
        monthly_empty[mk] = monthly_empty.get(mk, 0) + 1
    monthly_returns = {}
    if cal_dates:
        all_months = pd.period_range(min(cal_dates).to_period("M"), max(cal_dates).to_period("M"), freq="M")
    else:
        all_months = [pd.Period(k) for k in sorted(monthly_rets.keys())]
    for p in all_months:
        month_key = f"{p.year}-{p.month:02d}"
        month_rets = monthly_rets.get(month_key, [])
        cum = 1.0
        for r in month_rets:
            cum *= (1 + r)
        monthly_returns[month_key] = {
            "return": round(cum - 1, 4),
            "trades": len(month_rets),
            "open_days": monthly_open.get(month_key, 0),
            "empty_days": monthly_empty.get(month_key, 0),
            "total_days": monthly_cal.get(month_key, 0),
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


def _print_metrics(mode: str, m: dict, take_profit: float = 0.020, stop_loss: float = None):
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
    # 年度收益（含开仓/空仓天数）
    yearly = m.get("yearly_returns", {})
    if yearly:
        logger.info("    --- 年度收益（开仓/空仓天数）---")
        for year, ym in sorted(yearly.items()):
            logger.info(
                f"    {year}年: 收益={ym['return'] * 100:+.2f}%, "
                f"开仓{ym.get('open_days', ym['trades'])}天/空仓{ym.get('empty_days', 0)}天(共{ym.get('total_days', ym['trades'])}交易日), "
                f"交易{ym['trades']}次, 胜率={ym['win_rate'] * 100:.1f}%, 单次均收={ym['avg_return'] * 100:.4f}%")

    # 最新年度月度收益（每个月都打印，含 0 交易的纯空仓月及开仓/空仓天数）
    monthly = m.get("monthly_returns", {})
    latest_year = max((k.split("-")[0] for k in monthly.keys()), default=str(pd.Timestamp.now().year))
    monthly_latest = {k: v for k, v in monthly.items() if k.startswith(latest_year)}
    if monthly_latest:
        logger.info(f"    --- {latest_year}年月度收益（空仓月也列出，开仓/空仓天数）---")
        for month_key, mm in sorted(monthly_latest.items()):
            if mm["trades"] == 0:
                logger.info(
                    f"    {month_key}: 空仓月（开仓0天/空仓{mm.get('empty_days', 0)}天，共{mm.get('total_days', 0)}交易日，收益0%；逐日原因见 empty_days 结果 csv）")
            else:
                logger.info(
                    f"    {month_key}: 收益={mm['return'] * 100:+.2f}%, "
                    f"开仓{mm.get('open_days', mm['trades'])}天/空仓{mm.get('empty_days', 0)}天, "
                    f"交易{mm['trades']}次, 胜率={mm['win_rate'] * 100:.1f}%, 单次均收={mm['avg_return'] * 100:.4f}%")


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
        if "empty_days" in res and not res["empty_days"].empty:
            res["empty_days"].to_csv(out_dir / f"empty_days_{h}.csv", index=False, encoding="utf-8-sig")
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
