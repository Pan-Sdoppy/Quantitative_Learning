"""
回测引擎（1天持仓策略，动态卖出）：
- 1d_close: T日预测选股，T日尾盘买入，动态卖出——高开≥2%开盘止盈，低开≤-2%开盘止损，否则尾盘收盘价卖出（跌超2%按-2%止损）
- 每个交易日调仓，置信度加权Top3（Top1 50%, Top2 30%, Top3 20%）
- 扣除交易成本（0.35%/次），剔除T日涨停股（尾盘封单买不进）
- 1d_close模式：动态止盈2%+动态止损-2%（经参数优化最优阈值）
- 市场择时：沪深300低于20日均线5%以上空仓（极端熊市保护）
- 当天大阳线过滤（可选）：T日实体涨幅(close-open)/open或日涨幅(ret_1d_raw)超阈值则剔除，避免追高
- 计算累计收益、年化、夏普、最大回撤、胜率、年度收益
- 输出回测报告与净值曲线
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import RESULT_DIR, get_logger

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
                 weights: list = None) -> dict:
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
        logger.info(f"开始回测: 1天持仓-动态卖出 ({h}){tp_str}{sl_str}, Top{top_n}{filt_str}")
        res = _backtest_single_mode(factor_data, models[h], h, top_n, transaction_cost,
                                    take_profit, stop_loss,
                                    filter_today_body_ret, filter_today_day_ret,
                                    filter_today_day_ret_min,
                                    filter_yin_line, filter_upper_shadow,
                                    intraday_take_profit,
                                    weights)
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
                          weights: list = None) -> dict:
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
    """
    feature_cols = [c for c in (model.feature_names or []) if c is not None and c in factor_data.columns]
    if not feature_cols:
        raise ValueError("模型特征列与数据不匹配")

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

        X = day_data[feature_cols].values
        preds = model.predict(X)
        day_data["pred"] = preds

        # 剔除T日涨停股（尾盘封单买不进，实盘无法成交）
        if "ret_1d_raw" in day_data.columns:
            before = len(day_data)
            day_data = day_data[day_data["ret_1d_raw"] < LIMIT_UP_THRESHOLD]
            if len(day_data) < before:
                logger.debug(f"  {rebal_date:%Y-%m-%d}: 剔除{before - len(day_data)}只涨停股")

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

        # 选TopN，按置信度加权（Top1 60%, Top2 40%, Top3 0%，只买Top1+Top2）
        selected = day_data.nlargest(top_n, "pred")
        selected_codes = selected["code"].tolist()
        selected_preds = selected["pred"].tolist()
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
            buy_price = row["close"]  # T日尾盘买入价
            if pd.isna(buy_price) or buy_price <= 0:
                continue

            hold_days = 1
            is_stop_loss = False

            # 动态卖出：动态止盈+止损
            # 1. 开盘高开≥2% → 开盘价止盈
            # 2. 开盘低开≤-2% → 开盘价止损（开盘已跌超止损线，只能按实际开盘价成交）
            # 3. 盘中最低价跌破-2%止损线 → 按止损价成交（即使尾盘拉回止损线以上，
            #    实盘止损单在盘中触及-2%时已触发成交，回测必须按止损价计，不能按收盘价高估收益）
            # 4. 否则 → 收盘价卖出
            open_d1 = row["_open_d1"]
            close_d1 = row["_close_d1"]
            low_d1 = row.get("_low_d1", np.nan)
            if pd.isna(close_d1):
                continue
            sell_price = None
            if not pd.isna(open_d1):
                open_gap = open_d1 / buy_price - 1
                if take_profit and open_gap >= take_profit:
                    # 高开达到止盈阈值，开盘价止盈卖出
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
                high_d1 = row.get("_high_d1", np.nan)
                if not pd.isna(high_d1):
                    high_ret = high_d1 / buy_price - 1
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
    monthly_returns = {}
    for month_key in sorted(monthly.keys()):
        month_rets = monthly[month_key]
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

    # 2026年月度收益
    monthly = m.get("monthly_returns", {})
    monthly_2026 = {k: v for k, v in monthly.items() if k.startswith("2026")}
    if monthly_2026:
        logger.info(f"    --- 2026年月度收益 ---")
        for month_key, mm in sorted(monthly_2026.items()):
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
