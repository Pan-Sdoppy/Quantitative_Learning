"""
实验：空仓日回退买入"pred最高可买入股票" 与 "pred最高可买入股>阈值才买" 对比回测
============================================================================
背景（用户需求）：
  1. 变体A：回测判定空仓（二次排序后Top1 pred < min_pred）时，改为买入"pred最高的可买入股票"
  2. 变体B：买入判定不看二次排序Top1，改为"过滤后全市场pred最高的可买入股票"是否 >= min_pred
对比铁律（完全相同样本集并排）：
  - 同股票池/同时段/同过滤条件：factor_data.parquet 缓存 + 模型 latest/backtest，回测起点 2022-01-04
  - 维度 = 开仓天数 + 平均收益 + 胜率 + 相关系数（相关系数按全交易日日收益序列，空仓日=0，与基线Pearson）
  - 区分口径：A1=候选池(二次排序Top10)内pred最高；A2=过滤后全市场pred最高；B=过滤后全市场pred最高直判
  - 阈值多档从松到紧：min_pred = {0.8%, 0.9%, 1.0%, 1.1%, 1.2%} 全扫
  - 开仓天数几乎一样 = 阈值没起作用（须逐档核对）
  - 不改生产代码：引擎实验参数 empty_fallback / buy_predmax_only 默认关闭
基线（2026-11-02 18:22 生产回测日志）：836开仓天 / 累计188867030255655.47% / 单次均收3.4740% / 胜率89.23% / 夏普14.4844 / 回撤-16.18%
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtest_engine import run_backtest
from core.model import EnsembleModel
from core.utils import MODEL_DIR, PROCESSED_DIR, get_logger

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Pan\Documents\GitHub\Quantitative_Learning")
OUT_DIR = ROOT / "results" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger("exp_empty_fallback")

BACKTEST_TAG = "latest/backtest"
TOP_N = 3
HORIZONS = ["1d_close"]
FILTER_TODAY_DAY_RET = 0.090
TAKE_PROFIT = 0.02
INTRADAY_TAKE_PROFIT = 0.04
DYNAMIC_TP = (0.01, 0.015, 0.03)
RERANK_CANDIDATES = 10
RERANK_METHOD = "composite"
RERANK_WEIGHTS = [2, 2, 1, 1, 1]

MIN_PRED_GRID = [0.008, 0.009, 0.010, 0.011, 0.012]  # 0.8% ~ 1.2%，从松到紧
# 4 方案：基线 / A1候选池回退 / A2全市场回退 / B全市场predmax直判
CONFIGS = [
    ("基线(生产)", dict(empty_fallback=None, buy_predmax_only=False)),
    ("A1_候选池回退", dict(empty_fallback="cand", buy_predmax_only=False)),
    ("A2_全市场回退", dict(empty_fallback="market", buy_predmax_only=False)),
    ("B_全市场predmax直判", dict(empty_fallback=None, buy_predmax_only=True)),
]


def load_factor_data():
    factor_path = PROCESSED_DIR / "factor_data.parquet"
    logger.info(f"加载因子缓存: {factor_path}")
    fd = pd.read_parquet(factor_path)
    if "low" not in fd.columns:
        logger.warning("因子缓存缺少low列（实验不补，沿用缓存）")
    return fd


def load_model():
    model = EnsembleModel(horizon="1d_close")
    model.load(tag=BACKTEST_TAG)
    return model


def daily_return_series(res):
    """从回测结果 trades 构建全交易日日收益序列（空仓日=0.0）"""
    trades = res["trades"]
    if trades is None or (isinstance(trades, pd.DataFrame) and trades.empty) or (not isinstance(trades, pd.DataFrame) and len(trades) == 0):
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")["return"].sort_index()
    # 用净值曲线日期范围扩展全交易日（空仓日 return 记 0）
    eq = pd.DataFrame(res["equity_curve"])
    eq["date"] = pd.to_datetime(eq["date"])
    full = pd.Series(0.0, index=pd.date_range(eq["date"].min(), eq["date"].max(), freq="B"))
    full.loc[df.index] = df
    return full


def main():
    factor_data = load_factor_data()
    model = load_model()
    # 回测起点：与生产一致（模型 checkpoint）
    start_dt = None
    cp_path = MODEL_DIR / BACKTEST_TAG / "_checkpoint.json"
    if cp_path.exists():
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
        if cp.get("backtest_start"):
            start_dt = cp["backtest_start"]
        elif cp.get("train_end_date"):
            start_dt = (pd.Timestamp(cp["train_end_date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"回测起点: {start_dt}，区间: {start_dt} ~ {factor_data['date'].max():%Y-%m-%d}")

    rows = []
    series_store = {}

    for cfg_name, kw in CONFIGS:
        for mp in MIN_PRED_GRID:
            key = f"{cfg_name}|min_pred={mp:.3f}"
            logger.info("=" * 60)
            logger.info(f"运行: {key}")
            res = run_backtest(
                factor_data=factor_data,
                models={"1d_close": model},
                top_n=TOP_N,
                horizons=["1d_close"],
                start_date=start_dt,
                end_date=None,
                take_profit=TAKE_PROFIT,
                stop_loss=None,
                filter_today_day_ret=FILTER_TODAY_DAY_RET,
                filter_today_day_ret_min=None,
                intraday_take_profit=INTRADAY_TAKE_PROFIT,
                rerank_candidates=RERANK_CANDIDATES,
                rerank_method=RERANK_METHOD,
                rerank_weights=RERANK_WEIGHTS,
                min_pred=mp,
                dynamic_tp=DYNAMIC_TP,
                empty_fallback=kw["empty_fallback"],
                buy_predmax_only=kw["buy_predmax_only"],
            )
            r = res["1d_close"]
            m = r["metrics"]
            trades = r["trades"]
            open_days = len(trades)
            avg_ret = m.get("avg_return_per_trade")
            win_rate = m.get("win_rate")
            total_ret = m.get("total_return")
            sharpe = m.get("sharpe")
            mdd = m.get("max_drawdown")
            empty_cnt = len(r["empty_days"])
            series_store[key] = daily_return_series(r)
            rows.append({
                "方案": cfg_name,
                "min_pred": mp,
                "开仓天数": open_days,
                "空仓天数": empty_cnt,
                "总交易日": open_days + empty_cnt,
                "单次均收%": round(avg_ret * 100, 4) if avg_ret is not None else None,
                "胜率%": round(win_rate * 100, 2) if win_rate is not None else None,
                "总收益%": round(total_ret, 2) if total_ret is not None else None,
                "夏普": round(sharpe, 4) if sharpe is not None else None,
                "最大回撤%": round(mdd * 100, 2) if mdd is not None else None,
            })
            logger.info(f"完成: {key} -> 开仓{open_days}天/空仓{empty_cnt}天, 均收{avg_ret * 100:.4f}%, "
                        f"胜率{win_rate * 100:.2f}%, 总收益{total_ret:.2f}%, 夏普{sharpe:.4f}, 回撤{mdd * 100:.2f}%")

    df = pd.DataFrame(rows)
    # 相关系数：基线 min_pred=0.010 与所有方案（同档/跨档都算：与同档基线算，跨档用基线同min_pred档）
    base_key = f"基线(生产)|min_pred=0.010"
    base_s = series_store.get(base_key)
    corr = []
    for _, row in df.iterrows():
        key = f"{row['方案']}|min_pred={row['min_pred']:.3f}"
        s = series_store.get(key)
        if base_s is not None and s is not None:
            c = base_s.corr(s) if len(base_s) > 1 else np.nan
            corr.append(round(float(c), 4) if not np.isnan(c) else None)
        else:
            corr.append(None)
    df["相关系数(与基线1.0%)"] = corr

    # 再补：同方案内部跨档对照基线同档（相关系数列已覆盖同min_pred档，因key按方案+档）
    df = df.sort_values(["方案", "min_pred"])
    df["min_pred"] = df["min_pred"].apply(lambda x: f"{x * 100:.1f}%")

    xlsx_path = OUT_DIR / "empty_fallback_对比_20261102.xlsx"
    df.to_excel(xlsx_path, index=False)
    logger.info(f"并排对比表已保存: {xlsx_path}")

    print("\n" + "=" * 100)
    print("空仓回退策略并排对比（同一样本集 2022-01-04 ~ 2026-10-30，模型 latest/backtest）")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\n对比表: {xlsx_path}")


if __name__ == "__main__":
    main()
