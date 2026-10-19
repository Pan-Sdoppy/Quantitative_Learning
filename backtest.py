"""
回测入口文件
============
功能：
1. 加载已训练的回测模型
2. 基于训练文件的最新数据，在验证集上回测1天持仓动态卖出：
      - 1d_close: T日尾盘买，动态卖出（开盘止盈按pred分段1.5%/3%，盘中止盈4.0%，不止损，弱信号空仓）
3. 每次买3只股票，输出累计收益、年化、夏普、最大回撤、胜率、年度收益等指标
4. 保存交易明细和净值曲线（覆盖保存）
5. 已启用当天涨幅过滤：T日日涨幅>9.0%剔除涨停股（不设下限）

用法：
    python backtest.py
"""
import os
import warnings

import pandas as pd

from core.backtest_engine import run_backtest, save_backtest_results
from core.data_loader import load_all_data
from core.factors import compute_factors
from core.model import EnsembleModel
from core.utils import PROCESSED_DIR, MODEL_DIR, get_logger

warnings.filterwarnings("ignore")
logger = get_logger("backtest")

BACKTEST_TAG = "latest/backtest"  # 回测自动加载回测模型（训练到2021-12-30，回测从2022-01-04起）
TOP_N = 3
HORIZONS = ["1d_close"]
# 当天日涨幅区间过滤：T日日涨幅在(无下限, +9.0%)之外的剔除
# 上限>9.0%剔除涨停股（尾盘封单买不进）；不设下限（网格搜索最优：暴跌股次日反弹概率高，无下限收益最高）
FILTER_TODAY_DAY_RET = 0.090  # T日日涨幅>9.0%剔除（涨停股买不进）
FILTER_TODAY_DAY_RET_MIN = None  # 不设下限（网格搜索最优：无下限收益最高）
TAKE_PROFIT = float(os.environ.get("OPEN_TP", "0.020"))  # 开盘止盈（环境变量控制，默认2.0%）
STOP_LOSS = None  # 不止损（回测最优：-5%太紧会被洗盘甩出，不设止损收益最高）
INTRADAY_TAKE_PROFIT = float(os.environ.get("INTRADAY_TP", "0.040"))  # 盘中最高价止盈（环境变量控制，默认4.0%）
# 分段动态开盘止盈（按当日Top1 pred）：pred<1%用1.5%、pred≥1%用3%（回测+9.4%，2024/25/26三年一致）
# tuple=(pred分界, 弱信号tp, 强信号tp)；启用后 TAKE_PROFIT 仅作回退基准
# 环境变量DYNAMIC_TP格式："0.01,0.015,0.03"，设为"none"关闭动态止盈
_dtp_env = os.environ.get("DYNAMIC_TP", "0.01,0.015,0.03")
if _dtp_env.lower() == "none":
    DYNAMIC_TP = None
else:
    _parts = _dtp_env.split(",")
    DYNAMIC_TP = (float(_parts[0]), float(_parts[1]), float(_parts[2]))
# 二次排序：模型粗筛Top10后，按综合规则重排选Top3。两部分：
#   ① 基础5因子（权重见 RERANK_WEIGHTS，顺序 [振幅小, 5日量比小, 跳空高, 5日位置高, 3日动量高]）
#   ② K线形态（上影线短/光头阳，权重常量在 backtest_engine）+ 规模/量能/资金流3因子（mv_log_yi/vol_ratio_10/mf_intraday，权重各1）
# 完整回测：R2(min0.5) +2.768%/81.56% → 三因子(min1.0) +3.470%/89.57%，夏普12.95→14.98，回撤-25.1%→-15.2%
RERANK_CANDIDATES = 10
RERANK_METHOD = "composite"  # composite=5基础因子+K线形态+规模/量能/资金流综合rank排序
# 权重顺序: [振幅小, 5日量比小, 跳空高, 5日位置高, 3日动量高]（仅5个基础因子，各因子均"越小越靠前"）
RERANK_WEIGHTS = [2, 2, 1, 1, 1]  # 振幅2+5日量比2，其余各1（网格扫描1:1:1三因子最优点，权重不敏感）

# 信号强度过滤：二次排序后Top1的pred<阈值则空仓（环境变量MIN_PRED控制，默认0.010）
# 阈值扫描（2022起，全年度单调）：min1.0 → 844开仓天/+3.470%/胜率89.57%/夏普14.98/回撤-15.2%（用户选定，夏普最高档）
MIN_PRED = float(os.environ.get("MIN_PRED", "0.010"))


def main():
    logger.info("=" * 70)
    logger.info("A股量化策略回测开始")
    logger.info(f"配置: 每次买{TOP_N}只, 1天持仓动态卖出={HORIZONS}, 模型={BACKTEST_TAG}")
    logger.info(f"当天涨幅过滤: T日[无下限, {FILTER_TODAY_DAY_RET * 100:.1f}%]区间外剔除")
    logger.info(
        f"开盘止盈(动态): pred<1%为1.5%、pred≥1%为3%；盘中止盈{INTRADAY_TAKE_PROFIT * 100:.1f}%，{'不止损' if STOP_LOSS is None else f'止损{STOP_LOSS * 100:.0f}%'}")
    logger.info(f"二次排序: Top{RERANK_CANDIDATES}候选→{RERANK_METHOD}综合排序→Top{TOP_N}")

    # ========== Step 1: 加载因子数据（基于训练文件的最新数据） ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: 加载因子数据")
    factor_path = PROCESSED_DIR / "factor_data.parquet"
    factor_data = None
    if factor_path.exists():
        try:
            logger.info(f"从缓存加载因子数据: {factor_path}")
            factor_data = pd.read_parquet(factor_path)
        except Exception as e:
            logger.warning(f"缓存读取失败({e})，重新计算")

    if factor_data is None:
        logger.info("计算因子...")
        all_data = load_all_data()
        factor_data, _ = compute_factors(all_data, skip_redundancy=True)
        factor_data.to_parquet(factor_path, index=False)

    # 兼容旧因子缓存：若缺少low列（盘中最低价止损需要），从raw数据补齐
    if "low" not in factor_data.columns:
        logger.info("因子缓存缺少low列，从raw数据补齐（盘中最低价止损需要）...")
        import glob as _glob
        raw_dir = r"C:\Users\Pan\Documents\GitHub\Quantitative_Learning\data\raw"
        low_parts = []
        for fp in _glob.glob(raw_dir + "\\*.parquet"):
            try:
                fname = os.path.basename(fp).replace(".parquet", "")
                if not fname.isdigit():
                    continue  # 跳过 _meta.json 等元数据文件
                sub = pd.read_parquet(fp, columns=["date", "low"])
                sub["code"] = fname  # raw按股票存文件，文件名为代码，无code列
                low_parts.append(sub)
            except Exception:
                continue
        low_df = pd.concat(low_parts, ignore_index=True) if low_parts else pd.DataFrame(columns=["date", "code", "low"])
        low_df["date"] = pd.to_datetime(low_df["date"])
        factor_data["date"] = pd.to_datetime(factor_data["date"])
        factor_data = factor_data.merge(low_df, on=["date", "code"], how="left")
        del low_parts, low_df
        import gc;
        gc.collect()
        logger.info(f"low列补齐完成，缺失: {int(factor_data['low'].isna().sum())} 行")

    logger.info(f"因子数据: {len(factor_data)} 行, {factor_data['code'].nunique()} 只股票")
    logger.info(f"因子数据范围: {factor_data['date'].min():%Y-%m-%d} ~ {factor_data['date'].max():%Y-%m-%d}")

    # ========== Step 2: 加载模型 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 加载已训练模型")
    models = {}
    for h in HORIZONS:
        try:
            model = EnsembleModel(horizon=h)
            model.load(tag=BACKTEST_TAG)
            models[h] = model
            logger.info(f"  1天持仓-动态卖出 ({h}) 模型加载成功")
        except Exception as e:
            logger.error(f"  1天持仓-动态卖出 ({h}) 模型加载失败: {e}，跳过该模式")

    if not models:
        logger.error("没有可用的模型，请先运行 python train.py")
        return

    # ========== Step 3: 运行回测 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: 运行回测")

    # 严格无泄露：优先读取 checkpoint 记录的回测起点；旧模型无该字段时回退到训练截止日+1
    start_dt = None
    cp_path = MODEL_DIR / BACKTEST_TAG / "_checkpoint.json"
    if cp_path.exists():
        try:
            import json
            with open(cp_path, encoding="utf-8") as f:
                cp = json.load(f)
            if cp.get("backtest_start"):
                start_dt = cp["backtest_start"]
                logger.info(f"模型记录回测起点: {start_dt}（训练截止 {cp.get('train_end_date')}，严格样本外）")
            else:
                train_end = cp.get("train_end_date")
                if train_end:
                    start_dt = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info(f"训练数据截止: {train_end}，回测从 {start_dt} 开始（验证集，无数据泄露）")
        except Exception:
            pass

    bt_end = factor_data["date"].max().strftime("%Y-%m-%d")
    logger.info(f"回测区间: {start_dt or '全部'} ~ {bt_end}")

    available_horizons = list(models.keys())
    results = run_backtest(
        factor_data=factor_data,
        models=models,
        top_n=TOP_N,
        horizons=available_horizons,
        start_date=start_dt,
        end_date=None,
        take_profit=TAKE_PROFIT,  # 开盘止盈基准（被dynamic_tp分段动态覆盖）
        stop_loss=STOP_LOSS,  # 不止损
        filter_today_day_ret=FILTER_TODAY_DAY_RET,  # T日日涨幅>9.0%剔除
        filter_today_day_ret_min=FILTER_TODAY_DAY_RET_MIN,  # 不设下限
        intraday_take_profit=INTRADAY_TAKE_PROFIT,  # 盘中最高价涨4.0%止盈
        rerank_candidates=RERANK_CANDIDATES,
        rerank_method=RERANK_METHOD,
        rerank_weights=RERANK_WEIGHTS,
        min_pred=MIN_PRED,  # 二次排序后Top1 pred<1.0%空仓
        dynamic_tp=DYNAMIC_TP,  # 按pred分段动态开盘止盈
    )

    # ========== Step 4: 保存结果 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 保存回测结果")
    summary = save_backtest_results(results, tag=BACKTEST_TAG)

    # ========== 输出汇总 ==========
    logger.info("\n" + "=" * 70)
    logger.info("回测汇总:")
    # 仅打印核心指标列，排除超长的 monthly_returns/weights 字典列（避免终端表格被撑爆显示异常）
    display_cols = ["sell_mode", "total_return", "annual_return", "sharpe",
                    "max_drawdown", "win_rate", "profit_loss_ratio",
                    "avg_return_per_trade", "num_trades"]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(index=False))

    # 年度收益汇总（2024/2025/2026），不打印月度收益；月度明细见 results/latest/backtest/monthly_returns.csv
    for h, res in results.items():
        yr = res["metrics"].get("yearly_returns", {})
        if yr:
            logger.info(f"\n年度收益 ({h}):")
            yr_df = pd.DataFrame([{"year": y, **v} for y, v in sorted(yr.items())])
            print(yr_df.to_string(index=False))
    failed = [h for h in HORIZONS if h not in models]
    if failed:
        logger.warning(f"以下模式因模型加载失败被跳过: {failed}")
    logger.info("\n回测完成！详细结果见 results/ 目录")


if __name__ == "__main__":
    main()
