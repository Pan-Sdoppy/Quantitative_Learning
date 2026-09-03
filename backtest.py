"""
回测入口文件
============
功能：
1. 加载已训练的回测模型
2. 基于训练文件的最新数据，在验证集上回测1天持仓动态卖出：
      - 1d_close: T日尾盘买，动态卖出（高开2.5%止盈+跌5%止损）
3. 每次买3只股票，输出累计收益、年化、夏普、最大回撤、胜率、年度收益等指标
4. 保存交易明细和净值曲线（覆盖保存）

用法：
    python backtest.py
"""
import warnings

import pandas as pd

from core.backtest_engine import run_backtest, save_backtest_results
from core.data_loader import load_all_data
from core.factors import compute_factors
from core.model import EnsembleModel
from core.utils import PROCESSED_DIR, MODEL_DIR, get_logger

warnings.filterwarnings("ignore")
logger = get_logger("backtest")

BACKTEST_TAG = "latest/backtest"  # 回测自动加载回测模型（训练到2023-12-31）
TOP_N = 3
HORIZONS = ["1d_close"]


def main():
    logger.info("=" * 70)
    logger.info("A股量化策略回测开始")
    logger.info(f"配置: 每次买{TOP_N}只, 1天持仓动态卖出={HORIZONS}, 模型={BACKTEST_TAG}")

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
        factor_data, _ = compute_factors(all_data)
        factor_data.to_parquet(factor_path, index=False)

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

    # 从训练结束日的下一天开始回测（验证集），避免数据泄露
    start_dt = None
    cp_path = MODEL_DIR / BACKTEST_TAG / "_checkpoint.json"
    if cp_path.exists():
        try:
            import json
            with open(cp_path) as f:
                cp = json.load(f)
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
    )

    # ========== Step 4: 保存结果 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 保存回测结果")
    summary = save_backtest_results(results, tag=BACKTEST_TAG)

    # ========== 输出汇总 ==========
    logger.info("\n" + "=" * 70)
    logger.info("回测汇总:")
    print(summary.to_string(index=False))
    failed = [h for h in HORIZONS if h not in models]
    if failed:
        logger.warning(f"以下模式因模型加载失败被跳过: {failed}")
    logger.info("\n回测完成！详细结果见 results/ 目录")


if __name__ == "__main__":
    main()
