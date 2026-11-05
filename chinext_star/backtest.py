"""
回测入口文件
============
功能：
1. 加载已训练的回测模型（models/latest/backtest，训练止于 2021-12-30，回测从 2022-01-04 起，严格样本外）
2. 在样本外区间回测 1 天持仓动态卖出（B16 生产口径）：
      - 1d_close: T日尾盘买，T+1 动态卖出（开盘止盈按 pred 分段 1.2%/2.5%；移动止盈：冲高 0.75% 激活、
        从高点回撤 0.25% 卖出；不止损，弱信号空仓）
      - 追高过滤：T日日涨幅 ≥12% 剔除、≤-6% 剔除（双创 20cm 固定阈值；下限剔除暴跌股）
      - 大盘择时：沪深300当日收阴（hs300_ret_1d<0）则空仓；极端熊市择时已关闭
3. 模型粗筛 Top8 候选 → 池内剔除极端放量票 → composite 二次排序选 Top3、全仓买 Top1（权重 100/0/0），
   输出累计收益、年化、夏普、最大回撤、胜率、年度/月度收益等指标
4. 每个交易日都有"开仓/空仓"交代：空仓日按原因分组计数（逐条走 DEBUG），年度/月度含开仓/空仓天数
   （0 交易空仓月也列出）；保存交易明细、空仓日明细、净值曲线（覆盖保存）

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

BACKTEST_TAG = "latest/backtest"  # 回测自动加载回测模型（训练到2021-12-30，回测从2022-01-04起）
TOP_N = 3
HORIZONS = ["1d_close"]

# ===== B16 生产参数（守卫并排回测验证落地，禁止随意改动；改动前必须先跑完整回测证明超基线）=====
# 当天日涨幅区间过滤：双创固定 12% 追高阈值（T日日涨幅≥12%剔除）；下限 -6% 剔除 T 日跌超 6% 的暴跌股
FILTER_TODAY_DAY_RET = 0.12
FILTER_TODAY_DAY_RET_MIN = -0.06
# 大盘当日收阴空仓（沪深300当日收益<0则不买）；极端熊市择时关闭（-1.0）
SKIP_IF_INDEX_DOWN = True
MARKET_TIMING_EMPTY = -1.0
TAKE_PROFIT = 0.020  # 开盘止盈基准（被 dynamic_tp 分段动态覆盖）
STOP_LOSS = None  # 不止损
INTRADAY_TAKE_PROFIT = 0.070  # 固定盘中止盈 7.0%（B16 启用移动止盈后被互斥短路，保留传参仅作兼容）
# 分段动态开盘止盈（按当日 Top1 pred）：tuple=(pred 分界, 弱信号 tp, 强信号 tp)；pred<1% 用 1.2%、pred≥1% 用 2.5%
DYNAMIC_TP = (0.01, 0.012, 0.025)
# 移动止盈（B16）：开盘动态止盈优先；盘中冲高达 TRAILING_START 激活，从当日高点回撤 TRAILING_DD 卖出；
# 未激活（高点 < TRAILING_START）或收盘仍在回撤线上方则持有到收盘兜底。None=关闭、退回固定盘中止盈 7%
TRAILING_START = 0.0075
TRAILING_DD = 0.0025

# 二次排序：模型粗筛 Top8 后按 composite 综合规则重排选 Top3
#   ① 基础 5 因子（权重 RERANK_WEIGHTS，顺序 [振幅小, 5日量比小, 跳空高, 5日位置高, 3日动量高]，5日位置项权重 0）
#   ② K线形态（上影线短/光头阳，权重常量在 backtest_engine）
#   ③ 规模/量能/资金流 3 因子（mv_log_yi/vol_ratio_10/mf_intraday，资金流半权重）
RERANK_CANDIDATES = 8
RERANK_METHOD = "composite"
RERANK_WEIGHTS = [4, 2, 1, 0, 1]
# 候选池内因子过滤：list of (col, direction, q)，在 pred Top8 候选内按当日横截面 rank 剔除；
# dir='hi' 剔除分位>q（极端放量）；过滤后为空则该因子跳过（引擎保证至少可买）
CANDIDATE_FACTORS = [("mf_net_proxy", "hi", 0.65)]

# 信号强度过滤：二次排序后 Top1 的 pred < 0.010 当日空仓（作用于二次排序后的 Top1）
MIN_PRED = 0.010


def main():
    logger.info("=" * 70)
    logger.info("双创量化策略回测开始")
    logger.info(f"配置: 每次买{TOP_N}只, 1天持仓动态卖出={HORIZONS}, 模型={BACKTEST_TAG}")
    logger.info(f"当天涨幅过滤: T日[{FILTER_TODAY_DAY_RET_MIN * 100:.1f}%, {FILTER_TODAY_DAY_RET * 100:.1f}%]区间外剔除（B16）")
    logger.info(f"大盘择时: 沪深300当日收阴空仓={'启用' if SKIP_IF_INDEX_DOWN else '关闭'}; 极端熊市择时={'启用' if MARKET_TIMING_EMPTY > -0.9 else '关闭'}（B16关闭）")
    _tp_low = DYNAMIC_TP[1] if DYNAMIC_TP else TAKE_PROFIT
    _tp_high = DYNAMIC_TP[2] if DYNAMIC_TP else TAKE_PROFIT
    logger.info(
        f"开盘止盈(动态): pred<1%为{_tp_low * 100:.1f}%、pred≥1%为{_tp_high * 100:.1f}%；{'不止损' if STOP_LOSS is None else f'止损{STOP_LOSS * 100:.0f}%'}")
    if TRAILING_START is not None:
        logger.info(f"移动止盈(B16): 盘中冲高≥{TRAILING_START * 100:.2f}%激活，从高点回撤{TRAILING_DD * 100:.2f}%卖出；未激活或收盘在回撤线上方持有到收盘")
    else:
        logger.info(f"固定盘中止盈{INTRADAY_TAKE_PROFIT * 100:.1f}%（移动止盈关闭）")
    logger.info(f"二次排序(B16): Top{RERANK_CANDIDATES}候选→池内剔除mf_net_proxy极端放量(>65%分位)→{RERANK_METHOD}综合排序→Top{TOP_N}")

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
        filter_today_day_ret=FILTER_TODAY_DAY_RET,  # B16：双创固定T日涨幅≥12%剔除
        filter_today_day_ret_min=FILTER_TODAY_DAY_RET_MIN,  # B16：剔除T日跌超6%的暴跌股
        intraday_take_profit=INTRADAY_TAKE_PROFIT,  # 固定盘中7%（B16移动止盈启用后被互斥短路，保留兼容）
        skip_if_index_down=SKIP_IF_INDEX_DOWN,  # 沪深300当日收阴空仓（B16保留）
        market_timing_empty=MARKET_TIMING_EMPTY,  # B16：关闭极端熊市择时（-1.0）
        rerank_candidates=RERANK_CANDIDATES,
        rerank_method=RERANK_METHOD,
        rerank_weights=RERANK_WEIGHTS,
        candidate_factors=CANDIDATE_FACTORS,  # 候选池内剔除 mf_net_proxy 极端放量票
        min_pred=MIN_PRED,  # 二次排序后Top1 pred<1.0%空仓
        dynamic_tp=DYNAMIC_TP,  # 按pred分段动态开盘止盈
        trailing_start=TRAILING_START,  # B16：移动止盈激活线 0.75%
        trailing_dd=TRAILING_DD,  # B16：从高点回撤 0.25% 触发卖出
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

    # 逐年收益汇总（空仓年也列出）；月度明细见 results/latest/backtest/monthly_returns.csv
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
