"""
训练入口文件
============
功能：
1. 增量下载A股全市场数据（自动剔除ST/退市/次新/停牌）
2. 计算技术因子 + 横截面标准化（缓存复用）
3. 训练1天持仓动态卖出的 LightGBM + XGBoost + MLP 集成模型
      - 1d_close: T日尾盘买，T+1动态卖
4. 严格无泄露切分（训练 2016-2021，回测 2022-01-04 起）：
   - 因子去冗余相关性矩阵只用 <=2021-12-31 的样本，未来数据不参与因子筛选
   - 回测模型阶段1：训练<=2020-12-31 + 2021全年验证（至2021-12-30）做早停，定迭代轮数
   - 回测模型阶段2：固定轮数，用<=2021-12-30 全量样本refit（模型最晚见到2021-12-31的价格）
   - 候选回测模型先存临时目录，跑 2022-01-04 起回测与现有基线比总收益：
     不劣于基线才转正，并继续训练预测模型（100%全量数据、轮数继承阶段1早停）；
     低于基线则丢弃候选、保留原回测+预测模型，不更新。回测与预测必须一起训，不允许只训其一。
5. 模型覆盖保存，不产生垃圾文件

用法：
    python train.py              # 用本地存量数据训练（先回测守卫，通过后再训预测）
    python train.py --update     # 先增量下载截止当前最新交易日的数据，再训练
    python train.py --data-only  # 只增量更新日K/沪深300，不训练
"""
import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from core.backtest_engine import run_backtest, save_backtest_results
from core.data_loader import update_data, load_all_data, update_hs300
from core.factors import compute_factors, create_labels
from core.model import EnsembleModel
from core.utils import PROCESSED_DIR, MODEL_DIR, RESULT_DIR, get_logger, detect_gpu

warnings.filterwarnings("ignore")
logger = get_logger("train")

# 两种模型各自的保存路径
BACKTEST_TAG = "latest/backtest"  # 回测模型：训练到2021-12-30，回测从2022-01-04开始
PREDICT_TAG = "latest/predict"  # 预测模型：100%全量数据训练，迭代次数从阶段1早停继承
BACKTEST_CANDIDATE_TAG = "latest/_backtest_candidate"  # 新回测模型暂存目录（守卫通过前不覆盖正式模型）
BACKTEST_STAGE1_TAG = "latest/_bt_stage1"  # 回测模型阶段1（早停）暂存目录，阶段2训完即删

# 固定配置
START_DATE = "20160101"
HORIZONS_DEFAULT = ["1d_close"]

# ===== 严格无泄露切分（训练 2016-2021，回测 2022-01-04 起）=====
# 因子去冗余：相关性矩阵只用 <= REDUNDANCY_CUTOFF 的样本（2021-12-31 因子值仅依赖当天收盘前数据，无未来信息）
REDUNDANCY_CUTOFF = "2021-12-31"
# 回测模型阶段1：训练集 <= STAGE1_TRAIN_END；验证集 STAGE1_TRAIN_END+1 ~ STAGE1_VAL_END（2021全年）
# 2021-12-31 特征日的标签需 2022-01-04 价格，故验证集止于 2021-12-30（标签 2021-12-31 实现）
STAGE1_TRAIN_END = "2020-12-31"
STAGE1_VAL_END = "2021-12-30"
# 回测模型阶段2：固定轮数全量 refit，训练集 <= BT_TRAIN_END（模型最晚见到 2021-12-31 价格）
BT_TRAIN_END = "2021-12-30"
# 回测起点（引擎自动取首个交易日 2022-01-04）
BACKTEST_START = "2022-01-01"
BACKTEST_FIRST_DAY = "2022-01-04"

# 守卫回测参数：必须与 predict.py 完全一致（Top1全仓 + composite二次排序 + 动态止盈 + 弱信号空仓）
# composite = 5基础因子[振幅2,5日量比2,跳空1,5日位置1,3日动量1] + K线形态(上影1/光头0.5)
#             + 规模/量能/资金流3因子(流通市值/10日量比/日内资金流，各1，定义在 backtest_engine 常量)
# min_pred=1.0%：2022-01起严格样本外全阈值扫描，1.0%夏普最高（14.99），兼顾交易次数与胜率
GUARD_BT_KW = dict(
    top_n=3,
    take_profit=0.020, stop_loss=None,
    filter_today_day_ret=0.090, filter_today_day_ret_min=None,
    intraday_take_profit=0.040,
    rerank_candidates=10, rerank_method="composite",
    rerank_weights=[2, 2, 1, 1, 1],
    min_pred=0.010,
    dynamic_tp=(0.01, 0.015, 0.03),
)


def parse_args():
    parser = argparse.ArgumentParser(description="A股量化模型训练")
    parser.add_argument("--update", action="store_true",
                        help="先增量下载截止当前最新交易日的数据，再训练")
    parser.add_argument("--data-only", action="store_true",
                        help="仅增量更新日K和沪深300数据到最新交易日，不计算因子、不训练模型；"
                             "之后手动训练时因子缓存会自动检测数据过期并重算")
    return parser.parse_args()


def load_checkpoint(tag: str) -> dict:
    """加载训练断点"""
    cp_path = MODEL_DIR / tag / "_checkpoint.json"
    if cp_path.exists():
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_checkpoint(tag: str, cp: dict):
    """保存训练断点"""
    cp_path = MODEL_DIR / tag / "_checkpoint.json"
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(cp, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"断点保存失败: {e}")


def train_models_with_split(labeled, factor_cols, horizons, split_ratio=0.8, save_tag="",
                            mode_name="", split_date=None, val_end=None):
    """用指定分割比例或固定日期训练一套模型，保存到save_tag目录
    split_date: 固定分割日期（如"2020-12-31"），优先于split_ratio
    val_end: 验证集截止特征日（含），None 时验证集=split_date之后全部数据。
             严格无泄露切分必须传入，防止未来样本进入早停验证集。
    """
    if split_date:
        train_end = pd.Timestamp(split_date)
    else:
        all_dates = sorted(labeled["date"].unique())
        split_idx = int(len(all_dates) * split_ratio)
        train_end = all_dates[split_idx]
    train_df = labeled[labeled["date"] <= train_end]
    if val_end is not None:
        val_df = labeled[(labeled["date"] > train_end) & (labeled["date"] <= pd.Timestamp(val_end))]
    else:
        val_df = labeled[labeled["date"] > train_end]
    train_start = train_df["date"].min()
    val_max = val_df["date"].max() if len(val_df) > 0 else train_end
    logger.info(f"[{mode_name}] 训练数据: {train_start:%Y-%m-%d} ~ {train_end:%Y-%m-%d} ({len(train_df)}样本)")
    logger.info(
        f"[{mode_name}] 验证数据: {train_end + pd.Timedelta(days=1):%Y-%m-%d} ~ {val_max:%Y-%m-%d} ({len(val_df)}样本)")

    # 直接取float32（模型实际只用float32精度），避免float64全量数组与float32副本双份占用内存
    X_train = train_df[factor_cols].to_numpy(dtype=np.float32)
    X_val = val_df[factor_cols].to_numpy(dtype=np.float32)
    data_signature = f"{len(factor_cols)}f_{train_df['date'].max():%Y%m%d}_{len(train_df)}"
    train_end_date = train_end.strftime("%Y-%m-%d")

    # 每次运行都重新训练（断点文件仅记录本次训练进度，不从旧断点恢复，不跳过模型）
    trained_horizons = []

    logger.info(f"开始训练，数据签名: {data_signature}")
    cp = {"data_signature": data_signature, "trained_horizons": [],
          "factor_count": len(factor_cols), "train_end_date": train_end_date,
          "started_at": datetime.now().isoformat()}

    models = {}
    for h in horizons:
        is_last_h = (h == horizons[-1])
        logger.info(f"\n--- 训练 1天持仓-动态卖出 ({h}) 模型 ---")
        try:
            y_col = f"label_ret_{h}"
            y_train = train_df[y_col].to_numpy(dtype=np.float32)
            y_val = val_df[y_col].to_numpy(dtype=np.float32)
            mask_tr = np.abs(y_train) < 0.5
            mask_vl = np.abs(y_val) < 0.5
            X_tr = np.ascontiguousarray(X_train[mask_tr])
            y_tr = np.ascontiguousarray(y_train[mask_tr])
            X_vl = np.ascontiguousarray(X_val[mask_vl])
            y_vl = np.ascontiguousarray(y_val[mask_vl])
            del y_train, y_val, mask_tr, mask_vl
            # 最后一个horizon：numpy数组已全部取出，释放全量数组与DataFrame切片副本，压低训练峰值内存
            if is_last_h:
                del X_train, X_val, train_df, val_df
            import gc as _gc;
            _gc.collect()
            logger.info(f"有效训练样本: {len(X_tr)}, 验证样本: {len(X_vl)}")

            model = EnsembleModel(horizon=h)
            model.train(X_tr, y_tr, X_vl, y_vl, feature_names=factor_cols)
            model.save(tag=save_tag)
            models[h] = model

            if h not in trained_horizons:
                trained_horizons.append(h)
            cp["trained_horizons"] = trained_horizons
            cp["last_trained"] = datetime.now().isoformat()
            save_checkpoint(save_tag, cp)
            logger.info(f"1天持仓-动态卖出 ({h}) 模型训练完成并保存到 {save_tag}，断点已更新")

        except Exception as e:
            logger.error(f"1天持仓-动态卖出 ({h}) 模型训练失败: {e}", exc_info=True)
            logger.info(f"跳过 {h}，继续训练其他模型")
            continue

    return models


def train_models_full_data(labeled, factor_cols, horizons, save_tag,
                           fixed_iters, mode_name="预测模型"):
    """用给定样本全量训练一套模型（无验证集，固定迭代轮数）
    mode_name: 日志标识——"回测模型-阶段2refit"（<=2021）或"预测模型"（100%全量）
    fixed_iters: {horizon: {"lgb": int, "xgb": int, "mlp": int}} 各模型固定迭代次数
    """
    train_df = labeled  # 本函数只读取数据、不修改，直接引用，避免整表copy（数百万行可省数GB内存）
    train_start = train_df["date"].min()
    train_end = train_df["date"].max()
    logger.info(f"[{mode_name}] 训练数据: {train_start:%Y-%m-%d} ~ {train_end:%Y-%m-%d} ({len(train_df)}样本)")

    X_train = train_df[factor_cols].to_numpy(dtype=np.float32)
    data_signature = f"{len(factor_cols)}f_{train_end:%Y%m%d}_{len(train_df)}_full"
    train_end_date = train_end.strftime("%Y-%m-%d")

    # 每次运行都重新训练（断点文件仅记录本次训练进度，不从旧断点恢复）
    trained_horizons = []
    logger.info(f"开始训练（{mode_name}，无验证集，固定迭代轮数），数据签名: {data_signature}")
    cp = {"data_signature": data_signature, "trained_horizons": [],
          "factor_count": len(factor_cols), "train_end_date": train_end_date,
          "started_at": datetime.now().isoformat()}

    models = {}
    for h in horizons:
        is_last_h = (h == horizons[-1])
        logger.info(f"\n--- 训练 1天持仓-动态卖出 ({h}) 模型 [{mode_name}] ---")
        try:
            y_col = f"label_ret_{h}"
            y_train = train_df[y_col].to_numpy(dtype=np.float32)
            mask_tr = np.abs(y_train) < 0.5
            X_tr = np.ascontiguousarray(X_train[mask_tr])
            y_tr = np.ascontiguousarray(y_train[mask_tr])
            del y_train, mask_tr
            if is_last_h:
                del X_train, train_df
            import gc as _gc;
            _gc.collect()
            logger.info(f"有效训练样本: {len(X_tr)} (无验证集，固定迭代轮数)")

            fi = fixed_iters.get(h, {})
            logger.info(f"继承迭代次数: LGB={fi.get('lgb', '?')}, XGB={fi.get('xgb', '?')}, MLP={fi.get('mlp', '?')}")

            model = EnsembleModel(horizon=h)
            model.train(X_tr, y_tr, X_val=None, y_val=None, feature_names=factor_cols,
                        fixed_lgb_rounds=fi.get("lgb"), fixed_xgb_rounds=fi.get("xgb"),
                        fixed_mlp_epochs=fi.get("mlp"))
            model.save(tag=save_tag)
            models[h] = model

            if h not in trained_horizons:
                trained_horizons.append(h)
            cp["trained_horizons"] = trained_horizons
            cp["last_trained"] = datetime.now().isoformat()
            save_checkpoint(save_tag, cp)
            logger.info(f"1天持仓-动态卖出 ({h}) 模型训练完成并保存到 {save_tag}，断点已更新")

        except Exception as e:
            logger.error(f"1天持仓-动态卖出 ({h}) 模型训练失败 [预测模型]: {e}", exc_info=True)
            logger.info(f"跳过 {h}，继续训练其他模型")
            continue

    return models


def extract_best_iters(bt_models, horizons):
    """从回测模型中提取各模型最佳迭代次数（处理-1/None情况）"""
    result = {}
    for h in horizons:
        if h in bt_models:
            m = bt_models[h]
            # LightGBM: best_iteration可能为-1（早停未触发），用num_trees()兜底
            lgb_iter = getattr(m.lgb.model, "best_iteration", None)
            if not lgb_iter or lgb_iter <= 0:
                try:
                    lgb_iter = m.lgb.model.num_trees()
                except Exception:
                    lgb_iter = 500
            if lgb_iter <= 0:
                lgb_iter = 500
            # XGBoost: best_iteration可能为-1
            xgb_iter = getattr(m.xgb.model, "best_iteration", None)
            if not xgb_iter or xgb_iter <= 0:
                try:
                    xgb_iter = m.xgb.model.best_ntree_limit
                except Exception:
                    xgb_iter = 500
            if xgb_iter <= 0:
                xgb_iter = 500
            # MLP: best_epoch可能为None
            mlp_epoch = getattr(m.mlp, "best_epoch", None)
            if not mlp_epoch or mlp_epoch <= 0:
                mlp_epoch = 50
            result[h] = {"lgb": lgb_iter, "xgb": xgb_iter, "mlp": mlp_epoch}
        else:
            result[h] = {"lgb": 500, "xgb": 500, "mlp": 50}
    return result


def _load_baseline_metrics(horizon):
    """读取现有回测模型的基线指标（results/latest/backtest/backtest_summary.csv）。无基线返回None。
    口径校验：基线的回测起点必须与当前 BACKTEST_START 一致（_guard_meta.json 记录）；
    旧口径基线（如 2024 起）与新口径（2022 起）不可比，按无基线处理，候选直接采纳。
    """
    d = RESULT_DIR / BACKTEST_TAG
    p = d / "backtest_summary.csv"
    if not p.exists():
        return None
    meta_p = d / "_guard_meta.json"
    if not meta_p.exists():
        logger.warning("现有基线无口径标记（旧版训练产物），与当前切分口径不可比，按无基线处理（候选直接采纳）")
        return None
    try:
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("backtest_start") != BACKTEST_START:
            logger.warning(
                f"现有基线口径(回测{meta.get('backtest_start')}起, 训练止于{meta.get('train_end_date')})"
                f"与当前口径(回测{BACKTEST_START}起, 训练止于{BT_TRAIN_END})不一致，不可比，按无基线处理")
            return None
    except Exception as e:
        logger.warning(f"基线口径标记读取失败({e})，按无基线处理（候选模型直接采纳）")
        return None
    try:
        df = pd.read_csv(p)
        row = df[df["sell_mode"] == horizon]
        if row.empty:
            return None
        return {
            "total_return": float(row["total_return"].iloc[0]),
            "win_rate": float(row["win_rate"].iloc[0]),
            "sharpe": float(row["sharpe"].iloc[0]),
            "max_drawdown": float(row["max_drawdown"].iloc[0]),
        }
    except Exception as e:
        logger.warning(f"读取基线回测指标失败({e})，按无基线处理（候选模型直接采纳）")
        return None


def _write_guard_meta():
    """守卫转正后写入新基线的切分口径标记（供后续训练做同口径对比）。"""
    meta_p = RESULT_DIR / BACKTEST_TAG / "_guard_meta.json"
    try:
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump({
                "backtest_start": BACKTEST_START,
                "backtest_first_day": BACKTEST_FIRST_DAY,
                "train_end_date": BT_TRAIN_END,
                "redundancy_cutoff": REDUNDANCY_CUTOFF,
                "promoted_at": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"基线口径标记写入失败: {e}")


def guard_backtest_model(factor_data, candidate_models, horizons):
    """对候选回测模型跑回测，与现有基线比总收益。

    - 候选总收益 >= 基线（或无基线，首次训练）：转正（覆盖正式回测目录、保存回测结果），返回True
    - 候选总收益 < 基线：丢弃候选目录，保留原回测/预测模型，返回False（调用方据此不训练预测模型）
    判定只看总收益（收益率为最高准则），其余胜率/夏普/回撤仅打印参考。
    """
    h = horizons[0]
    if h not in candidate_models:
        logger.error("候选回测模型训练失败，无法进行守卫对比，本次训练终止，保留原模型")
        cand_dir = MODEL_DIR / BACKTEST_CANDIDATE_TAG
        shutil.rmtree(cand_dir, ignore_errors=True)
        return False

    logger.info(f"  4a.1 候选回测模型跑回测（{BACKTEST_FIRST_DAY}起，严格样本外），与现有基线比总收益...")
    new_results = run_backtest(
        factor_data=factor_data, models=candidate_models,
        horizons=horizons, start_date=BACKTEST_START, end_date=None,
        **GUARD_BT_KW,
    )
    nm = new_results[h]["metrics"]
    new_ret = nm["total_return"]
    base = _load_baseline_metrics(h)

    cand_dir = MODEL_DIR / BACKTEST_CANDIDATE_TAG
    tgt_dir = MODEL_DIR / BACKTEST_TAG

    if base is None:
        promoted = True
        logger.info("未找到现有回测基线（首次训练或口径变更），候选回测模型直接采纳")
    else:
        base_ret = base["total_return"]
        logger.info("=" * 70)
        logger.info(f"基线回测：总收益 {base_ret * 100:.1f}%，胜率 {base['win_rate'] * 100:.2f}%，"
                    f"夏普 {base['sharpe']}，最大回撤 {base['max_drawdown'] * 100:.1f}%")
        logger.info(f"候选回测：总收益 {new_ret * 100:.1f}%，胜率 {nm['win_rate'] * 100:.2f}%，"
                    f"夏普 {nm['sharpe']}，最大回撤 {nm['max_drawdown'] * 100:.1f}%")
        promoted = new_ret >= base_ret
        if promoted:
            logger.info(f"候选总收益不劣于基线（{new_ret * 100:.1f}% ≥ {base_ret * 100:.1f}%），通过守卫")
        else:
            logger.error("=" * 70)
            logger.error(f"候选回测总收益 {new_ret * 100:.1f}% 低于基线 {base_ret * 100:.1f}%，未优于原模型。")
            logger.error("已丢弃候选回测模型，原回测模型与预测模型保持不变，本次不再训练预测模型。")
            logger.error("=" * 70)

    if promoted:
        # 转正：候选目录文件覆盖正式回测目录（文件名固定，逐一覆盖），再清理候选目录
        tgt_dir.mkdir(parents=True, exist_ok=True)
        for f in cand_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, tgt_dir / f.name)
        shutil.rmtree(cand_dir, ignore_errors=True)
        # 候选回测结果作为新基线保存
        save_backtest_results(new_results, tag=BACKTEST_TAG)
        _write_guard_meta()
        logger.info(f"候选回测模型已转正并保存到 {tgt_dir}")
    else:
        shutil.rmtree(cand_dir, ignore_errors=True)

    return promoted


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("A股量化模型训练开始")
    logger.info(
        f"参数: update={args.update}, data_only={args.data_only}, "
        f"start={START_DATE}, horizons={HORIZONS_DEFAULT}")

    # GPU检测
    gpu_info = detect_gpu()
    logger.info(f"GPU: LightGBM={'GPU' if gpu_info['lightgbm_gpu'] else 'CPU'}, "
                f"PyTorch={gpu_info['torch_device']}({gpu_info['torch_gpu_name']}, "
                f"{gpu_info['torch_gpu_mem_gb']}GB)")

    # ========== Step 1: 数据更新 ==========
    logger.info("\n" + "=" * 70)
    if args.update or args.data_only:
        logger.info("Step 1: 增量更新数据到最新交易日")
        try:
            stats = update_data(start_date=START_DATE, force_full=False, is_intraday=False)
            logger.info(f"数据更新统计: {stats}")
            # 数据完整性校验：失败股票数>5则终止训练，防止脏数据污染模型
            FAILED_THRESHOLD = 5
            n_failed = stats.get("failed", 0)
            if n_failed > FAILED_THRESHOLD:
                logger.error("=" * 70)
                logger.error(
                    f"数据完整性校验失败：本次增量更新有 {n_failed} 只股票下载失败，超过阈值 {FAILED_THRESHOLD} 只。")
                logger.error(
                    "为保证数据唯一准确性，终止本次运行。请重新运行 python train.py --update 补齐失败股票后再训练。")
                logger.error("=" * 70)
                sys.exit(1)
            try:
                update_hs300()
            except Exception as e:
                logger.warning(f"沪深300更新失败: {e}")
            # 仅更新数据：不计算因子、不训练，直接结束
            if args.data_only:
                logger.info("=" * 70)
                logger.info("数据已更新到最新交易日（--data-only），跳过因子计算与模型训练。")
                logger.info("需要训练时运行: python train.py（数据已最新，因子缓存会自动检测过期并重算）")
                logger.info("=" * 70)
                return
        except Exception as e:
            if args.data_only:
                logger.error(f"数据更新失败({e})，本次未更新任何模型，退出。请重试 python train.py --data-only")
                sys.exit(1)
            logger.error(f"数据更新失败({e})，尝试使用已有数据继续训练")
    else:
        logger.info("Step 1: 使用本地存量数据（跳过更新，如需更新请加 --update）")

    # ========== Step 2: 加载数据 + 计算因子（缓存复用） ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 加载数据 + 计算因子")
    factor_path = PROCESSED_DIR / "factor_data.parquet"
    factor_cols_path = PROCESSED_DIR / "factor_cols.json"
    factor_meta_path = PROCESSED_DIR / "factor_meta.json"
    factor_data = None
    factor_cols = None

    # 检查因子缓存是否可用（口径标记 + 数据日期同时匹配才复用）
    # 口径标记 redundancy_cutoff 不同（如旧版全量去冗余 vs 新版仅用2021前样本）必须重算
    if factor_path.exists() and factor_cols_path.exists():
        try:
            cache_cutoff = None
            if factor_meta_path.exists():
                with open(factor_meta_path, encoding="utf-8") as f:
                    cache_cutoff = json.load(f).get("redundancy_cutoff")
            if cache_cutoff != REDUNDANCY_CUTOFF:
                logger.info(f"因子缓存口径不符（缓存cutoff={cache_cutoff}，当前={REDUNDANCY_CUTOFF}），重新计算")
            else:
                cached = pd.read_parquet(factor_path)
                cached_date = cached["date"].max()
                # 检查本地数据最新日期
                from core.data_loader import RAW_DIR
                meta_path = RAW_DIR / "_meta.json"
                data_latest = None
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta:
                        data_latest = max(meta.values())
                if data_latest and cached_date >= pd.Timestamp(data_latest):
                    logger.info(f"因子缓存可用（最新日期 {cached_date:%Y-%m-%d}，口径cutoff={cache_cutoff}），直接复用")
                    factor_data = cached
                    with open(factor_cols_path, "r", encoding="utf-8") as f:
                        factor_cols = json.load(f)
                    logger.info(f"从缓存加载factor_cols: {len(factor_cols)}个因子")
                else:
                    logger.info("因子缓存过期，重新计算")
        except Exception as e:
            logger.warning(f"因子缓存读取失败，重新计算: {e}")

    if factor_data is None:
        start_dt = f"{START_DATE[:4]}-{START_DATE[4:6]}-{START_DATE[6:]}"
        all_data = load_all_data(start_date=start_dt)
        factor_data, factor_cols = compute_factors(all_data, redundancy_cutoff=REDUNDANCY_CUTOFF)
        del all_data
        import gc;
        gc.collect()
        factor_data.to_parquet(factor_path, index=False)
        with open(factor_cols_path, "w", encoding="utf-8") as f:
            json.dump(factor_cols, f, ensure_ascii=False)
        with open(factor_meta_path, "w", encoding="utf-8") as f:
            json.dump({"redundancy_cutoff": REDUNDANCY_CUTOFF,
                       "backtest_start": BACKTEST_START,
                       "train_end": BT_TRAIN_END}, f, ensure_ascii=False, indent=2)
        logger.info(f"因子数据已缓存: {factor_path}, factor_cols: {factor_cols_path}, 口径: {factor_meta_path}")

    logger.info(f"因子: {len(factor_cols)}个, 样本: {len(factor_data)}")

    # ========== Step 3: 创建标签 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: 创建标签 (1天持仓，动态卖出：开盘止盈0.5% + 盘中止盈4.5% + 不止损)")
    labeled = create_labels(factor_data, horizons=HORIZONS_DEFAULT)

    # ========== Step 4: 严格无泄露两阶段训练回测模型，收益守卫通过后再训练预测模型 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 训练集成模型 (LightGBM + XGBoost + MLP)")
    logger.info("  4a. 回测模型（严格无泄露，训练 2016-2021，回测 2022-01-04 起）：")
    logger.info(f"      阶段1 早停：训练 <= {STAGE1_TRAIN_END}，验证 2021 全年（至 {STAGE1_VAL_END}），定迭代轮数")
    logger.info(f"      阶段2 refit：固定轮数，用 <= {BT_TRAIN_END} 全量样本重训（模型最晚见到 2021-12-31 价格）")

    # 阶段1：2016-2020 训练 + 2021 全年验证（验证集严格止于 2021-12-30，标签最晚 2021-12-31 实现）
    stage1_models = train_models_with_split(
        labeled, factor_cols, HORIZONS_DEFAULT,
        split_date=STAGE1_TRAIN_END, val_end=STAGE1_VAL_END,
        save_tag=BACKTEST_STAGE1_TAG, mode_name="回测模型-阶段1早停"
    )
    best_iters = extract_best_iters(stage1_models, HORIZONS_DEFAULT)
    logger.info(f"阶段1早停确定的迭代轮数（阶段2与预测模型固定继承）: {best_iters}")
    del stage1_models
    import gc as _gc;
    _gc.collect()

    # 阶段2：固定轮数，用 <=2021-12-30 全量样本 refit（2021-12-31 特征日标签需 2022-01-04 价格，天然被排除）
    bt_train = labeled[labeled["date"] <= pd.Timestamp(BT_TRAIN_END)].copy()
    logger.info(f"阶段2 refit 样本区间: {bt_train['date'].min():%Y-%m-%d} ~ {bt_train['date'].max():%Y-%m-%d}，"
                f"共 {len(bt_train)} 行")
    bt_models = train_models_full_data(
        bt_train, factor_cols, HORIZONS_DEFAULT,
        save_tag=BACKTEST_CANDIDATE_TAG, fixed_iters=best_iters,
        mode_name="回测模型-阶段2 refit"
    )
    del bt_train;
    _gc.collect()

    # 阶段1 临时目录已无用，立即删除；候选 checkpoint 补写回测起点（backtest.py 精确读取）
    shutil.rmtree(MODEL_DIR / BACKTEST_STAGE1_TAG, ignore_errors=True)
    cand_cp_path = MODEL_DIR / BACKTEST_CANDIDATE_TAG / "_checkpoint.json"
    if cand_cp_path.exists():
        try:
            with open(cand_cp_path, encoding="utf-8") as f:
                cand_cp = json.load(f)
            cand_cp["backtest_start"] = BACKTEST_FIRST_DAY
            cand_cp["redundancy_cutoff"] = REDUNDANCY_CUTOFF
            cand_cp["fixed_iters"] = best_iters
            with open(cand_cp_path, "w", encoding="utf-8") as f:
                json.dump(cand_cp, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"候选回测checkpoint补写回测起点失败: {e}")

    # 收益守卫：候选回测（2022-01-04起）总收益不劣于同口径基线才转正；否则丢弃、终止本次训练
    if not guard_backtest_model(factor_data, bt_models, HORIZONS_DEFAULT):
        logger.info("本次训练结束：新回测模型未优于基线，已保留原回测模型与原预测模型，不做任何覆盖。")
        return

    logger.info(
        "\n  4b. 预测模型 (100%全量数据 2016~最新 训练，迭代轮数继承阶段1早停，用于实盘选股；与回测模型是两套独立模型)")
    pred_models = train_models_full_data(
        labeled, factor_cols, HORIZONS_DEFAULT,
        save_tag=PREDICT_TAG, fixed_iters=best_iters,
        mode_name="预测模型-100%全量"
    )

    # ========== 完成 ==========
    logger.info("\n" + "=" * 70)
    bt_success = [h for h in HORIZONS_DEFAULT if h in bt_models]
    pred_success = [h for h in HORIZONS_DEFAULT if h in pred_models]
    logger.info(f"训练完成！回测模型（已通过收益守卫）: {bt_success}, 预测模型成功: {pred_success}")
    logger.info(f"回测模型路径: {MODEL_DIR / BACKTEST_TAG}")
    logger.info(f"预测模型路径: {MODEL_DIR / PREDICT_TAG}")
    if bt_success:
        logger.info("可运行 python backtest.py 进行回测（自动加载回测模型，动态卖出策略）")
    if pred_success:
        logger.info("可运行 python predict.py 进行预测（自动加载预测模型）")


if __name__ == "__main__":
    main()
