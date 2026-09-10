"""
训练入口文件
============
功能：
1. 增量下载A股全市场数据（自动剔除ST/退市/次新/停牌）
2. 计算技术因子 + 横截面标准化（缓存复用）
3. 训练1天持仓动态卖出的 LightGBM + XGBoost + MLP 集成模型
      - 1d_close: T日尾盘买，T+1动态卖
4. 一次训练产出两套模型：回测模型（训练到2023-12-31）、预测模型（100%全量数据）
5. 模型覆盖保存，不产生垃圾文件

用法：
    python train.py           # 用本地存量数据训练
    python train.py --update  # 先增量下载截止当前最新交易日的数据，再训练
"""
import argparse
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from core.data_loader import update_data, load_all_data, update_hs300
from core.factors import compute_factors, create_labels
from core.model import EnsembleModel
from core.utils import PROCESSED_DIR, MODEL_DIR, get_logger, detect_gpu

warnings.filterwarnings("ignore")
logger = get_logger("train")

# 两种模型各自的保存路径
BACKTEST_TAG = "latest/backtest"  # 回测模型：训练到2023-12-31，回测从2024-01-01开始
PREDICT_TAG = "latest/predict"  # 预测模型：100%全量数据训练，迭代次数从回测模型继承

# 固定配置
START_DATE = "20160101"
HORIZONS_DEFAULT = ["1d_close"]


def parse_args():
    parser = argparse.ArgumentParser(description="A股量化模型训练")
    parser.add_argument("--update", action="store_true",
                        help="先增量下载截止当前最新交易日的数据，再训练")
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
                            mode_name="", retrain=False, split_date=None):
    """用指定分割比例或固定日期训练一套模型，保存到save_tag目录
    split_date: 固定分割日期（如"2023-12-31"），优先于split_ratio
    """
    if split_date:
        train_end = pd.Timestamp(split_date)
    else:
        all_dates = sorted(labeled["date"].unique())
        split_idx = int(len(all_dates) * split_ratio)
        train_end = all_dates[split_idx]
    train_df = labeled[labeled["date"] <= train_end]
    val_df = labeled[labeled["date"] > train_end]
    train_start = train_df["date"].min()
    val_end = val_df["date"].max() if len(val_df) > 0 else train_end
    logger.info(f"[{mode_name}] 训练数据: {train_start:%Y-%m-%d} ~ {train_end:%Y-%m-%d} ({len(train_df)}样本)")
    logger.info(
        f"[{mode_name}] 验证数据: {train_end + pd.Timedelta(days=1):%Y-%m-%d} ~ {val_end:%Y-%m-%d} ({len(val_df)}样本)")

    X_train = train_df[factor_cols].values
    X_val = val_df[factor_cols].values
    data_signature = f"{len(factor_cols)}f_{train_df['date'].max():%Y%m%d}_{len(train_df)}"
    train_end_date = train_end.strftime("%Y-%m-%d")

    # 每次运行都重新训练，不清空断点文件（保留训练过程中的宕机恢复能力）
    cp = load_checkpoint(save_tag)
    trained_horizons = []  # 每次运行都从头训练，不跳过已训练模型
    cp_data_sig = cp.get("data_signature", "")

    logger.info(f"开始训练，数据签名: {data_signature}")
    cp = {"data_signature": data_signature, "trained_horizons": [],
          "factor_count": len(factor_cols), "train_end_date": train_end_date,
          "started_at": datetime.now().isoformat()}

    models = {}
    for h in horizons:
        logger.info(f"\n--- 训练 1天持仓-动态卖出 ({h}) 模型 ---")
        try:
            y_col = f"label_ret_{h}"
            y_train = train_df[y_col].values
            y_val = val_df[y_col].values
            mask_tr = np.abs(y_train) < 0.5
            mask_vl = np.abs(y_val) < 0.5
            X_tr = X_train[mask_tr]
            y_tr = y_train[mask_tr]
            X_vl = X_val[mask_vl]
            y_vl = y_val[mask_vl]
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
                           fixed_iters, retrain=False):
    """用100%数据训练模型（无验证集），迭代次数从回测模型继承
    fixed_iters: {horizon: {"lgb": int, "xgb": int, "mlp": int}} 各模型固定迭代次数
    """
    train_df = labeled.copy()
    train_start = train_df["date"].min()
    train_end = train_df["date"].max()
    logger.info(f"[预测模型] 训练数据: {train_start:%Y-%m-%d} ~ {train_end:%Y-%m-%d} ({len(train_df)}样本, 100%全量)")

    X_train = train_df[factor_cols].values
    data_signature = f"{len(factor_cols)}f_{train_end:%Y%m%d}_{len(train_df)}_full"
    train_end_date = train_end.strftime("%Y-%m-%d")

    # 每次运行都重新训练
    cp = load_checkpoint(save_tag)
    trained_horizons = []
    logger.info(f"开始训练(预测模型-100%数据)，数据签名: {data_signature}")
    cp = {"data_signature": data_signature, "trained_horizons": [],
          "factor_count": len(factor_cols), "train_end_date": train_end_date,
          "started_at": datetime.now().isoformat()}

    models = {}
    for h in horizons:
        logger.info(f"\n--- 训练 1天持仓-动态卖出 ({h}) 模型 [预测模型-100%数据] ---")
        try:
            y_col = f"label_ret_{h}"
            y_train = train_df[y_col].values
            mask_tr = np.abs(y_train) < 0.5
            X_tr = X_train[mask_tr]
            y_tr = y_train[mask_tr]
            logger.info(f"有效训练样本: {len(X_tr)} (100%数据，无验证集)")

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


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("A股量化模型训练开始")
    logger.info(f"参数: update={args.update}, start={START_DATE}, horizons={HORIZONS_DEFAULT}")

    # GPU检测
    gpu_info = detect_gpu()
    logger.info(f"GPU: LightGBM={'GPU' if gpu_info['lightgbm_gpu'] else 'CPU'}, "
                f"PyTorch={gpu_info['torch_device']}({gpu_info['torch_gpu_name']}, "
                f"{gpu_info['torch_gpu_mem_gb']}GB)")

    # ========== Step 1: 数据更新 ==========
    logger.info("\n" + "=" * 70)
    if args.update:
        logger.info("Step 1: 增量更新数据到最新交易日")
        try:
            stats = update_data(start_date=START_DATE, force_full=False, is_intraday=False)
            logger.info(f"数据更新统计: {stats}")
            try:
                update_hs300()
            except Exception as e:
                logger.warning(f"沪深300更新失败: {e}")
        except Exception as e:
            logger.error(f"数据更新失败({e})，尝试使用已有数据继续训练")
    else:
        logger.info("Step 1: 使用本地存量数据（跳过更新，如需更新请加 --update）")

    # ========== Step 2: 加载数据 + 计算因子（缓存复用） ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 加载数据 + 计算因子")
    factor_path = PROCESSED_DIR / "factor_data.parquet"
    factor_cols_path = PROCESSED_DIR / "factor_cols.json"
    factor_data = None
    factor_cols = None

    # 检查因子缓存是否可用（数据日期匹配则复用）
    if factor_path.exists() and factor_cols_path.exists():
        try:
            cached = pd.read_parquet(factor_path)
            cached_date = cached["date"].max()
            # 检查本地数据最新日期
            from core.data_loader import RAW_DIR
            meta_path = RAW_DIR / "_meta.json"
            data_latest = None
            if meta_path.exists():
                import json as _json
                with open(meta_path) as f:
                    meta = _json.load(f)
                if meta:
                    data_latest = max(meta.values())
            if data_latest and cached_date >= pd.Timestamp(data_latest):
                logger.info(f"因子缓存可用（最新日期 {cached_date:%Y-%m-%d}），直接复用")
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
        factor_data, factor_cols = compute_factors(all_data)
        del all_data
        import gc;
        gc.collect()
        factor_data.to_parquet(factor_path, index=False)
        with open(factor_cols_path, "w", encoding="utf-8") as f:
            json.dump(factor_cols, f, ensure_ascii=False)
        logger.info(f"因子数据已缓存: {factor_path}, factor_cols: {factor_cols_path}")

    logger.info(f"因子: {len(factor_cols)}个, 样本: {len(factor_data)}")

    # ========== Step 3: 创建标签 ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: 创建标签 (1天持仓，动态卖出（高开2%止盈+跌2%止损）)")
    labeled = create_labels(factor_data, horizons=HORIZONS_DEFAULT)

    # ========== Step 4: 训练两套模型（回测用 + 预测用） ==========
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 训练集成模型 (LightGBM + XGBoost + MLP)")
    logger.info("  4a. 回测模型 (训练到2023-12-31，回测从2024-01-01开始)")
    bt_models = train_models_with_split(
        labeled, factor_cols, HORIZONS_DEFAULT,
        split_date="2023-12-31", save_tag=BACKTEST_TAG,
        mode_name="回测模型", retrain=False
    )

    logger.info("\n  4b. 预测模型 (100%全量数据训练，迭代次数从回测模型继承，用于实盘选股)")
    best_iters = extract_best_iters(bt_models, HORIZONS_DEFAULT)
    logger.info(f"从回测模型继承的最佳迭代次数: {best_iters}")
    pred_models = train_models_full_data(
        labeled, factor_cols, HORIZONS_DEFAULT,
        save_tag=PREDICT_TAG, fixed_iters=best_iters, retrain=False
    )

    # ========== 完成 ==========
    logger.info("\n" + "=" * 70)
    bt_success = [h for h in HORIZONS_DEFAULT if h in bt_models]
    pred_success = [h for h in HORIZONS_DEFAULT if h in pred_models]
    logger.info(f"训练完成！回测模型成功: {bt_success}, 预测模型成功: {pred_success}")
    logger.info(f"回测模型路径: {MODEL_DIR / BACKTEST_TAG}")
    logger.info(f"预测模型路径: {MODEL_DIR / PREDICT_TAG}")
    if bt_success:
        logger.info(f"可运行 python backtest.py 进行回测（自动加载回测模型，动态卖出策略）")
    if pred_success:
        logger.info(f"可运行 python predict.py 进行预测（自动加载预测模型）")


if __name__ == "__main__":
    main()
