"""涨停板专项 - 模型训练（二分类：预测次日最高价涨幅是否为正）+ 集成 + 样本权重"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from .utils import MODEL_DIR

logger = logging.getLogger(__name__)

# 训练参数
TRAIN_END = "2021-12-31"  # 训练截止日期（严格无泄露）
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 500,
}
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbosity': 0,
    'n_estimators': 500,
}
# 集成权重
ENSEMBLE_WEIGHTS = {'lgb': 0.5, 'xgb': 0.3, 'mlp': 0.2}


def compute_sample_weights(df):
    """计算样本权重：连板数越高，权重越高

    Args:
        df: 涨停板样本数据（含 consecutive_limit_up 列）

    Returns:
        Series: 样本权重
    """
    if 'consecutive_limit_up' not in df.columns:
        return pd.Series(1.0, index=df.index)

    weights = pd.Series(1.0, index=df.index)
    # 2连板: 1.2x
    weights[df['consecutive_limit_up'] == 2] = 1.2
    # 3连板: 1.5x
    weights[df['consecutive_limit_up'] == 3] = 1.5
    # 4连板及以上: 2.0x
    weights[df['consecutive_limit_up'] >= 4] = 2.0

    logger.info(f"样本权重: 均值={weights.mean():.2f}, 最大={weights.max():.2f}")
    return weights


def train_model(df, feature_cols):
    """训练 LightGBM + XGBoost 集成二分类模型

    Args:
        df: 涨停板样本数据（含因子列和 next_high_ret 标签）
        feature_cols: 因子列名列表

    Returns:
        dict: 集成模型
    """
    # 严格时间切分：训练 <= 2021-12-31，验证用 2022 年
    train_df = df[df['date'] <= TRAIN_END].copy()
    valid_df = df[(df['date'] > TRAIN_END) & (df['date'] <= '2022-12-31')].copy()

    logger.info(f"训练集: {len(train_df)} 行 ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    logger.info(f"验证集: {len(valid_df)} 行 ({valid_df['date'].min().date()} ~ {valid_df['date'].max().date()})")

    # 二分类标签：next_high_ret > 0 为 1（次日有溢价），否则为 0
    y_train = (train_df['next_high_ret'] > 0).astype(int).values
    y_valid = (valid_df['next_high_ret'] > 0).astype(int).values

    logger.info(f"训练集正样本比例: {y_train.mean():.4f}")
    logger.info(f"验证集正样本比例: {y_valid.mean():.4f}")

    X_train = train_df[feature_cols].values
    X_valid = valid_df[feature_cols].values

    # 样本权重
    sample_weights = compute_sample_weights(train_df).values
    logger.info(f"样本权重: 训练集均值={sample_weights.mean():.2f}")

    # 1. LightGBM
    logger.info("训练 LightGBM...")
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    lgb_pred = lgb_model.predict_proba(X_valid)[:, 1]
    lgb_auc = roc_auc_score(y_valid, lgb_pred)
    logger.info(f"LightGBM 验证集 AUC: {lgb_auc:.4f}")

    # 2. XGBoost
    logger.info("训练 XGBoost...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    xgb_pred = xgb_model.predict_proba(X_valid)[:, 1]
    xgb_auc = roc_auc_score(y_valid, xgb_pred)
    logger.info(f"XGBoost 验证集 AUC: {xgb_auc:.4f}")

    # 3. 集成预测
    ensemble_pred = (ENSEMBLE_WEIGHTS['lgb'] * lgb_pred +
                     ENSEMBLE_WEIGHTS['xgb'] * xgb_pred)

    # 验证集成效果
    ensemble_auc = roc_auc_score(y_valid, ensemble_pred)
    y_pred_binary = (ensemble_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_valid, y_pred_binary)
    precision = precision_score(y_valid, y_pred_binary)
    recall = recall_score(y_valid, y_pred_binary)
    f1 = f1_score(y_valid, y_pred_binary)

    logger.info(f"集成验证集: Accuracy={accuracy:.4f}, AUC={ensemble_auc:.4f}, "
                f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # 保存
    ensemble = {
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'feature_cols': feature_cols,
        'weights': ENSEMBLE_WEIGHTS,
    }
    model_path = MODEL_DIR / "limit_up_ensemble.pkl"
    joblib.dump(ensemble, model_path)
    logger.info(f"集成模型已保存: {model_path}")

    return ensemble


def predict(ensemble, df, feature_cols):
    """用集成模型预测次日最高价涨幅是否为正

    Args:
        ensemble: 集成模型 dict
        df: 预测数据（含因子列）
        feature_cols: 因子列名列表

    Returns:
        Series: 预测为正的概率
    """
    X = df[feature_cols].values
    lgb_pred = ensemble['lgb_model'].predict_proba(X)[:, 1]
    xgb_pred = ensemble['xgb_model'].predict_proba(X)[:, 1]
    pred = (ensemble['weights']['lgb'] * lgb_pred +
            ensemble['weights']['xgb'] * xgb_pred)
    return pred
