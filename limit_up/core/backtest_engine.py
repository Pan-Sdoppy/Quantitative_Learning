"""涨停板专项 - 回测引擎（评估预测准确度：预测值 vs 实际次日最高价涨幅）"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def evaluate_prediction(df, pred_col='pred_next_high_ret', actual_col='next_high_ret'):
    """评估预测准确度：预测值和实际次日最高价涨幅的接近程度

    核心指标：
    - MAE: 平均绝对误差
    - RMSE: 均方根误差
    - Corr: 相关性
    - 排序一致性: TopN 预测高的，实际是否也高
    - 分档准确度: 预测值分档后，实际值是否符合预期

    Args:
        df: 涨停板样本数据（含 pred_next_high_ret 和 next_high_ret）
        pred_col: 预测值列名
        actual_col: 实际值列名

    Returns:
        dict: 评估指标
    """
    df = df.dropna(subset=[pred_col, actual_col]).copy()

    pred = df[pred_col].values
    actual = df[actual_col].values

    # 基础误差指标
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    corr = np.corrcoef(pred, actual)[0, 1]
    bias = np.mean(pred - actual)  # 正=高估，负=低估

    # 排序一致性：TopN 预测高的，实际是否也高
    result = {
        'total_samples': len(df),
        'mae': mae,
        'rmse': rmse,
        'corr': corr,
        'bias': bias,
    }

    # 按预测值分档，统计每档实际值
    df['pred_bin'] = pd.qcut(df[pred_col], q=5, labels=['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)'], duplicates='drop')
    bin_stats = df.groupby('pred_bin', observed=True).agg(
        pred_mean=(pred_col, 'mean'),
        actual_mean=(actual_col, 'mean'),
        actual_median=(actual_col, 'median'),
        actual_min=(actual_col, 'min'),
        actual_max=(actual_col, 'max'),
        count=(actual_col, 'count'),
    ).reset_index()

    result['bin_stats'] = bin_stats

    return result


def evaluate_ranking(df, pred_col='pred_next_high_ret', actual_col='next_high_ret'):
    """评估排序质量：每天预测的排序，和实际值排序的一致性

    场景：同一天有多个涨停板，模型预测哪个次日溢价更高，实际是否一致
    """
    df = df.dropna(subset=[pred_col, actual_col]).copy()

    daily_corr = []
    for date, group in df.groupby('date'):
        if len(group) >= 3:  # 至少3只涨停板才有排序意义
            c = np.corrcoef(group[pred_col], group[actual_col])[0, 1]
            if not np.isnan(c):
                daily_corr.append(c)

    avg_daily_corr = np.mean(daily_corr) if daily_corr else 0

    # Top1 命中率：预测最高的，实际是否也最高
    top1_hit = 0
    total_days = 0
    for date, group in df.groupby('date'):
        if len(group) >= 3:
            pred_top1 = group.loc[group[pred_col].idxmax(), 'code']
            actual_top1 = group.loc[group[actual_col].idxmax(), 'code']
            if pred_top1 == actual_top1:
                top1_hit += 1
            total_days += 1

    top1_hit_rate = top1_hit / total_days * 100 if total_days > 0 else 0

    return {
        'avg_daily_corr': avg_daily_corr,
        'top1_hit_rate': top1_hit_rate,
        'top1_hit_days': top1_hit,
        'total_days': total_days,
    }
