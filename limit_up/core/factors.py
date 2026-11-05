"""涨停板专项因子计算"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_limit_up_features(df):
    """添加涨停板专属因子

    Args:
        df: 涨停板样本数据（含 date, code, pct_chg, close, high, low, volume 等）

    Returns:
        DataFrame: 添加了专属因子的数据
    """
    df = df.sort_values(['code', 'date']).copy()

    # 1. 连板数（连续涨停天数）
    df['is_limit_up'] = (df['pct_chg'] >= 9.5).astype(int)
    df['consecutive_limit_up'] = df.groupby('code')['is_limit_up'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    # 2. 当日振幅（(high - low) / close * 100）
    df['intraday_amplitude'] = (df['high'] - df['low']) / df['close'] * 100

    # 3. 当日量比（已有 vol_ratio_5，跳过）
    # 4. 距 60 日高点距离
    df['high_60d'] = df.groupby('code')['high'].rolling(60).max().reset_index(0, drop=True)
    df['dist_from_60d_high'] = (df['close'] - df['high_60d']) / df['high_60d'] * 100

    # 5. 前 N 日涨停次数（近 5 日/10 日/20 日）
    for window in [5, 10, 20]:
        df[f'limit_up_count_{window}d'] = df.groupby('code')['is_limit_up'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )

    # 6. 涨停后首日 vs 连续涨停
    df['is_first_limit_up'] = (df['consecutive_limit_up'] == 1).astype(int)
    df['is_multi_limit_up'] = (df['consecutive_limit_up'] >= 2).astype(int)

    # 7. 当日换手率（如果有的话）
    if 'turnover_ma5' in df.columns:
        df['turnover_ratio'] = df.get('turnover_ma5', np.nan)

    # 8. 市值因子
    if 'mv_log_yi' in df.columns:
        df['mv_factor'] = df['mv_log_yi']

    # 9. 涨停日距离上次涨停的天数
    df['days_since_last_limit_up'] = df.groupby('code')['is_limit_up'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
    )

    # 10. 当日资金流入强度
    if 'mf_intraday' in df.columns:
        df['mf_strength'] = df['mf_intraday']

    logger.info(f"添加了 {10} 个涨停板专属因子")
    return df


def get_limit_up_feature_cols(df):
    """获取涨停板专属因子列名"""
    limit_up_features = [
        'consecutive_limit_up',
        'intraday_amplitude',
        'dist_from_60d_high',
        'limit_up_count_5d',
        'limit_up_count_10d',
        'limit_up_count_20d',
        'is_first_limit_up',
        'is_multi_limit_up',
        'turnover_ratio',
        'days_since_last_limit_up',
        'mf_strength',
    ]
    return [c for c in limit_up_features if c in df.columns]
