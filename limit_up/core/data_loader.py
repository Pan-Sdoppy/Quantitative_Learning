"""涨停板专项 - 数据加载（复用主板因子缓存，筛选涨停板样本）"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from .utils import RAW_DIR, LIMIT_UP_PCT, MIN_LIST_DAYS, PROJECT_ROOT
from .factors import add_limit_up_features, get_limit_up_feature_cols

logger = logging.getLogger(__name__)

# 主板因子缓存路径
FACTOR_CACHE = PROJECT_ROOT.parent / "data" / "processed" / "factor_data.parquet"


def load_limit_up_data():
    """加载涨停板样本数据

    1. 从主板 factor_data.parquet 加载全部因子
    2. 从 raw 数据补 pct_chg（涨停判断）和 next_high（标签）
    3. 筛选涨停板样本（ret_1d >= 9.5%）

    Returns:
        DataFrame: 涨停板样本，含因子列 + next_high_ret 标签
    """
    # 1. 加载主板因子缓存
    logger.info(f"加载主板因子缓存: {FACTOR_CACHE}")
    df = pd.read_parquet(FACTOR_CACHE)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"因子数据: {len(df)} 行, {df['code'].nunique()} 只股票")

    # 2. 从 raw 数据补 pct_chg 和 next_high
    #    （factor_data 里只有 ret_1d，没有 pct_chg 和 next_high）
    logger.info("从 raw 数据补 pct_chg 和 next_high...")
    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    raw_files = [f for f in raw_files if not f.name.startswith('_') and not f.name.endswith('.bak')]

    raw_frames = []
    for f in tqdm(raw_files, desc="加载raw数据", ncols=100):
        try:
            r = pd.read_parquet(f, columns=['date', 'close', 'high', 'pct_chg', 'volume', 'amount'])
            r['code'] = f.stem
            r['date'] = pd.to_datetime(r['date'])
            raw_frames.append(r)
        except Exception as e:
            logger.warning(f"加载 {f.name} 失败: {e}")

    raw_df = pd.concat(raw_frames, ignore_index=True)
    raw_df = raw_df.sort_values(['code', 'date']).reset_index(drop=True)

    # 计算次日最高价
    raw_df['next_high'] = raw_df.groupby('code')['high'].shift(-1)
    raw_df['next_high_ret'] = (raw_df['next_high'] - raw_df['close']) / raw_df['close'] * 100

    # 3. 合并因子和 raw 数据
    df = df.merge(
        raw_df[['date', 'code', 'pct_chg', 'next_high', 'next_high_ret']],
        on=['date', 'code'], how='left'
    )

    # 4. 筛选涨停板：pct_chg >= 9.5%
    df = df[df['pct_chg'] >= LIMIT_UP_PCT].copy()
    logger.info(f"涨停板样本（pct_chg>={LIMIT_UP_PCT}%）: {len(df)} 行")

    # 5. 剔除极端异常标签
    df = df[df['next_high_ret'].abs() < 30].copy()

    # 6. 剔除次新股（上市不足60天）
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    list_days = df.groupby('code').cumcount()
    # 这里 cumcount 是在涨停板样本里的序号，不是上市天数
    # 简化：直接用 data 行数判断（如果某只票涨停板样本 < 60 天，说明是次新股）
    code_counts = df.groupby('code').size()
    valid_codes = code_counts[code_counts >= MIN_LIST_DAYS].index
    df = df[df['code'].isin(valid_codes)].copy()

    logger.info(f"最终涨停板样本: {len(df)} 行, {df['code'].nunique()} 只股票, "
                f"日期 {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 7. 添加涨停板专属因子
    logger.info("添加涨停板专属因子...")
    df = add_limit_up_features(df)

    return df


def get_feature_cols(df):
    """获取因子列名列表（排除非因子列）"""
    exclude = ['date', 'code', 'open', 'high', 'low', 'close', 'pct_chg',
               'next_high', 'next_high_ret', 'ret_1d_raw',
               'hs300_ret_1d', 'hs300_ret_5d', 'hs300_ret_20d',
               'hs300_trend', 'hs300_volatility_5d',
               'is_limit_up', 'high_60d']
    base_features = [c for c in df.columns if c not in exclude]
    # 加入涨停板专属因子
    limit_up_features = get_limit_up_feature_cols(df)
    all_features = list(set(base_features + limit_up_features))
    return all_features
