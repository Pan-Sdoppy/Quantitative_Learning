"""临时脚本：按 train.py 生产口径刷新因子缓存（下载新数据后使用，用完即删）"""
import json
import warnings

import pandas as pd

import train  # 复用全部生产常量
from core.data_loader import load_all_data
from core.factors import compute_factors
from core.utils import PROCESSED_DIR

warnings.filterwarnings("ignore")

factor_path = PROCESSED_DIR / "factor_data.parquet"
factor_cols_path = PROCESSED_DIR / "factor_cols.json"
factor_meta_path = PROCESSED_DIR / "factor_meta.json"

start_dt = f"{train.START_DATE[:4]}-{train.START_DATE[4:6]}-{train.START_DATE[6:]}"
print(f"recompute factors, start={start_dt}, redundancy_cutoff={train.REDUNDANCY_CUTOFF}")
factor_data, factor_cols = compute_factors(load_all_data(start_date=start_dt),
                                           redundancy_cutoff=train.REDUNDANCY_CUTOFF)
factor_data.to_parquet(factor_path, index=False)
with open(factor_cols_path, "w", encoding="utf-8") as f:
    json.dump(factor_cols, f, ensure_ascii=False)
with open(factor_meta_path, "w", encoding="utf-8") as f:
    json.dump({"redundancy_cutoff": train.REDUNDANCY_CUTOFF,
               "backtest_start": train.BACKTEST_START,
               "train_end": train.BT_TRAIN_END}, f, ensure_ascii=False, indent=2)
print(f"factor_data rows={len(factor_data)}, stocks={factor_data['code'].nunique()}, "
      f"cols={len(factor_cols)}, date range={factor_data['date'].min()} ~ {factor_data['date'].max()}")
print("factor cache refreshed OK")
