"""涨停板专项 - 预测入口（预测当天涨停板股票的次日最高价涨幅）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
import joblib
from pathlib import Path

from core.data_loader import load_limit_up_data, get_feature_cols
from core.model import predict
from core.utils import MODEL_DIR, RESULT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("涨停板专项预测开始")
    logger.info("=" * 60)

    # 1. 加载模型
    model_path = MODEL_DIR / "limit_up_lgb.pkl"
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}，请先运行 train.py")
        return
    bundle = joblib.load(model_path)
    model = bundle['model']
    feature_cols = bundle['feature_cols']
    logger.info(f"模型已加载: {model_path}")

    # 2. 加载数据
    df = load_limit_up_data()

    # 3. 取最新一天的涨停板样本
    latest_date = df['date'].max()
    today_df = df[df['date'] == latest_date].copy()
    logger.info(f"最新交易日: {latest_date.date()}，涨停板 {len(today_df)} 只")

    if len(today_df) == 0:
        logger.info("今日无涨停板股票，无需预测")
        return

    # 4. 预测
    today_df['pred_next_high_ret'] = predict(model, today_df, feature_cols)

    # 5. 按预测值排序
    today_df = today_df.sort_values('pred_next_high_ret', ascending=False).reset_index(drop=True)

    # 6. 输出结果
    print("\n" + "=" * 80)
    print(f"涨停板次日最高价预测 - {latest_date.date()}")
    print("=" * 80)
    print(f"{'排名':<4} {'股票代码':<8} {'收盘价':<8} {'当日涨幅':<8} {'预测次日最高涨幅':<12}")
    print("-" * 80)
    for i, row in today_df.head(20).iterrows():
        print(f"{i+1:<4} {row['code']:<8} {row['close']:<8.2f} {row['pct_chg']:<8.2f}% {row['pred_next_high_ret']:<12.2f}%")

    # 7. 保存结果
    out_path = RESULT_DIR / "predictions.xlsx"
    today_df.to_excel(out_path, index=False)
    logger.info(f"预测结果已保存: {out_path}")


if __name__ == "__main__":
    main()
