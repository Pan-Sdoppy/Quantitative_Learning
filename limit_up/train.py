"""涨停板专项 - 训练入口"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from core.data_loader import load_limit_up_data, get_feature_cols
from core.model import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("涨停板专项模型训练开始")
    logger.info("=" * 60)

    # 1. 加载涨停板样本数据
    df = load_limit_up_data()
    feature_cols = get_feature_cols(df)
    logger.info(f"因子数: {len(feature_cols)}")

    # 2. 训练模型
    model = train_model(df, feature_cols)

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
