"""涨停板专项 - 回测入口（二分类：预测次日最高价涨幅是否为正）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from core.data_loader import load_limit_up_data, get_feature_cols
from core.model import predict
from core.utils import MODEL_DIR, RESULT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("涨停板专项回测（二分类：预测次日最高价涨幅是否为正）")
    logger.info("=" * 60)

    # 1. 加载模型
    model_path = MODEL_DIR / "limit_up_ensemble.pkl"
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}，请先运行 train.py")
        return
    ensemble = joblib.load(model_path)
    feature_cols = ensemble['feature_cols']
    logger.info(f"集成模型已加载: {model_path}")

    # 2. 加载数据
    df = load_limit_up_data()

    # 3. 预测（概率）
    df['pred_prob'] = predict(ensemble, df, feature_cols)

    # 4. 真实标签
    df['label'] = (df['next_high_ret'] > 0).astype(int)

    # 5. 样本外评估（2022-01-04 起）
    test_df = df[df['date'] >= '2022-01-04'].copy()
    logger.info(f"评估区间: {test_df['date'].min().date()} ~ {test_df['date'].max().date()}, "
                f"共 {len(test_df)} 个涨停板样本")

    # 6. 基础分类指标
    y_true = test_df['label'].values
    y_prob = test_df['pred_prob'].values
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n" + "=" * 80)
    print("二分类预测准确度评估（样本外 2022-01-04 起）")
    print("=" * 80)
    print(f"样本数: {len(test_df)}")
    print(f"正样本比例（次日有溢价）: {y_true.mean():.2%}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # 7. 按预测概率分档
    print("\n" + "=" * 80)
    print("按预测概率分档（Q1最低 ~ Q5最高）")
    print("=" * 80)
    test_df['prob_bin'] = pd.qcut(test_df['pred_prob'], q=5, labels=['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)'], duplicates='drop')
    bin_stats = test_df.groupby('prob_bin', observed=True).agg(
        prob_mean=('pred_prob', 'mean'),
        actual_pos_rate=('label', 'mean'),
        actual_next_high_ret_mean=('next_high_ret', 'mean'),
        count=('label', 'count'),
    ).reset_index()

    print(f"{'档位':<12} {'预测概率均值':<12} {'实际正样本率':<12} {'实际次日最高涨幅均值%':<18} {'样本数':<8}")
    print("-" * 80)
    for _, row in bin_stats.iterrows():
        print(f"{row['prob_bin']:<12} {row['prob_mean']:<12.2%} {row['actual_pos_rate']:<12.2%} "
              f"{row['actual_next_high_ret_mean']:<18.2f} {row['count']:<8}")

    # 8. 分年度评估
    print("\n" + "=" * 80)
    print("分年度预测准确度")
    print("=" * 80)
    test_df['year'] = test_df['date'].dt.year
    print(f"{'年份':<8} {'样本数':<8} {'Accuracy':<10} {'AUC':<8} {'正样本率':<10}")
    print("-" * 80)
    for year, group in test_df.groupby('year'):
        y_t = group['label'].values
        y_p = group['pred_prob'].values
        y_pd = (y_p > 0.5).astype(int)
        acc = accuracy_score(y_t, y_pd)
        a = roc_auc_score(y_t, y_p)
        pos_rate = y_t.mean()
        print(f"{year:<8} {len(group):<8} {acc:<10.4f} {a:<8.4f} {pos_rate:<10.2%}")

    logger.info("回测完成！")


if __name__ == "__main__":
    main()
