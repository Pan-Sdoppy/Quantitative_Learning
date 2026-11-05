# 涨停板专项研究（Limit-Up Board）

## 策略逻辑

- **训练样本**：只训练主板涨停板股票（当日涨幅 >= 9.5%）
- **预测目标**：次日最高价相对当日收盘价的涨幅（%）
- **预测范围**：只预测当天涨停板股票
- **输出**：按预测次日最高价涨幅排序

## 目录结构

```
limit_up/
├── core/
│   ├── __init__.py
│   ├── utils.py          # 路径配置（复用主板 data/raw/）
│   ├── data_loader.py   # 数据加载（复用主板 factor_data.parquet）
│   ├── model.py         # LightGBM 模型训练
│   ├── backtest_engine.py  # 回测引擎（待实现）
│   └── factors.py       # 因子计算（复用主板因子）
├── models/              # 模型文件
├── results/             # 预测结果
├── logs/                # 日志
├── train.py             # 训练入口
├── predict.py           # 预测入口
└── backtest.py           # 回测入口（待实现）
```

## 数据来源

- 复用主板 `data/raw/` 日线数据（只读）
- 复用主板 `data/processed/factor_data.parquet` 因子缓存（只读）

## 使用方法

```bash
# 训练
python limit_up/train.py

# 预测当天涨停板
python limit_up/predict.py
```
