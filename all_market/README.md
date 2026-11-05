# 全市场量化模型（主板 + 创业板 + 科创板，按个股自适应 ±10%/±20%）

主板隔日 T+1 模型的**全市场子项目**，代码逻辑、因子、标签、二次排序、动态卖出、择时与选股参数
与根目录主板模型**完全一致**，差异是股票池覆盖全部 A 股（沪深主板 + 双创），且涨停/追高/跳变
阈值按**每只个股所属板块逐行自适应**（主板 ±10%、双创 ±20%）。

> **数据铁律：全市场不复制、不下载任何个股日线，只读合并两个物理源目录。**
> 个股日线物理上只存一份，由两个源头项目各自维护，全市场直接读：
> - 主板 → 仓库根目录 `data/raw`（在**仓库根目录**更新）
> - 双创 → `chinext_star/data/raw`（在 **chinext_star** 更新）
>
> 本目录 `data/raw` **不存放个股 parquet**，只放全市场特有的小缓存（`_stock_list.csv`、`_stock_industry.csv`）；
> 全市场因子的横截面 rank 必须在"主板+双创"全集上重算，因此**因子缓存独立**放在本目录 `data/processed`，
> 模型 / 结果 / 日志同样各自独立。沪深300 指数直接读根目录共享文件，不重复下载。

## 一、股票池与涨跌幅

- **股票池**：沪深主板（`60` / `00`）+ 创业板（`300` / `301`）+ 科创板（`688` / `689`），剔除 ST / 退市 / 次新（上市不足 60 个交易日）。
- **不纳入**：北交所（8/4/920）。
- **涨跌幅（逐股）**：
  - 主板：恒 ±10%。
  - 科创板：自 2019-07-22 开市起即 ±20%。
  - 创业板：2020-08-24 注册制改革前为 ±10%，之后为 ±20%。
- **阈值（见 `core/board_profile.py`，逐股 `code + date` 计算）**：
  - 涨停判定 `_limit_thr`：主板 9.5% / 双创 19.5%（创业板按注册制日期分界）。
  - 追高 / 封单剔除：`_chase_thr = _limit_thr - 0.5%`（主板 9.0% / 双创 19.0%）。
  - 盘中实时价 vs qfq 昨收异常跳变：主板 |涨幅| > 15%、双创 > 21% 判为复权串行污染并剔除。
  - 盘中实时快照（新浪 `stock_zh_a_spot`）数量下限 / 提前停抓：`SPOT_MIN_BOARD=5000` / `SPOT_TARGET_BOARD=5250`（新浪实测单轮快照约 5222 只，含ST）。
  - 个股日线源目录：`MAIN_RAW_DIR`（根目录 `data/raw`）、`CHINEXT_RAW_DIR`（`chinext_star/data/raw`）、`STOCK_RAW_DIRS`（只读合并）。

## 二、与主板模型的差异

| 维度 | 主板（根目录） | 全市场（本目录） |
|---|---|---|
| 股票池正则 | `^(60\|00)` | `^(60\|00\|300\|301\|688\|689)` |
| 个股日线来源 | 根目录 `data/raw`（自维护、自下载） | **只读合并** 根目录 `data/raw` + `chinext_star/data/raw`（不复制、不下载） |
| 涨停线 | 统一 9.5% | 主板 9.5% / 双创 19.5%（逐股） |
| 追高剔除 | 统一 9.0% | 主板 9.0% / 双创 19.0%（逐股） |
| 盘中异常跳变 | 统一 15% | 主板 15% / 双创 21%（逐股） |
| 实时快照数量阈值 | 2800 / 3050 | 5000 / 5250 |
| 股票数量（快照含ST） | 约 3195 只（剔ST后约3050） | 约 5222 只（主板+双创，剔ST后约5000） |

> 因子列表、标签、Top3 仓位、composite 二次排序、`MIN_PRED=1.0%`、动态止盈、盘中 4% 止盈、
> 不止损、交易成本 0.35%、沪深300 择时（-5% 空仓）等**全部沿用主板口径**。
> 全市场把主板（±10%）与双创（±20%）混在一个选股池里排序，涨停/追高阈值逐股自适应；
> 独立回测后如发现板块间收益结构差异，可再考虑分板块配额或调参，当前先保持同口径。

## 三、目录结构与数据共享

```
Quantitative_Learning/
├── data/raw/                       # 【主板数据源】根目录项目自维护，3053 只 60/00 parquet
├── chinext_star/
│   └── data/raw/                   # 【双创数据源】chinext_star 项目自维护，300/301/688/689
├── all_market/                     # ← 本项目（只读上面两个目录，不存个股）
│   ├── train.py / backtest.py / predict.py
│   ├── requirements.txt
│   ├── core/
│   │   ├── board_profile.py        # 板块配置：股票池正则、涨停/追高/跳变阈值、源目录常量
│   │   ├── data_loader.py          # _stock_files() 只读合并两个源；update_data 只校验不下载
│   │   ├── factors.py / model.py / utils.py
│   │   └── backtest_engine.py      # 预处理 add_chase_threshold 生成逐股 _limit_thr/_chase_thr
│   ├── data/
│   │   ├── raw/                    # 只放全市场小缓存（_stock_list.csv / _stock_industry.csv），无个股 parquet
│   │   └── processed/              # 全市场因子缓存（横截面 rank 必须在全集重算，独立维护）
│   ├── models/  results/  logs/    # 全市场独立模型 / 回测结果 / 日志
```

数据分工一句话：**主板在根目录更新、双创在 chinext_star 更新、全市场只读合并这两个目录**，
同一个个股 parquet 在磁盘上只有一份，全市场绝不复制、绝不重复下载。

## 四、首次运行（baostock 恢复后执行）

> 前置：baostock 晚高峰（约 19:00–23:00）易拥塞，**白天或错峰执行**；日请求 ≤ 5 万次、**禁止并发**，
> 不要同时跑根目录与 chinext_star 的下载。

### 1) 先在两个源头把数据更新好（全市场自己不下载）

```powershell
# 主板：在仓库根目录（已是最新可跳过）
cd C:\Users\Pan\Documents\GitHub\Quantitative_Learning
$env:PYTHONPATH=(Get-Location).Path
& C:\Users\Pan\miniconda3\python.exe train.py --data-only

# 双创：在 chinext_star 目录（约 2000 只，全量下载，耗时较长）
cd C:\Users\Pan\Documents\GitHub\Quantitative_Learning\chinext_star
$env:PYTHONPATH=(Get-Location).Path
& C:\Users\Pan\miniconda3\python.exe train.py --data-only
```

### 2) 全市场：校验共享源 → 训练 → 回测 → 预测

所有命令都在 `all_market/` 目录内执行，并设置 `PYTHONPATH`：

```powershell
cd C:\Users\Pan\Documents\GitHub\Quantitative_Learning\all_market
$env:PYTHONPATH=(Get-Location).Path
$py = "C:\Users\Pan\miniconda3\python.exe"

# 1) 校验两个源目录（只统计主板/双创只数，不下载、不复制；任一源为空会直接报错退出）
& $py train.py --data-only

# 2) 训练模型（在“主板+双创”全集上重算横截面因子，缓存到本目录 data/processed）
& $py train.py

# 3) 回测（2022-01-04 起；同时输出开仓日与空仓日、含空仓月的月度汇总）
& $py backtest.py

# 4) 当日预测（盘中实时最新快照；接口失败/只数不足直接 sys.exit，不降级旧快照）
& $py predict.py
```

> `train.py --data-only` 在全市场只做**数据源校验 + 股票列表缓存刷新 + 沪深300 共享读取**，
> 不会登录 baostock、不会下载个股；若提示"双创数据源不存在或为空"，说明 chinext_star 尚未下载完成。

## 五、注意事项

- **训练/回测/预测因子列表必须完全一致**，从模型 `meta/ensemble.json` 读 `feature_names`，禁止实时 `compute_factors` 去冗余。
- 回测产物必须同时保留开仓日交易明细与空仓日记录（空仓标注原因：大盘择时 / min_pred 未达 / 无 T+1，写入 `empty_days` csv）；控制台 INFO 仅按原因分组计数，逐条日期+原因走 DEBUG（默认不打印，临时需要逐日分析时开启级别查看）；月度汇总含 0 交易空仓月。
- `SPOT_MIN_BOARD / SPOT_TARGET_BOARD` 已按新浪实测单轮快照（全市场约 5222 只）设为 5000 / 5250；若后续上市数量明显变化需重新校准，避免漏票。
- 任何时候全市场 `data/raw` 出现个股 `*.parquet` 都是错误的（个股只应在根目录或 chinext_star 两处）。
