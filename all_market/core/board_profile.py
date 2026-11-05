"""
板块配置（决定股票池与涨跌幅口径）
==================================
本文件为「全市场（主板+创业板+科创板）」子项目的板块配置。
对应目录：all_market/

股票池：
  - 沪深主板 600/601/603/605/000/001/002/003（前缀 60/00）
  - 创业板 300 / 301
  - 科创板 688 / 689
  - 不含北交所（8/4/920）。

涨跌幅规则（前复权口径，用于"涨停封单买不进"和"追高过滤"），按个股 + 日期逐行判定：
  - 主板（60/00）：恒为 ±10%，涨停阈值 9.5%。
  - 科创板（688/689）：2019-07-22 开市起即 ±20%，涨停阈值 19.5%。
  - 创业板（300/301）：2020-08-24 注册制改革前 ±10%，改革后 ±20%。
  - 次新股上市前 5 个交易日不设涨跌幅：数据层已要求上市满 60 个交易日
    （MIN_LIST_DAYS=60）才纳入，训练/回测/预测样本中不会出现前 5 日无涨跌幅次新股，无需特判。

其余因子、标签、二次排序、动态卖出、市场择时（沪深300）等逻辑与主板项目完全一致，
仅"涨停/追高"这两处板块强相关阈值随板块自适应；止盈/止损/min_pred 等选股参数
先沿用主板口径，后续在全市场样本上独立回测再调优。
"""
from pathlib import Path

import pandas as pd

BOARD_KEY = "all_market"
BOARD_NAME = "主板+创业板+科创板（全市场）"

# 股票池代码前缀正则（6位代码）：主板(60/00) + 创业板(300/301) + 科创板(688/689)
UNIVERSE_REGEX = r"^(60|00|300|301|688|689)"

# 关键日期
CHINEXT_REG_DATE = pd.Timestamp("2020-08-24")  # 创业板存量股涨跌幅由10%放宽至20%

# 涨停判定阈值（收益率 >= 该值视为涨停）
MAIN_LIMIT_THR = 0.095    # 10%板（考虑四舍五入用9.5%）
STAR_LIMIT_THR = 0.195    # 20%板（考虑四舍五入用19.5%）

# 追高过滤阈值 = 涨停线 - 0.5%（主板9.0%、双创19.0%）
CHASE_MARGIN = 0.005

# 盘中实时快照（新浪 stock_zh_a_spot）数量健壮性阈值：
# 多轮并集后总数低于 MIN 判定接口分页截断/异常并拒绝出预测；达到 TARGET 即提前停抓。
# 全市场（主板+双创）：新浪 stock_zh_a_spot 实测单轮快照约 5222 只
# （主板约3195 + 双创约2027，含ST，2026-10）；ST 在后续股票池过滤再剔。
# 多轮并集目标 5250（略高于单轮全量，确保抓全不漏票）；并集后低于 5000 判定分页截断/异常并拒绝。
SPOT_MIN_BOARD = 5000
SPOT_TARGET_BOARD = 5250

# 盘中实时价 vs qfq昨收 异常跳变阈值：超出涨停线过大视为复权串行污染，剔除该股当日行。
# 主板涨停10%用15%（+5%缓冲），双创涨停20%用21%（+1%缓冲）。
JUMP_BAD_MAIN = 0.15
JUMP_BAD_STAR = 0.21

# ========== 个股日线数据源（全市场只读共享，不复制、不下载）==========
# 个股日线物理上只存一份，由两个源头项目各自维护，全市场直接读这两个目录合并：
#   主板 -> 仓库根目录 data/raw（在仓库根目录运行 train.py --data-only 更新）
#   双创 -> chinext_star/data/raw（在 chinext_star 运行 train.py --data-only 更新）
# 本目录的 data/raw 只放全市场特有的缓存（_stock_list.csv / _stock_industry.csv 等），
# 绝不存放任何个股 parquet。
_REPO_ROOT = Path(__file__).resolve().parents[2]                 # Quantitative_Learning/
MAIN_RAW_DIR = _REPO_ROOT / "data" / "raw"                        # 主板数据源（仓库根目录）
CHINEXT_RAW_DIR = _REPO_ROOT / "chinext_star" / "data" / "raw"    # 双创数据源
STOCK_RAW_DIRS = [CHINEXT_RAW_DIR, MAIN_RAW_DIR]                  # 合并顺序（双创在前；两源前缀不重叠）


def limit_up_threshold(code, date=None) -> float:
    """单只股票在某日的涨停判定阈值。
    科创板恒为20%；创业板在2020-08-24前为10%、之后为20%；主板恒为10%。"""
    code = str(code).zfill(6)
    if code.startswith(("688", "689")):
        return STAR_LIMIT_THR
    if code.startswith(("300", "301")):
        if date is not None and pd.Timestamp(date) < CHINEXT_REG_DATE:
            return MAIN_LIMIT_THR
        return STAR_LIMIT_THR
    return MAIN_LIMIT_THR


def add_limit_threshold(df: pd.DataFrame, code_col: str = "code", date_col: str = "date") -> pd.DataFrame:
    """给含 code/date 的 DataFrame 逐行添加涨停阈值列 `_limit_thr`（不修改原表）。
    向量化实现：主板=9.5%；科创板=19.5%；创业板按2020-08-24分界。"""
    out = df.copy()
    codes = out[code_col].astype(str).str.zfill(6)
    dates = pd.to_datetime(out[date_col])
    is_star = codes.str.startswith(("688", "689"))
    is_chinext = codes.str.startswith(("300", "301"))
    is_old_chinext = is_chinext & (dates < CHINEXT_REG_DATE)

    thr = pd.Series(MAIN_LIMIT_THR, index=out.index, dtype="float64")
    thr[is_star] = STAR_LIMIT_THR
    thr[is_chinext & ~is_old_chinext] = STAR_LIMIT_THR
    # 主板与2020-08-24前的创业板保持 MAIN_LIMIT_THR（10%）
    out["_limit_thr"] = thr
    return out


def add_chase_threshold(df: pd.DataFrame, code_col: str = "code", date_col: str = "date") -> pd.DataFrame:
    """在含 _limit_thr 列的表上追加追高过滤阈值列 `_chase_thr = _limit_thr - 0.5%`。"""
    if "_limit_thr" not in df.columns:
        df = add_limit_threshold(df, code_col=code_col, date_col=date_col)
    out = df.copy()
    out["_chase_thr"] = out["_limit_thr"] - CHASE_MARGIN
    return out
