"""
板块配置（决定股票池与涨跌幅口径）
==================================
本文件为「创业板+科创板（双创）」子项目的板块配置。
对应目录：chinext_star/

股票池：
  - 创业板 300 / 301
  - 科创板 688 / 689
  - 不含沪深主板（60/00）、不含北交所（8/4/920）。

涨跌幅规则（前复权口径，用于"涨停封单买不进"剔除）：
  - 科创板（688/689）：开市起即 ±20%，涨停阈值 19.5%（考虑四舍五入）。
  - 创业板（300/301）：2020-08-24 注册制改革前 ±10%，改革后 ±20%。
    回测严格从 2022 年起，回测期内创业板全部已是 20%；
    训练样本回溯到 2016 年，limit_up_threshold 仍按日期分界处理，保证口径严谨。
  - 次新股上市前 5 个交易日不设涨跌幅：数据层要求上市满 60 个交易日
    （MIN_LIST_DAYS=60）才纳入，训练/回测/预测样本中不会出现前 5 日无涨跌幅次新股，无需特判。

止盈/止损/min_pred/二次排序/市场择时等选股参数已在双创样本上独立回测调优（B16 生产口径），
不再沿用主板口径。
"""
import pandas as pd

BOARD_KEY = "chinext_star"
BOARD_NAME = "创业板+科创板（双创）"

# 股票池代码前缀正则（6位代码）：创业板300/301 + 科创板688/689
UNIVERSE_REGEX = r"^(300|301|688|689)"

# 关键日期
CHINEXT_REG_DATE = pd.Timestamp("2020-08-24")  # 创业板存量股涨跌幅由10%放宽至20%

# 涨停判定阈值（收益率 >= 该值视为涨停）
MAIN_LIMIT_THR = 0.095    # 10%板（考虑四舍五入用9.5%）
STAR_LIMIT_THR = 0.195    # 20%板（考虑四舍五入用19.5%）

# 盘中实时快照（新浪 stock_zh_a_spot）数量健壮性阈值：
# 多轮并集后总数低于 MIN 判定接口分页截断/异常并拒绝出预测；达到 TARGET 即提前停抓。
# 双创（300/301+688/689）：新浪 stock_zh_a_spot 实测单轮快照约 2027 只
# （创业板约1410 + 科创板约617，含ST，2026-10）；ST 在后续股票池过滤再剔。
# 多轮并集目标 2050（略高于单轮全量，确保抓全不漏票）；并集后低于 1900 判定分页截断/异常并拒绝。
SPOT_MIN_BOARD = 1900
SPOT_TARGET_BOARD = 2050

# 盘中实时价 vs qfq昨收 异常跳变阈值：超出涨停线过大视为复权串行污染，剔除该股当日行。
# 主板涨停10%用15%（+5%缓冲），双创涨停20%用21%（+1%缓冲）。
JUMP_BAD_MAIN = 0.15
JUMP_BAD_STAR = 0.21


def limit_up_threshold(code, date=None) -> float:
    """单只股票在某日的涨停判定阈值。
    科创板恒为20%；创业板在2020-08-24前为10%、之后为20%；其余（理论上不会出现在本股票池）按10%。"""
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
    向量化实现：科创板=19.5%；创业板按2020-08-24分界；其余=9.5%。"""
    out = df.copy()
    codes = out[code_col].astype(str).str.zfill(6)
    dates = pd.to_datetime(out[date_col])
    is_star = codes.str.startswith(("688", "689"))
    is_chinext = codes.str.startswith(("300", "301"))
    is_old_chinext = is_chinext & (dates < CHINEXT_REG_DATE)

    thr = pd.Series(MAIN_LIMIT_THR, index=out.index, dtype="float64")
    thr[is_star] = STAR_LIMIT_THR
    thr[is_chinext & ~is_old_chinext] = STAR_LIMIT_THR
    # 2020-08-24 前的创业板仍是 10%，保持 MAIN_LIMIT_THR
    out["_limit_thr"] = thr
    return out
