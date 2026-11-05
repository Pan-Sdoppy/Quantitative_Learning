"""
数据加载模块（全市场·只读共享）：
- 个股日线【不下载、不复制】，只读合并两个物理源目录：
  主板 = 仓库根目录 data/raw（由根目录项目更新）；双创 = chinext_star/data/raw（由双创项目更新）
- 加载时自动剔除 ST、退市、停牌、次新股与极端异常行
- 全市场股票列表/行业走在线接口（akshare/baostock），小缓存放本目录 data/raw
- 沪深300 指数优先读主板共享文件（根目录 data/raw）
"""
import socket
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import board_profile
from .utils import RAW_DIR, get_logger

warnings.filterwarnings("ignore")
logger = get_logger("data_loader")

socket.setdefaulttimeout(30)

START_DATE = "20160101"
MIN_LIST_DAYS = 60


def _bs_login(max_retries: int = 8, wait_sec: int = 8):
    """baostock 登录（带重试）。

    免费服务端在晚间/高峰常出现网络超时（timed out）、服务器连接失败、
    接收数据异常（error_code 10002007）等，单次登录成功率不稳定（实测约1/4）。
    每次失败先清理可能残留的坏 socket，再等待后重试；全部失败才抛错。
    返回已登录的 baostock 模块（调用方结束后仍需 bs.logout()）。
    """
    import baostock as bs
    lg = None
    for attempt in range(1, max_retries + 1):
        try:
            lg = bs.login()
        except Exception as e:
            logger.warning(f"baostock 登录异常(第{attempt}/{max_retries}次): {e}")
            lg = None
        if lg is not None and getattr(lg, "error_code", "1") == "0":
            if attempt > 1:
                logger.info(f"baostock 第{attempt}次尝试登录成功")
            return bs
        msg = getattr(lg, "error_msg", "未知错误")
        logger.warning(f"baostock 登录失败(第{attempt}/{max_retries}次): {msg}，{wait_sec}s后重试")
        # 失败后 socket 可能处于坏状态（WinError 10057），先清理再重试
        try:
            bs.logout()
        except Exception:
            pass
        if attempt < max_retries:
            time.sleep(wait_sec)
    raise RuntimeError(
        f"baostock 连续{max_retries}次登录失败（多为免费服务端网络拥塞，非本程序问题），"
        f"请稍等几分钟后重试 train.py --data-only"
    )


def _stock_parquet_files(d) -> list:
    """列出单个数据源目录下的个股 parquet（排除 _ 开头的缓存/指数文件）。"""
    if not d.exists():
        return []
    return [f for f in sorted(d.glob("*.parquet")) if not f.stem.startswith("_")]


def _stock_files() -> list:
    """全市场个股日线【只读合并】两个物理源目录（双创 chinext_star + 主板根目录），
    不复制、不在本目录下载。按 code 去重（双创在前优先；两源前缀不重叠，去重仅作保护）。"""
    seen, files = set(), []
    for d in board_profile.STOCK_RAW_DIRS:
        for fp in _stock_parquet_files(d):
            if fp.stem in seen:
                continue
            seen.add(fp.stem)
            files.append(fp)
    return sorted(files, key=lambda p: p.stem)


# ========== 股票列表 ==========
def _filter_universe_non_st(df: pd.DataFrame) -> pd.DataFrame:
    """股票列表统一过滤：按本子项目板块配置 board_profile.UNIVERSE_REGEX 限定股票池、
    剔除ST/*ST/退市。当前板块为：{name}（{regex}）。
    无论列表来自在线API还是本地缓存都必须过一遍，防止含ST或非本板块的脏缓存被直接返回、混入下载与训练""".format(
        name=board_profile.BOARD_NAME, regex=board_profile.UNIVERSE_REGEX)
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    st_mask = df["name"].astype(str).str.contains("ST|退", case=False, na=False)
    df = df[~st_mask]  # 剔除ST/*ST/退市
    df = df[df["code"].str.match(board_profile.UNIVERSE_REGEX)]  # 按板块配置限定股票池
    return df.reset_index(drop=True)


def get_stock_list(refresh: bool = False) -> pd.DataFrame:
    """获取A股全市场股票列表，剔除ST/退市，缓存到 _stock_list.csv（缓存7天）"""
    import socket
    cache_path = RAW_DIR / "_stock_list.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, dtype={"code": str})
        df = _filter_universe_non_st(df)  # 缓存也强制剔ST/非本板块，杜绝含ST脏缓存直接返回
        logger.info(f"从缓存加载股票列表(已剔ST/非本板块)，共 {len(df)} 只")
        return df

    import akshare as ak
    logger.info("正在获取A股全市场股票列表...")
    socket.setdefaulttimeout(30)
    try:
        df = ak.stock_info_a_code_name()
    except Exception as e:
        logger.warning(f"股票列表API失败({e})，尝试使用过期缓存")
        if cache_path.exists():
            df = pd.read_csv(cache_path, dtype={"code": str})
            df = _filter_universe_non_st(df)
            logger.info(f"回退到缓存股票列表(已剔ST/非本板块)，共 {len(df)} 只")
            return df
        raise
    finally:
        socket.setdefaulttimeout(None)
    df.columns = [c.lower() for c in df.columns]
    if "code" not in df.columns:
        for col in df.columns:
            if "代码" in col or "code" in col.lower():
                df = df.rename(columns={col: "code"})
                break
    if "name" not in df.columns:
        for col in df.columns:
            if "名称" in col or "name" in col.lower():
                df = df.rename(columns={col: "name"})
                break

    df = _filter_universe_non_st(df)  # 统一剔ST/退市、按板块配置限定股票池

    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    logger.info(f"股票列表获取完成，剔除ST后退市后共 {len(df)} 只，已缓存")
    return df


def get_stock_industry(refresh: bool = False) -> pd.DataFrame:
    """获取A股股票行业分类（baostock），缓存到 _stock_industry.csv（缓存30天）
    返回列: code(6位), name, industry(行业名称)
    """
    cache_path = RAW_DIR / "_stock_industry.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, dtype={"code": str})
        logger.info(f"从缓存加载股票行业分类，共 {len(df)} 只")
        return df

    logger.info("正在获取A股股票行业分类（baostock）...")
    try:
        bs = _bs_login()
    except RuntimeError as e:
        logger.warning(f"{e}，返回空行业表")
        return pd.DataFrame(columns=["code", "name", "industry"])
    try:
        rs = bs.query_stock_industry()
        data_list = []
        while (rs.error_code == "0") and rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
    finally:
        bs.logout()

    if df.empty:
        logger.warning("行业分类获取为空，返回空表")
        return pd.DataFrame(columns=["code", "name", "industry"])

    df.columns = [c.lower() for c in df.columns]
    # baostock返回: code, code_name, industry, industryclassification
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    if "code_name" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"code_name": "name"})
    keep = [c for c in ["code", "name", "industry"] if c in df.columns]
    df = df[keep].drop_duplicates(subset=["code"]).reset_index(drop=True)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    logger.info(f"股票行业分类获取完成，共 {len(df)} 只，已缓存")
    return df


def update_data(start_date: str = START_DATE, force_full: bool = False,
                is_intraday: bool = False) -> dict:
    """全市场子项目【不下载、不复制个股日线】。

    个股日线物理上只存一份，由两个源头项目各自维护，全市场只读合并：
      主板 -> 仓库根目录 data/raw（在仓库根目录运行 train.py --data-only 更新）
      双创 -> chinext_star/data/raw（在 chinext_star 运行 train.py --data-only 更新）
    因此本函数不登录 baostock、不写任何个股 parquet，仅做：
      1) 两个源目录存在性 / 个股数量校验；
      2) 刷新全市场股票列表缓存（akshare，在线列表，非日线下载，很轻量）。
    沪深300指数由 update_hs300 直接读主板共享文件。
    """
    stats = {"total": 0, "updated": 0, "new": 0, "failed": 0, "skipped": 0,
             "main": 0, "chinext": 0}

    n_main = len(_stock_parquet_files(board_profile.MAIN_RAW_DIR))
    n_cx = len(_stock_parquet_files(board_profile.CHINEXT_RAW_DIR))
    files = _stock_files()
    stats["main"] = n_main
    stats["chinext"] = n_cx
    stats["total"] = len(files)

    logger.info("=" * 70)
    logger.info("全市场数据源为【只读共享】模式（不复制、不下载个股）：")
    logger.info(f"  主板源 {board_profile.MAIN_RAW_DIR}：{n_main} 只")
    logger.info(f"  双创源 {board_profile.CHINEXT_RAW_DIR}：{n_cx} 只")
    logger.info(f"  合并可用：{len(files)} 只")
    logger.info("=" * 70)

    if not board_profile.MAIN_RAW_DIR.exists() or n_main == 0:
        raise RuntimeError(
            f"主板数据源不存在或为空：{board_profile.MAIN_RAW_DIR}。"
            "请先在【仓库根目录】运行 python train.py --data-only 更新主板数据。")
    if not board_profile.CHINEXT_RAW_DIR.exists() or n_cx == 0:
        raise RuntimeError(
            f"双创数据源不存在或为空：{board_profile.CHINEXT_RAW_DIR}。"
            "请先在【chinext_star】目录运行 python train.py --data-only 下载双创数据。")

    # 刷新全市场股票列表缓存（akshare 在线拉取列表，非日线下载，很轻量）
    try:
        get_stock_list(refresh=True)
    except Exception as e:
        logger.warning(f"全市场股票列表刷新失败（不影响训练，可稍后重试）：{e}")

    logger.info("全市场个股日线请在【根目录】与【chinext_star】两个源头分别更新，本项目不下载。")
    return stats


def load_all_data(start_date: str = "2016-01-01") -> pd.DataFrame:
    """加载本地所有股票数据，合并为长表，剔除停牌/异常/次新，float32降精度"""
    import gc
    files = _stock_files()  # 只读合并主板（根目录）+ 双创（chinext_star）两个源目录
    logger.info(f"加载本地数据，共 {len(files)} 只股票文件（主板源 + 双创源）")

    dfs = []
    for fp in tqdm(files, desc="加载数据", ncols=100):
        try:
            df = pd.read_parquet(fp)
            if df.empty or len(df) < MIN_LIST_DAYS:
                continue
            df["code"] = fp.stem
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= pd.Timestamp(start_date)]
            if df.empty:
                continue
            if "volume" in df.columns:
                df = df[df["volume"].fillna(0) > 0]
            df = df[(df["close"] > 0) & (df["close"] < 10000)]
            # 只保留因子计算需要的列
            keep = [c for c in ["date", "code", "open", "high", "low", "close",
                                "volume", "amount", "turnover"] if c in df.columns]
            df = df[keep]
            # 降精度
            for c in df.columns:
                if c not in ["date", "code"] and df[c].dtype == "float64":
                    df[c] = df[c].astype("float32")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"加载 {fp.name} 失败: {e}")

    if not dfs:
        raise RuntimeError("没有可用的本地数据，请先运行 train.py 下载数据")

    all_data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    all_data["code"] = all_data["code"].astype("category")
    all_data = all_data.sort_values(["date", "code"]).reset_index(drop=True)
    logger.info(f"数据加载完成: {len(all_data)} 行, {all_data['code'].nunique()} 只股票, "
                f"日期范围 {all_data['date'].min():%Y-%m-%d} ~ {all_data['date'].max():%Y-%m-%d}")

    # 异常值过滤：剔除日收益率绝对值超过50%的极端异常行（前复权跳变/数据错误）
    # 注意：不剔除20%-50%的行，避免剔除后下一行ret仍异常；因子计算时横截面±5σ截断会处理
    all_data["_ret"] = all_data.groupby("code")["close"].pct_change()
    before = len(all_data)
    all_data = all_data[(all_data["_ret"].abs() <= 0.50) | all_data["_ret"].isna()]
    all_data = all_data.drop(columns=["_ret"])
    if before != len(all_data):
        logger.info(f"剔除极端异常收益率(>50%)行: {before - len(all_data)} 行, 剩余 {len(all_data)} 行")

    return all_data


# ========== 沪深300指数数据（用于市场择时和超额收益计算） ==========
# 与主板共用同一份沪深300（根目录 data/raw），优先读共享文件；仅当根目录缺失时才在线下载到本目录 fallback。
HS300_PATH = RAW_DIR / "_index_hs300.parquet"                            # 本目录 fallback
HS300_SHARED_PATH = board_profile.MAIN_RAW_DIR / "_index_hs300.parquet"  # 主板共享（首选）


def update_hs300():
    """获取沪深300指数日线：优先直接读主板共享文件（随根目录数据更新，不重复下载）；
    根目录缺失时才在线下载并写本目录 fallback。"""
    if HS300_SHARED_PATH.exists():
        logger.info("沪深300指数与主板共用（读根目录 data/raw），无需单独下载")
        return pd.read_parquet(HS300_SHARED_PATH)
    import akshare as ak
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df is None or df.empty:
            logger.warning("沪深300指数数据为空")
            return None
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= pd.Timestamp("2016-01-01")]
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(HS300_PATH)
        logger.info(f"沪深300指数更新完成: {len(df)} 行, {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
        return df
    except Exception as e:
        logger.warning(f"沪深300指数下载失败: {e}")
        if HS300_PATH.exists():
            logger.info("回退到已缓存的沪深300数据")
            return pd.read_parquet(HS300_PATH)
        return None


def load_hs300() -> pd.DataFrame:
    """加载沪深300：优先主板共享文件，其次本目录 fallback，都没有再在线下载。"""
    if HS300_SHARED_PATH.exists():
        return pd.read_parquet(HS300_SHARED_PATH)
    if HS300_PATH.exists():
        return pd.read_parquet(HS300_PATH)
    logger.info("沪深300指数数据不存在，开始在线下载...")
    return update_hs300()
