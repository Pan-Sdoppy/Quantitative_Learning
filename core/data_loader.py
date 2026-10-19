"""
数据加载模块：
- A股全市场日线数据增量下载（baostock，稳定快速，不依赖V8引擎）
- 自动剔除ST、退市、停牌、次新股
- 本地parquet缓存 + 元数据缓存（避免重复读文件）
- 断点续存，失败不影响已下载数据
- tqdm进度条，单股2次重试
"""
import json
import socket
import threading
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import RAW_DIR, get_logger

warnings.filterwarnings("ignore")
logger = get_logger("data_loader")

socket.setdefaulttimeout(30)

START_DATE = "20160101"
MIN_LIST_DAYS = 60
MAX_WORKERS = 5
META_PATH = RAW_DIR / "_meta.json"

_stats_lock = threading.Lock()
_meta_lock = threading.Lock()


def _get_today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def get_expected_latest_trading_day(now: datetime = None) -> datetime:
    """获取预期的最新交易日（已收盘的最近一个交易日）
    - 周一到周五 15:00后：今天
    - 周一到周五 15:00前：上一个工作日
    - 周六周日：上一个周五
    （节假日由akshare接口自然处理，休市日下载返回空数据）
    """
    if now is None:
        now = datetime.now()
    d = now.replace(hour=0, minute=0, second=0, microsecond=0)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    if now.hour < 15 and d == now.replace(hour=0, minute=0, second=0, microsecond=0):
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
    return d


# ========== 元数据缓存 ==========
def _load_meta() -> dict:
    """加载股票最新日期元数据缓存 {code: 'YYYY-MM-DD'}"""
    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_meta(meta: dict):
    """保存元数据缓存（线程安全）"""
    with _meta_lock:
        try:
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"元数据保存失败: {e}")


def _update_meta(code: str, latest_date: str, meta: dict):
    """更新单只股票的元数据"""
    meta[code] = latest_date


def _build_meta_from_files() -> dict:
    """从已有parquet文件重建元数据（首次运行或元数据丢失时）"""
    meta = {}
    files = list(RAW_DIR.glob("*.parquet"))
    files = [f for f in files if not f.stem.startswith("_")]
    logger.info(f"重建元数据缓存，扫描 {len(files)} 个文件...")
    for fp in tqdm(files, desc="扫描元数据", ncols=100):
        try:
            df = pd.read_parquet(fp, columns=["date"])
            if not df.empty:
                meta[fp.stem] = df["date"].max().strftime("%Y-%m-%d")
        except Exception:
            pass
    _save_meta(meta)
    logger.info(f"元数据重建完成，共 {len(meta)} 只股票")
    return meta


# ========== 股票列表 ==========
def _filter_main_board_non_st(df: pd.DataFrame) -> pd.DataFrame:
    """股票列表统一过滤：仅沪深主板(60/00)、剔除ST/*ST/退市。
    无论列表来自在线API还是本地缓存都必须过一遍，防止含ST的脏缓存被直接返回、混入下载与训练"""
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    st_mask = df["name"].astype(str).str.contains("ST|退", case=False, na=False)
    df = df[~st_mask]  # 剔除ST/*ST/退市
    df = df[df["code"].str.match(r"^(60|00)")]  # 仅主板，剔除创业板30/科创板68/北交所
    return df.reset_index(drop=True)


def get_stock_list(refresh: bool = False) -> pd.DataFrame:
    """获取A股全市场股票列表，剔除ST/退市，缓存到 _stock_list.csv（缓存7天）"""
    import socket
    cache_path = RAW_DIR / "_stock_list.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, dtype={"code": str})
        df = _filter_main_board_non_st(df)  # 缓存也强制剔ST/非主板，杜绝含ST脏缓存直接返回
        logger.info(f"从缓存加载股票列表(已剔ST/非主板)，共 {len(df)} 只")
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
            df = _filter_main_board_non_st(df)
            logger.info(f"回退到缓存股票列表(已剔ST/非主板)，共 {len(df)} 只")
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

    df = _filter_main_board_non_st(df)  # 统一剔ST/退市、仅保留沪深主板

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

    import baostock as bs
    logger.info("正在获取A股股票行业分类（baostock）...")
    lg = bs.login()
    if lg.error_code != "0":
        logger.warning(f"baostock登录失败: {lg.error_msg}，返回空行业表")
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


def _code_to_baostock_symbol(code: str) -> str:
    """转换为baostock代码格式：sh.600000 / sz.000001"""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "90")):
        return f"sh.{code}"
    elif code.startswith(("00", "30", "20")):
        return f"sz.{code}"
    return f"sz.{code}"


def _download_single(code: str, start_date: str, end_date: str, retries: int = 2) -> pd.DataFrame:
    """下载单只股票日线数据（baostock，前复权，稳定快速，不依赖V8引擎）"""
    import baostock as bs
    symbol = _code_to_baostock_symbol(code)
    # 日期格式转换：YYYYMMDD -> YYYY-MM-DD
    s = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    e = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    for attempt in range(retries):
        try:
            time.sleep(0.05)
            rs = bs.query_history_k_data_plus(
                symbol, "date,open,high,low,close,volume,amount,turn",
                start_date=s, end_date=e, frequency="d", adjustflag="2"  # 2=前复权
            )
            if rs.error_code != "0":
                if attempt < retries - 1:
                    time.sleep(0.5)
                    continue
                logger.warning(f"股票 {code}({symbol}) 下载失败: {rs.error_msg}")
                return pd.DataFrame()
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=rs.fields)
            # 转数值
            for c in ["open", "high", "low", "close", "volume", "amount", "turn"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df["pct_chg"] = df["close"].pct_change() * 100
            df = df.rename(columns={"turn": "turnover"})
            # 硬校验：相邻交易日涨跌幅超过±20%说明baostock返回了串行/残留数据，丢弃
            # 次新股上市初期（前60个交易日）不设涨跌幅/波动大，跳过校验（与次新股过滤口径一致）
            # 增量下载时df行数少自然跳过；全量下载时排除前60行
            if len(df) > 60:
                check_df = df.iloc[60:]
                if len(check_df) >= 2:
                    max_jump = check_df["close"].pct_change().abs().max()
                    if max_jump > 0.20:
                        logger.warning(
                            f"股票 {code}({symbol}) 数据异常(单日跳空{max_jump * 100:.1f}%)，可能串行或除权日qfq复权滞后，丢弃重试")
                        if attempt < retries - 1:
                            time.sleep(1)
                            continue
                        return pd.DataFrame()
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                logger.warning(f"股票 {code}({symbol}) 下载失败: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def _process_one_stock(code: str, name: str, start_date: str, today: str,
                       force_full: bool, stats: dict, failed_codes: list,
                       meta: dict, is_intraday: bool = False,
                       expected_latest: datetime = None) -> str:
    """处理单只股票，使用元数据缓存快速判断是否需要下载
    is_intraday=True时，即使meta显示今天已更新，也强制重新下载今天的数据（盘中价格在变化）
    expected_latest: 预期的最新交易日，本地数据已到此日期则skip，否则下载
    增量衔接跳变（>20%）时转全量重拉：串行污染可被纠正；除权日qfq复权滞后当天
    全量重拉也会被硬校验拦截（返回空），此时保留本地旧数据、计skipped，次日复权因子
    更新后重跑自动全量重拉修复。
    """
    file_path = RAW_DIR / f"{code}.parquet"
    today_ts = pd.Timestamp(today)
    retry_after_jump = False  # 衔接跳变转全量重拉的标志

    # 用元数据快速判断是否已最新
    if not force_full and code in meta:
        try:
            latest = pd.Timestamp(meta[code])
            # 盘中模式：即使今天已更新过，也要重新下载今天的数据（价格在变化）
            effective_latest = latest - timedelta(days=1) if is_intraday else latest

            # 用预期最新交易日判断是否需要下载（替代粗糙的skip_days逻辑）
            if expected_latest is not None and effective_latest >= expected_latest:
                with _stats_lock:
                    stats["skipped"] += 1
                return "skip"
            # 增量下载：往前多覆盖3天，确保最新交易日数据完整（盘中下载可能不完整）
            dl_start = (latest - timedelta(days=3)).strftime("%Y%m%d")
            new_data = _download_single(code, dl_start, today)
            if new_data.empty:
                # 无新数据（可能是节假日/休市），保持原meta不变，skip
                with _stats_lock:
                    stats["skipped"] += 1
                return "skip"
            # 合并
            existing = pd.read_parquet(file_path)
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
            new_data["date"] = pd.to_datetime(new_data["date"], errors="coerce")
            # 衔接校验：增量下载可能因baostock响应错位/串行把别的股票数据写入本文件。
            # 注意：增量下载会向前多覆盖3天（含与本地重叠的历史行），因此不能拿新数据首行
            # 与本地末日直接比较（二者可能相隔数个交易日，期间正常涨跌停会造成>20%假象）；
            # 必须只取严格晚于本地末日的第一行（真正新增的首个交易日）与本地末日比较。
            # 主板单日涨跌停±10%，>20%的两种原因：①串行污染；②除权日qfq复权因子当天滞后
            # （如10转4.5，真实价直接跌约30%）；次新股上市初期无涨跌幅限制，豁免。
            if not existing.empty and not new_data.empty:
                ex_sorted = existing.sort_values("date")
                new_sorted = new_data.sort_values("date")
                last_date = ex_sorted["date"].iloc[-1]
                last_existing = ex_sorted["close"].iloc[-1]
                added = new_sorted[new_sorted["date"] > last_date]
                if not added.empty:
                    first_new = added["close"].iloc[0]
                    if pd.notna(last_existing) and pd.notna(first_new) and last_existing > 0:
                        jump = (first_new - last_existing) / last_existing
                        is_new_stock = len(existing) < 60  # 次新股上市初期波动大，跳过校验
                        if abs(jump) > 0.20 and not is_new_stock:
                            # 两种可能：① baostock响应错位/串行污染（全量重拉可纠正）；
                            # ② 除权除息日当天qfq复权因子滞后（次日才更新，如10转4.5除权日
                            # 真实价直接跌30%），此时全量重拉会触发_download_single自身硬校验
                            # 返回空，保留本地旧数据，等次日复权因子更新后全量重拉自愈。
                            # 两种情况都不能直接合并，抛错转入下面的全量重拉路径。
                            retry_after_jump = True
                            raise RuntimeError(
                                f"增量衔接跳变{jump * 100:+.1f}%（{last_existing:.2f}->{first_new:.2f}）"
                                f"，转全量重拉校验（串行污染或除权日qfq滞后）"
                            )
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
            combined.to_parquet(file_path, index=False)
            _update_meta(code, combined["date"].max().strftime("%Y-%m-%d"), meta)
            with _stats_lock:
                stats["updated"] += 1
            return "updated"
        except Exception as e:
            logger.warning(f"股票 {code} 增量更新失败，重新全量下载: {e}")

    # 全量下载（新股票或force_full或元数据缺失，或增量衔接跳变后自愈重拉）
    new_data = _download_single(code, start_date, today)
    if new_data.empty:
        if retry_after_jump:
            # 衔接跳变后全量重拉仍失败：除权日qfq复权滞后（次日更新后自愈）或持续污染，
            # 本地旧数据完好，计skipped（不是网络失败），保留旧数据，次日重跑自动修复。
            logger.warning(f"股票 {code} 全量重拉仍异常，保留本地数据，次日自动重试（除权日qfq滞后或持续污染）")
            with _stats_lock:
                stats["skipped"] += 1
            return "skip"
        with _stats_lock:
            stats["failed"] += 1
            failed_codes.append(code)
        return "failed"
    if len(new_data) < MIN_LIST_DAYS:
        with _stats_lock:
            stats["skipped"] += 1
        return "new_skip"
    new_data.to_parquet(file_path, index=False)
    _update_meta(code, new_data["date"].max().strftime("%Y-%m-%d"), meta)
    with _stats_lock:
        stats["new"] += 1
    return "new"


def update_data(start_date: str = START_DATE, force_full: bool = False,
                max_workers: int = MAX_WORKERS, is_intraday: bool = False) -> dict:
    """增量更新全市场数据（baostock，单线程，稳定快速不依赖V8引擎）
    is_intraday=True时，强制重新下载今天的数据（盘中价格在变化）
    """
    import baostock as bs
    # baostock登录（全局只需一次）
    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock登录失败: {lg.error_msg}")
        raise RuntimeError(f"baostock登录失败: {lg.error_msg}")
    logger.info("baostock登录成功")

    today = _get_today_str()
    expected_latest = get_expected_latest_trading_day()
    logger.info(f"预期最新交易日: {expected_latest:%Y-%m-%d}")

    # 加载元数据缓存，不存在则重建
    meta = _load_meta()
    if not meta:
        meta = _build_meta_from_files()

    # 股票列表：缓存超过7天才刷新（股票列表变化极慢）
    cache_path = RAW_DIR / "_stock_list.csv"
    need_refresh = True
    if cache_path.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age < timedelta(days=7):
            need_refresh = False
    stock_list = get_stock_list(refresh=need_refresh)
    total = len(stock_list)

    mode_str = "盘中实时" if is_intraday else "增量"
    logger.info(f"开始{mode_str}更新数据，目标日期 {today}，共 {total} 只股票（单线程）")
    logger.info(f"元数据缓存: {len([k for k in meta if not k.startswith('__')])} 只股票已有记录")

    stats = {"total": total, "updated": 0, "new": 0, "failed": 0, "skipped": 0}
    failed_codes = []
    meta_save_counter = 0

    # 单线程顺序处理（py_mini_racer/V8不支持多线程）
    pbar = tqdm(total=total, desc="下载进度", ncols=100)
    for row in stock_list.itertuples(index=False):
        try:
            status = _process_one_stock(
                row.code, row.name, start_date, today,
                force_full, stats, failed_codes, meta,
                is_intraday=is_intraday, expected_latest=expected_latest
            )
            pbar.set_postfix_str(f"{row.code} {status}")
        except Exception as e:
            logger.warning(f"股票 {row.code} 处理异常: {e}")
            stats["failed"] += 1
            failed_codes.append(row.code)
        pbar.update(1)
        meta_save_counter += 1
        if meta_save_counter % 200 == 0:
            _save_meta(meta)

    pbar.close()
    _save_meta(meta)

    # baostock登出
    try:
        bs.logout()
        logger.info("baostock登出")
    except Exception:
        pass

    if failed_codes:
        logger.warning(f"下载失败股票({len(failed_codes)}): {failed_codes[:20]}...")
    logger.info(f"数据更新完成: 新增={stats['new']}, 更新={stats['updated']}, "
                f"跳过={stats['skipped']}, 失败={stats['failed']}")
    return stats


def load_all_data(start_date: str = "2016-01-01") -> pd.DataFrame:
    """加载本地所有股票数据，合并为长表，剔除停牌/异常/次新，float32降精度"""
    import gc
    files = sorted(RAW_DIR.glob("*.parquet"))
    files = [f for f in files if not f.stem.startswith("_")]
    logger.info(f"加载本地数据，共 {len(files)} 只股票文件")

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
HS300_PATH = RAW_DIR / "_index_hs300.parquet"


def update_hs300():
    """下载/更新沪深300指数日线数据（前复权）"""
    import akshare as ak
    today = _get_today_str()
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
    """加载沪深300指数数据，如不存在则下载"""
    if HS300_PATH.exists():
        return pd.read_parquet(HS300_PATH)
    logger.info("沪深300指数数据不存在，开始下载...")
    return update_hs300()
