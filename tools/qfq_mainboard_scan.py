"""主板全量复权因子重建扫描（与双创 qfq_full_rebuild_scan.py 同口径，增加会话自愈）
=====================================================================
对主板 _stock_list.csv 全部个股：用 baostock 不复权(raw)+pctChg 从后往前递推
重建正确 qfq，与本地 parquet 逐日对齐对比；OHLC 相对差异 >0.2% 的票（幽灵除权
因子错位）先备份到 _qfq_rebuild_backup/，再用重建结果覆盖。断点续跑、单线程。

会话自愈（主板 2026-10-24 实测）：
- baostock 服务端会在运行约一段时间后静默踢断登录态，原脚本只在每 150 只刷新一次，
  断连期间所有票都报"用户未登录"被记 error；且旧断点逻辑会把 error 也当作已完成永久跳过。
- 本脚本：每次查询前确保登录；查询返回非 0 错误先强制 logout→login 重登再重试；
  续跑时 error 状态不视为完成，会重新纳入扫描；启动时先清洗进度文件中的 error 记录。
- socket 默认 30 秒超时：服务端静默断连/半开连接时 rs.next() 不再永久阻塞，超时抛错后
  走强制重登重试；每 10 只写一条心跳日志，便于监控进程是否再次卡死。
- baostock 为全局单例会话，本脚本必须在其他联网扫描完全结束后再启动，禁止并发。
"""
import json
import shutil
import socket
import time
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from core.utils import RAW_DIR, RESULT_DIR

BACKUP_DIR = RAW_DIR / "_qfq_rebuild_backup"
BACKUP_DIR.mkdir(exist_ok=True)
PROGRESS = RESULT_DIR / "diagnose" / "mainboard_qfq_scan_progress.jsonl"
PROGRESS.parent.mkdir(parents=True, exist_ok=True)
LOG = RESULT_DIR / "diagnose" / "mainboard_qfq_scan.log"

DIFF_THRESHOLD = 0.002
START = "2016-01-01"
END = "2026-10-23"
REFRESH_EVERY = 120  # 主动刷新会话间隔（只查本地不拉新数据，服务端断连风险高于双创实测）
SOCKET_TIMEOUT = 30  # baostock socket 默认超时（秒），防止 rs.next() 在半开连接上永久阻塞
socket.setdefaulttimeout(SOCKET_TIMEOUT)

# 终态：这些状态在断点续跑时视为已完成、不再重扫；error 必须重扫
TERMINAL = {"ok", "repaired", "date_mismatch", "missing_local", "raw_too_short"}

_logged_in = False


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_login(force=False):
    """建立/刷新 baostock 单例会话；force=True 时先登出再登录。返回 error_code。"""
    global _logged_in
    if force and _logged_in:
        try:
            bs.logout()
        except Exception:
            pass
        _logged_in = False
        time.sleep(1)
    if not _logged_in:
        lg = bs.login()
        _logged_in = lg.error_code == "0"
        log(f"baostock login: {lg.error_code} {lg.error_msg}")
        time.sleep(0.5)
    return "0" if _logged_in else "-1"


def fetch_raw(symbol, retry=4):
    """拉取不复权全历史；超时/未登录/会话失效时先强制重登再重试。"""
    last_err = None
    for attempt in range(retry):
        ensure_login()
        try:
            rs = bs.query_history_k_data_plus(
                symbol, "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=START, end_date=END, frequency="d", adjustflag="3")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0":
                # 会话失效/未登录：强制重登后重试；其他错误也先重登一次兜底
                last_err = RuntimeError(f"[{rs.error_code}] {rs.error_msg}")
                ensure_login(force=True)
                time.sleep(1.5)
                continue
            df = pd.DataFrame(rows, columns=rs.fields)
            for c in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df = df.drop_duplicates(subset=["date"]).reset_index(drop=True)
            return df
        except (socket.timeout, TimeoutError, OSError) as e:
            # 半开连接/静默断连：rs.next() 被 socket 超时打断，强制重登后重试
            last_err = RuntimeError(f"网络超时/连接异常(第{attempt + 1}/{retry}次): {e}")
            log(f"[{symbol}] {last_err}，强制重登后重试")
            ensure_login(force=True)
            time.sleep(2)
        except Exception as e:
            last_err = e
            ensure_login(force=True)
            time.sleep(2)
    raise RuntimeError(f"重试{retry}次仍失败: {last_err}")


def rebuild_qfq(raw):
    """从最新日起 f=1，用 raw close 与 pctChg 递推逐日因子，重算 OHLC。"""
    df = raw.sort_values("date").reset_index(drop=True).copy()
    n = len(df)
    f = np.ones(n)
    close = df["close"].values
    pct = df["pctChg"].values
    for i in range(n - 1, 0, -1):
        p = pct[i]
        if np.isnan(p) or close[i] <= 0 or close[i - 1] <= 0:
            f[i - 1] = f[i]
        else:
            f[i - 1] = f[i] * close[i] / close[i - 1] / (1 + p / 100.0)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        out[col] = df[col].values * f
    out["volume"] = df["volume"].fillna(-1).astype("int64")
    out["amount"] = df["amount"].values
    out["turnover"] = df["turn"].values
    out["pct_chg"] = out["close"].pct_change() * 100
    return out


def main():
    codes = sorted(pd.read_csv(RAW_DIR / "_stock_list.csv", dtype=str)["code"].str.zfill(6).tolist())

    # 断点续跑：只把终态记录视为已完成；error 清洗出去纳入重扫，并回写进度文件
    done = {}
    kept_records = []
    if PROGRESS.exists():
        for line in open(PROGRESS, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") in TERMINAL:
                done[r["code"]] = r
                kept_records.append(r)
            # error / parse_error 直接丢弃，下一轮重扫
    with open(PROGRESS, "w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    todo = [c for c in codes if c not in done]
    log(f"主板共 {len(codes)} 只，终态已完成 {len(done)}，待扫描 {len(todo)}（含重扫 error）")

    ensure_login(force=True)

    n_fixed = 0
    n_ok = 0
    n_fail = 0
    for idx, code in enumerate(todo):
        symbol = ("sh." if code.startswith("6") else "sz.") + code
        fp = RAW_DIR / f"{code}.parquet"
        rec = {"code": code}
        try:
            if not fp.exists():
                rec["status"] = "missing_local"
                n_fail += 1
            else:
                if (idx + 1) % REFRESH_EVERY == 0:
                    ensure_login(force=True)
                raw = fetch_raw(symbol)
                if raw.empty or len(raw) < 60:
                    rec["status"] = "raw_too_short"
                    n_fail += 1
                else:
                    new = rebuild_qfq(raw)
                    old = pd.read_parquet(fp)
                    old["date"] = pd.to_datetime(old["date"])
                    new2 = new[new["date"].isin(old["date"])].reset_index(drop=True)
                    old2 = old[old["date"].isin(new["date"])].sort_values("date").reset_index(drop=True)
                    if len(new2) != len(old2):
                        rec["status"] = "date_mismatch"
                        rec["old_rows"] = len(old)
                        rec["new_rows"] = len(new)
                        log(f"[{code}] 行数不一致 本地{len(old)} vs 重建{len(new)}，仅记录不覆盖")
                        n_fail += 1
                    else:
                        ratios = []
                        for col in ["open", "high", "low", "close"]:
                            a = old2[col].values
                            b = new2[col].values
                            mask = np.abs(a) > 1e-8
                            ratios.append(np.max(np.abs(b[mask] / a[mask] - 1)))
                        max_diff = max(ratios)
                        rec["status"] = "ok"
                        rec["max_diff"] = round(float(max_diff), 6)
                        if max_diff > DIFF_THRESHOLD:
                            shutil.copy2(fp, BACKUP_DIR / f"{code}.parquet.bak")
                            new.to_parquet(fp, index=False)
                            rec["status"] = "repaired"
                            n_fixed += 1
                            cm = np.abs(new2["close"].values / old2["close"].values - 1)
                            bad_i = int(np.argmax(cm))
                            rec["worst_date"] = str(old2["date"].iloc[bad_i].date())
                            rec["worst_diff"] = round(float(cm[bad_i]), 4)
                            log(f"[{code}] 修复! 最大差异 {max_diff*100:.2f}% @ {rec['worst_date']}, 备份+覆盖")
                        else:
                            n_ok += 1
        except Exception as e:
            rec["status"] = "error"
            rec["err"] = str(e)[:150]
            n_fail += 1
            log(f"[{code}] 异常: {e}")
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (idx + 1) % 10 == 0:
            log(f"心跳 {idx+1}/{len(todo)}  最新[{code}] {rec['status']}  正常{n_ok} 修复{n_fixed} 异常{n_fail}")
        if (idx + 1) % 50 == 0:
            log(f"进度 {idx+1}/{len(todo)}  正常{n_ok} 修复{n_fixed} 异常{n_fail}")

    bs.logout()
    _logged_in = False
    log(f"完成: 正常 {n_ok}, 修复 {n_fixed}, 异常/跳过 {n_fail}")


if __name__ == "__main__":
    main()
