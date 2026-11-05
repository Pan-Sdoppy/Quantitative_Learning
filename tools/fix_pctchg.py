# -*- coding: utf-8 -*-
"""
离线补齐主板 + 双创 parquet 中孤立缺失的 pct_chg 字段。

口径（与 core/data_loader.py、chinext_star/core/data_loader.py 下载逻辑一致）：
    pct_chg = close.pct_change() * 100   （前复权 close 本地自算，baostock 不提供该字段）

原则：
- 纯离线、单线程、不发起任何 baostock 联网请求；
- 只补 pct_chg 为 NaN 的行，现有非 NaN 值原样保留，不改动 OHLC/volume/amount 等任何其他列；
- close<=0 或前收<=0 时 pct_chg 保持 NaN，不写 inf；
- apply 模式下，仅对实际有填补行的文件先备份到 _pctchg_fix_backup/ 再写回；
- 超板新填行（主板 12%、科创板 22%、创业板 2020-08-24 后 22%/前 12%，次新前 60 行豁免）
  单列清单供人工复核，脚本不擅自丢弃或改写 close。

用法：
    python fix_pctchg.py            # dry-run 只统计、不写回
    python fix_pctchg.py --apply    # 备份后写回
"""
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "diagnose"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHINEXT_REG = pd.Timestamp("2020-08-24")
JUMP_MAIN = 0.12
JUMP_STAR = 0.22
JUMP_OLD_CHINEXT = 0.12
NEW_STOCK_EXEMPT_ROWS = 60

BOARDS = [
    {"name": "主板", "dir": ROOT / "data" / "raw",
     "prefixes": ("000", "001", "002", "003", "600", "601", "603", "605"), "kind": "main"},
    {"name": "双创", "dir": ROOT / "chinext_star" / "data" / "raw",
     "prefixes": ("300", "301", "688", "689"), "kind": "chinext"},
]


def jump_threshold(kind, code, dates):
    if kind == "main":
        return pd.Series(JUMP_MAIN, index=dates.index)
    if code.startswith(("688", "689")):
        return pd.Series(JUMP_STAR, index=dates.index)
    return pd.Series(np.where(dates < CHINEXT_REG, JUMP_OLD_CHINEXT, JUMP_STAR), index=dates.index)


def main():
    apply_mode = "--apply" in sys.argv
    mode_tag = "APPLY 写回" if apply_mode else "DRY-RUN 仅统计"
    lines = [f"=== pct_chg 离线补齐 {mode_tag} ==="]

    grand = {"files": 0, "fixed_files": 0, "filled": 0, "kept": 0,
             "still_nan": 0, "missing_col": 0, "bad_close": 0}
    over_rows = []
    missing_col_files = []
    bad_close_files = []

    for board in BOARDS:
        raw = board["dir"]
        prefixes = board["prefixes"]
        files = [f for f in sorted(glob.glob(str(raw / "*.parquet")))
                 if not os.path.basename(f).startswith("_")
                 and os.path.basename(f).replace(".parquet", "").startswith(prefixes)]
        b = {"files": len(files), "fixed_files": 0, "filled": 0, "kept": 0,
             "still_nan": 0, "missing_col": 0, "bad_close": 0}
        backup_dir = raw / "_pctchg_fix_backup"

        for fp in files:
            code = os.path.basename(fp).replace(".parquet", "")
            try:
                df = pd.read_parquet(fp)
            except Exception as e:
                b["missing_col"] += 1
                missing_col_files.append((code, f"读取失败: {e}"))
                continue
            if "pct_chg" not in df.columns or "close" not in df.columns:
                b["missing_col"] += 1
                missing_col_files.append((code, "缺 pct_chg/close 列"))
                continue

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            close = df["close"]
            prev_close = close.shift(1)
            recon = (close / prev_close - 1.0) * 100.0
            bad = close.isna() | (close <= 0) | prev_close.isna() | (prev_close <= 0)
            recon[bad] = np.nan

            cur = df["pct_chg"]
            fill_mask = cur.isna() & recon.notna()
            if len(fill_mask):
                fill_mask.iloc[0] = False  # 首日保持 NaN
            still_nan = int((cur.isna() & recon.isna()).sum())
            kept = int(cur.notna().sum())
            n_fill = int(fill_mask.sum())

            # 超板新填行复核清单
            if n_fill:
                thr = jump_threshold(board["kind"], code, df["date"])
                exempt = np.arange(len(df)) < NEW_STOCK_EXEMPT_ROWS
                for i in np.where(fill_mask.values)[0]:
                    if exempt[i]:
                        continue
                    if abs(recon.iloc[i] / 100.0) > thr.iloc[i]:
                        over_rows.append({
                            "板块": board["name"], "code": code,
                            "date": df.loc[i, "date"].strftime("%Y-%m-%d"),
                            "新填pct_chg%": round(float(recon.iloc[i]), 3),
                            "close": float(close.iloc[i]), "前收": float(prev_close.iloc[i]),
                            "阈值%": round(float(thr.iloc[i] * 100), 1),
                            "行数": len(df),
                        })

            b["fixed_files"] += int(n_fill > 0)
            b["filled"] += n_fill
            b["kept"] += kept
            b["still_nan"] += still_nan
            b["bad_close"] += int(bad.sum())

            if apply_mode and n_fill:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fp, backup_dir / f"{code}.parquet")
                df.loc[fill_mask, "pct_chg"] = recon[fill_mask].astype("float64")
                df.to_parquet(fp, index=False)

        for k in b:
            grand[k] += b[k]
        lines.append(
            f"\n[{board['name']}] 文件={b['files']}  有填补={b['fixed_files']}  "
            f"填补行={b['filled']}  保留原值行={b['kept']}  仍NaN={b['still_nan']}  "
            f"缺列/损坏={b['missing_col']}  close异常行={b['bad_close']}")

    lines.append(f"\n=== 合计 ===  文件={grand['files']} 有填补={grand['fixed_files']} "
                 f"填补行={grand['filled']} 保留={grand['kept']} 仍NaN={grand['still_nan']} "
                 f"缺列/损坏={grand['missing_col']} close异常={grand['bad_close']}")

    if missing_col_files:
        lines.append(f"\n-- 缺列/损坏文件 {len(missing_col_files)} --")
        for c, msg in missing_col_files[:50]:
            lines.append(f"  {c}: {msg}")

    lines.append(f"\n-- 超板新填行复核清单 {len(over_rows)} 条（次新前60行已豁免）--")
    if over_rows:
        odf = pd.DataFrame(over_rows).sort_values(["板块", "code", "date"])
        lines.append(odf.to_string(index=False))

    tag = "applied" if apply_mode else "dryrun"
    report = OUT_DIR / f"pctchg_fix_{tag}.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已写入: {report}")


if __name__ == "__main__":
    main()
