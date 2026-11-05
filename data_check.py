# -*- coding: utf-8 -*-
"""
周末全量日线数据质量体检（主板 + 双创），离线 / 只读 / 绝不修改任何数据
=========================================================================
每周周末运行一次，对两个数据目录做全量体检：
  主板：data/raw/                 前缀 000/001/002/003/600/601/603/605，±10% 板，跳变阈值 0.12
  双创：chinext_star/data/raw/    前缀 300/301（创业板）/688/689（科创板），
                                  科创板及 2020-08-24 后创业板 ±20% 板，阈值 0.22；
                                  2020-08-24 前创业板 0.12

检查维度（已经过真实数据校准，区分“真问题”与“正常现象”）：
  A. 文件可读取性：parquet 损坏 / 空文件 / 缺列
  B. 结构合法性：OHLC 非正、OHLC 关系（low<=open/close<=high）、非停牌 NaN
  C. 日期合法性：重复 / 乱序 / 非交易日（以沪深300指数为日历）/ 周末
  D. 单日跳变（qfq close pct_change 超涨跌停）：
       与 pct_chg 原始列交叉——
         两者一致（差<1个百分点）= 真实无涨跌幅限制日（重组复牌/股改/恢复上市/退市整理），列“真实事件”，不报错；
         两者不一致 = 复权因子错位/串行污染，判硬错误。
       上市前 60 个交易日（次新）与停牌日豁免。
  E. 串行错位（强指纹）：同一交易日两只及以上非停牌股票 OHLC **且成交量** 完全相同
       （加成交量后两只不同票完全相同的概率为 0；仅 OHLC 相同会被低价股1分钱档位碰撞误报，已弃用）。
       票对累计 >=2 天判硬错误，单日巧合只记录不判错。
  F. 价格冻结：非停牌连续 >=5 日 close/high/low 全等（正常几乎不可能）。
  G. pct_chg 字段批量缺失：按交易日聚合，单日缺失票 >=50 视为“整批未拉该字段”
       （常见于近期某批次增量下载遗漏 pct_chg），列清单，判需补数据。
  H. 数据更新滞后/缺口：非次新票末日滞后交易日历——
       末段无成交（停牌）→ 停牌清单，正常；
       非停牌滞后 3~6 个交易日 → 警告（更新不齐）；滞后 >=7 个交易日 → 硬错误（数据缺口）。
       另检查全市场末日是否一致（众数占比 <95% 报警）。
  I. 缺票（stock_list 有、parquet 无）、行数恰好 2000（历史分页截断特征）、板块越界。

停牌统一识别：volume 为 NaN / <=0 / == -1（下载哨兵）均视为停牌，相关跳变/冻结/NaN 全部豁免。

输出：控制台汇总 + results/diagnose/data_check_qc.xlsx（多 sheet）
退出码：0 = 无硬错误；1 = 存在硬错误或需补数据（结构/复权错位/串行/缺口/pct_chg批量缺失）。

局限（必须知道）：离线能 100% 抓“超涨跌停”级复权错位与串行复制，但抓不到 0.2%~几个百分点的
微小复权因子偏差——那必须用 baostock 不复权 raw + pctChg 递推重建交叉验证（联网深度扫描）：
  主板：python qfq_mainboard_scan.py
  双创：python chinext_star/qfq_full_rebuild_scan.py
建议每月/每季度或数据源异常后额外跑一次。

用法：
  python data_check.py
"""
import glob
import os
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
LAG_WARN = 3  # 非停牌末日滞后交易日历：3~6 警告
LAG_HARD = 7  # >=7 硬错误
FREEZE_RUN = 5  # 连续价格冻结天数
PAGE_CAP = 2000  # 历史分页截断特征行数
BATCH_MISSING = 50  # 单日 pct_chg 缺失票数达到该值判“批量缺失”
JUMP_CROSS_TOL = 1.0  # 超板跳变 qfq 与 pct_chg 一致容差（百分点）
SERIAL_MIN_DAYS = 2  # 强指纹票对累计天数达到该值判串行
END_CONSENSUS = 0.95  # 全市场末日众数占比阈值
REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]

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


def is_halt(vol):
    """停牌：volume 为 NaN / <=0 / == -1（下载哨兵）。"""
    return vol.isna() | (vol.fillna(0) <= 0) | (vol.fillna(0) == -1)


def scan_board(board):
    name, raw, kind, prefixes = board["name"], board["dir"], board["kind"], board["prefixes"]
    issues, jump_rows, frozen_rows, gap_rows, fp_parts, file_report = [], [], [], [], [], []
    pc_missing = []  # (date, code) pct_chg 非首日缺失

    def add_issue(code, kind_, detail):
        issues.append({"板块": name, "code": code, "类别": kind_, "detail": detail})

    files = sorted(glob.glob(str(raw / "*.parquet")))
    stock_files = [f for f in files if not os.path.basename(f).startswith("_")]
    print(f"\n[{name}] 待检个股文件: {len(stock_files)}  ({raw})")

    cal_dates = []
    idx_path = raw / "_index_hs300.parquet"
    if idx_path.exists():
        idx = pd.read_parquet(idx_path)
        idx["date"] = pd.to_datetime(idx["date"])
        cal_dates = sorted(idx["date"].unique())
        print(f"  沪深300日历: {len(cal_dates)} 行, 末日 {pd.Timestamp(cal_dates[-1]).date()}")
    else:
        add_issue("_INDEX", "缺指数", "沪深300指数文件不存在，无法做非交易日/末日校验")

    for i, fp in enumerate(stock_files):
        code = os.path.basename(fp).replace(".parquet", "")
        if not code.startswith(prefixes):
            add_issue(code, "板块越界", f"代码前缀不属于{name}清单: {code}")
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            add_issue(code, "文件损坏", f"parquet 读取失败: {e}")
            continue
        if df.empty:
            add_issue(code, "空文件", "DataFrame 为空")
            continue
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            add_issue(code, "缺列", f"缺失列: {missing}")
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        halt = is_halt(df["volume"])

        # --- C. 日期合法性 ---
        if df["date"].duplicated().any():
            add_issue(code, "重复日期", f"{int(df['date'].duplicated().sum())} 个重复交易日")
        if not df["date"].is_monotonic_increasing:
            add_issue(code, "日期乱序", "date 非严格递增")
        if cal_dates:
            illegal = sorted(set(df["date"]) - set(cal_dates))
            wk = [d for d in illegal if d.weekday() >= 5]
            other = [d for d in illegal if d.weekday() < 5]
            if wk:
                add_issue(code, "周末行", f"{len(wk)} 个周末日期")
            if other:
                add_issue(code, "非交易日行", f"{len(other)} 个工作日日期不在交易日历: "
                                              f"{[d.strftime('%Y-%m-%d') for d in other[:5]]}")

        # --- B. 结构合法性 ---
        for c in ["open", "high", "low", "close"]:
            k = int((df[c] <= 0).sum())
            if k:
                add_issue(code, "非正价格", f"{c} 有 {k} 个<=0")
            kn = int(df[c].isna().sum())
            if kn:
                add_issue(code, "OHLC空值", f"{c} 有 {kn} 个 NaN")
        ohlc_bad = ((df["high"] < df["low"]) | (df["high"] < df["open"]) |
                    (df["high"] < df["close"]) | (df["low"] > df["open"]) |
                    (df["low"] > df["close"]))
        if ohlc_bad.any():
            add_issue(code, "OHLC逻辑错误", f"{int(ohlc_bad.sum())} 行 high/low 与 open/close 矛盾")
        # 非停牌的 volume/amount NaN / 负值才算问题；停牌哨兵(-1/NaN)豁免
        act = ~halt
        for c in ["volume", "amount"]:
            kn = int((df[c].isna() & act).sum())
            if kn:
                add_issue(code, "非停牌空值", f"非停牌日 {c} 有 {kn} 个 NaN")
        nv = int(((df["volume"] < 0) & (df["volume"] != -1) & act).sum())
        if nv:
            add_issue(code, "负成交量", f"非停牌日 volume 有 {nv} 个负值")
        na = int(((df["amount"] < 0) & act).sum())
        if na:
            add_issue(code, "负成交额", f"非停牌日 amount 有 {na} 个负值")

        # --- G. pct_chg 缺失（非首日），聚合到板块级，不逐条刷硬错误 ---
        pc_nan = df["pct_chg"].isna().copy()
        if len(pc_nan) > 0:
            pc_nan.iloc[0] = False
        if pc_nan.any():
            for d in df.loc[pc_nan, "date"]:
                pc_missing.append((d, code))

        # --- D. 单日跳变（交叉 pct_chg）---
        df["ret"] = df["close"].pct_change()
        thr = jump_threshold(kind, code, df["date"])
        over = (df["ret"].abs() > thr) & df["ret"].notna()
        for j in np.where(over.values)[0]:
            r = df.iloc[j]
            exempt = j < NEW_STOCK_EXEMPT_ROWS
            is_halt_day = bool(halt.iloc[j])
            q = float(r["ret"]) * 100
            p = float(r["pct_chg"]) if pd.notna(r["pct_chg"]) else np.nan
            real = pd.notna(p) and abs(q - p) < JUMP_CROSS_TOL
            if exempt or is_halt_day:
                flag = "豁免"
            elif real:
                flag = "真实事件"  # 复牌/股改/恢复上市等无涨跌幅日
            else:
                flag = "疑似错位"
                add_issue(code, "复权错位跳变",
                          f"{r['date'].date()} qfq {q:.1f}% 与 pct_chg {p if pd.notna(p) else 'NaN'} 不一致")
            jump_rows.append({
                "板块": name, "code": code, "date": r["date"].strftime("%Y-%m-%d"),
                "行号": int(j), "qfq涨跌幅%": round(q, 2),
                "pct_chg原始%": round(p, 2) if pd.notna(p) else np.nan,
                "阈值%": round(float(thr.iloc[j]) * 100, 1),
                "判定": flag,
                "close": round(float(r["close"]), 4),
            })

        # --- F. 价格冻结（非停牌连续>=5日）---
        active = ((df["close"].diff() == 0) & (df["high"].diff() == 0) &
                  (df["low"].diff() == 0) & act)
        grp = (~active).cumsum()
        run_len = active.groupby(grp).cumsum()
        for j in np.where(run_len.values == FREEZE_RUN)[0]:
            seg = df.iloc[max(0, j - FREEZE_RUN + 1): j + 1]
            avg_amt = float(seg["amount"].mean())
            frozen_rows.append({
                "板块": name, "code": code,
                "date": df.iloc[j]["date"].strftime("%Y-%m-%d"),
                "连续不变交易日>=": FREEZE_RUN, "close": float(df.iloc[j]["close"]),
                "段内日均成交额": round(avg_amt, 0),
                "提示": "有巨额成交→多为真实钉盘/1元保卫战；成交枯竭→疑似数据冻结，需人工确认",
            })
            # 价格冻结属极罕见现象，作“提示人工复核”而非硬错误（历史上仅 ST 大集1元保卫战等真实钉盘出现）
            # 不 add_issue、不影响退出码，仅入 Excel“价格冻结”页。

        # --- E. 强指纹（OHLC+volume），仅非停牌 ---
        fp = pd.DataFrame({
            "date": df["date"].values,
            "open": df["open"].astype("float32").round(3).values,
            "high": df["high"].astype("float32").round(3).values,
            "low": df["low"].astype("float32").round(3).values,
            "close": df["close"].astype("float32").round(3).values,
            "volume": df["volume"].fillna(0).astype("int64").values,
            "code": code,
        })
        fp_parts.append(fp[fp["volume"] > 0])

        # --- H. 末日滞后 / 停牌 ---
        if n > NEW_STOCK_EXEMPT_ROWS and cal_dates:
            end = pd.Timestamp(df["date"].max())
            lag = int((pd.Series(cal_dates) > end).sum())
            tail_halt = is_halt(df.tail(5)["volume"])
            if lag >= LAG_WARN:
                if bool(tail_halt.all()):
                    gap_rows.append({"板块": name, "code": code, "末日": end.strftime("%Y-%m-%d"),
                                     "滞后天数": lag, "判定": "停牌（末段无成交）", "问题": "否"})
                elif lag >= LAG_HARD:
                    gap_rows.append({"板块": name, "code": code, "末日": end.strftime("%Y-%m-%d"),
                                     "滞后天数": lag, "判定": "数据缺口（非停牌、末段有成交但停更）", "问题": "是"})
                    add_issue(code, "数据缺口", f"末日 {end.date()} 滞后全局 {lag} 个交易日且非停牌")
                else:
                    gap_rows.append({"板块": name, "code": code, "末日": end.strftime("%Y-%m-%d"),
                                     "滞后天数": lag, "判定": "更新滞后（非停牌）", "问题": "警告"})
        if n == PAGE_CAP:
            add_issue(code, "行数疑似分页截断", f"行数恰好 {PAGE_CAP}（历史分页上限特征，需核对）")

        file_report.append({
            "板块": name, "code": code, "行数": n,
            "首日": df["date"].min().strftime("%Y-%m-%d"),
            "末日": df["date"].max().strftime("%Y-%m-%d"),
            "停牌日": int(halt.sum()),
        })
        if (i + 1) % 500 == 0:
            print(f"  已检 {i + 1}/{len(stock_files)}")

    # --- E. 强指纹：OHLC+volume 全相同（向量化 nunique），按票对累计天数 ---
    dup_rows, serial_rows = [], []
    if fp_parts:
        fp_all = pd.concat(fp_parts, ignore_index=True)
        del fp_parts
        g = fp_all.groupby(["date", "open", "high", "low", "close", "volume"], observed=True)
        nun = g["code"].nunique()
        for key in nun[nun >= 2].index:
            sub = g.get_group(key)
            codes = sorted(sub["code"].unique())
            dup_rows.append({
                "板块": name, "date": pd.Timestamp(key[0]).strftime("%Y-%m-%d"),
                "open": float(key[1]), "high": float(key[2]), "low": float(key[3]),
                "close": float(key[4]), "volume": int(key[5]),
                "涉及股票": ",".join(codes), "只数": len(codes),
            })
        # 票对累计天数
        pair_days = {}
        for r in dup_rows:
            cs = sorted(r["涉及股票"].split(","))
            for a in range(len(cs)):
                for b in range(a + 1, len(cs)):
                    pair_days[(cs[a], cs[b])] = pair_days.get((cs[a], cs[b]), 0) + 1
        for (a, b), dnum in pair_days.items():
            serial_rows.append({"板块": name, "股票A": a, "股票B": b, "OHLC+量相同天数": dnum})
            if dnum >= SERIAL_MIN_DAYS:
                add_issue(a, "串行错位", f"与 {b} 有 {dnum} 个非停牌日 OHLC+成交量完全相同")
                add_issue(b, "串行错位", f"与 {a} 有 {dnum} 个非停牌日 OHLC+成交量完全相同")
        del fp_all

    # --- G. pct_chg 批量缺失（按交易日聚合）---
    pc_df = pd.DataFrame(pc_missing, columns=["date", "code"])
    batch_rows = []
    if not pc_df.empty:
        cnt = pc_df.groupby(pc_df["date"].dt.strftime("%Y-%m-%d"))["code"].agg(
            ["count", lambda s: ",".join(sorted(s.unique())[:8])])
        cnt.columns = ["缺失票只数", "票样"]
        for d, row in cnt.iterrows():
            is_batch = int(row["缺失票只数"]) >= BATCH_MISSING
            batch_rows.append({"板块": name, "日期": d, "缺失票只数": int(row["缺失票只数"]),
                               "批量缺失": "是" if is_batch else "否", "票样": row["票样"]})
            if is_batch:
                add_issue("_BATCH", "pct_chg批量缺失",
                          f"{d} 有 {int(row['缺失票只数'])} 只票 pct_chg 为空（该批次未拉该字段，需补数据）")

    # --- 缺票 ---
    missing = []
    sl = raw / "_stock_list.csv"
    have = {os.path.basename(f).replace(".parquet", "") for f in stock_files}
    if sl.exists():
        lst = pd.read_csv(sl, dtype={"code": str})
        lst["code"] = lst["code"].str.zfill(6)
        for _, r in lst[~lst["code"].isin(have)].iterrows():
            missing.append({"板块": name, "code": r["code"], "name": r.get("name", "")})

    # --- H. 全市场末日一致性（非次新票）---
    fdf = pd.DataFrame(file_report)
    consensus = None
    if not fdf.empty:
        vc = fdf["末日"].value_counts()
        consensus = (vc.index[0], int(vc.iloc[0]), int(vc.iloc[0] / len(fdf) * 100))
        if vc.iloc[0] / len(fdf) < END_CONSENSUS:
            add_issue("_MARKET", "末日不一致",
                      f"最常见末日 {vc.index[0]} 仅占 {vc.iloc[0] / len(fdf) * 100:.1f}%，全市场更新不齐")

    return {
        "name": name, "n_files": len(stock_files), "consensus": consensus,
        "issues": pd.DataFrame(issues, columns=["板块", "code", "类别", "detail"]),
        "jumps": pd.DataFrame(jump_rows),
        "frozen": pd.DataFrame(frozen_rows),
        "gaps": pd.DataFrame(gap_rows),
        "dups": pd.DataFrame(dup_rows),
        "serial": pd.DataFrame(serial_rows, columns=["板块", "股票A", "股票B", "OHLC+量相同天数"]),
        "batch": pd.DataFrame(batch_rows),
        "missing": pd.DataFrame(missing, columns=["板块", "code", "name"]),
        "files": fdf,
    }


def main():
    print("=" * 72)
    print("周末全量数据体检（主板 + 双创），离线只读，不修改任何数据")
    print("=" * 72)
    results = [scan_board(b) for b in BOARDS]

    def cat(key):
        dfs = [r[key] for r in results if not r[key].empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    idf, jdf, fdf, gdf, ddf, sdf, bdf, mdf, rdf = (cat(k) for k in
                                                   ["issues", "jumps", "frozen", "gaps", "dups", "serial", "batch",
                                                    "missing", "files"])

    print("\n" + "=" * 72)
    print("总体汇总")
    print("=" * 72)
    for r in results:
        n = r["name"]
        j = r["jumps"]
        misalign = j[j["判定"] == "疑似错位"] if not j.empty else pd.DataFrame()
        real = j[j["判定"] == "真实事件"] if not j.empty else pd.DataFrame()
        batch = r["batch"][r["batch"]["批量缺失"] == "是"] if not r["batch"].empty else pd.DataFrame()
        hard_gap = r["gaps"][r["gaps"]["问题"] == "是"] if not r["gaps"].empty else pd.DataFrame()
        cons = r["consensus"]
        print(
            f"\n[{n}] 文件 {r['n_files']}；末日主流 {cons[0]} 占 {cons[2]}%（{cons[1]} 只）" if cons else f"\n[{n}] 文件 {r['n_files']}")
        print(f"  硬错误条目: {len(r['issues'])}")
        if not r["issues"].empty:
            print(r["issues"]["类别"].value_counts().to_string())
        print(f"  超板跳变: 疑似错位 {len(misalign)} | 真实无涨跌幅事件 {len(real)} | 冻结 {len(r['frozen'])} | "
              f"串行强指纹票对 {len(r['serial'])} | 数据缺口 {len(hard_gap)} | 缺票 {len(r['missing'])}")
        if not real.empty:
            print("  真实无涨跌幅事件（复牌/重组，非数据问题）: " +
                  ", ".join(f"{x.code} {x.date}" for x in real.itertuples()))
        if not batch.empty:
            print(f"  ⚠ pct_chg 批量缺失 {len(batch)} 个交易日，需补该字段，最早 "
                  f"{batch['日期'].min()} 最晚 {batch['日期'].max()}，单日最多 {batch['缺失票只数'].max()} 只")
        if not r["gaps"].empty:
            print("  末日滞后/停牌/缺口明细:")
            print(r["gaps"].to_string(index=False))

    out = OUT_DIR / "data_check_qc.xlsx"
    CAP = 1_000_000

    def safe(d, cols=None):
        if d is None or d.empty:
            return pd.DataFrame(columns=cols)
        return d.head(CAP)

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        safe(idf, ["板块", "code", "类别", "detail"]).to_excel(w, sheet_name="硬错误", index=False)
        safe(jdf, ["板块", "code"]).to_excel(w, sheet_name="超板跳变明细", index=False)
        safe(sdf, ["板块", "股票A", "股票B"]).to_excel(w, sheet_name="串行强指纹", index=False)
        safe(bdf, ["板块", "日期"]).to_excel(w, sheet_name="pctchg批量缺失", index=False)
        safe(gdf, ["板块", "code"]).to_excel(w, sheet_name="末日缺口停牌", index=False)
        safe(fdf, ["板块", "code"]).to_excel(w, sheet_name="价格冻结", index=False)
        safe(mdf, ["板块", "code", "name"]).to_excel(w, sheet_name="缺票清单", index=False)
        safe(rdf, ["板块", "code"]).to_excel(w, sheet_name="逐文件汇总", index=False)
    print(f"\n报告已落盘: {out}")

    n_hard = len(idf)
    print("=" * 72)
    if n_hard:
        print(f"⚠ 发现 {n_hard} 条硬错误/需补数据信号，请查看 Excel“硬错误”“pctchg批量缺失”页。")
        print("提示：微复权因子偏差（<涨跌停）离线无法判定，需另跑联网深度重建 "
              "qfq_mainboard_scan.py / chinext_star/qfq_full_rebuild_scan.py")
        return 1
    print("✓ 未发现硬错误：结构、超板跳变（除真实无涨跌幅事件）、串行强指纹、数据缺口、pct_chg 批量缺失均通过。")
    print("提示：停牌/真实无涨跌幅日已自动排除；微复权偏差需周期性跑联网深度重建确认。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
