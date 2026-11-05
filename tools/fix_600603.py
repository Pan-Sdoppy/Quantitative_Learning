"""补拉 600603 的 2026-10-23~10-27 正确日线数据（被600604串行覆盖）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
import baostock as bs

CODE = "sh.600603"
LOCAL_CODE = "600603"
START = "2026-10-23"
END = "2026-10-27"
LOCAL_PATH = f"data/raw/{LOCAL_CODE}.parquet"

# 先看本地现状
local = pd.read_parquet(LOCAL_PATH)
print(f"本地现状: {len(local)} 行")
print("本地 10-22 之后:")
print(local[local['date'] >= '2026-10-22'][['date','open','high','low','close','volume','amount']].to_string(index=False))

# login
lg = bs.login()
print(f"baostock login: {lg.error_code} {lg.error_msg}")
try:
    # 拉前复权
    rs = bs.query_history_k_data_plus(
        CODE, "date,open,high,low,close,volume,amount,turn,pctChg",
        start_date=START, end_date=END, frequency="d", adjustflag="2"
    )
    print(f"query: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open","high","low","close","volume","amount","turn","pctChg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"turn":"turnover", "pctChg":"pct_chg"})
    df["date"] = pd.to_datetime(df["date"])
    print(f"\nbaostock 返回 {len(df)} 行:")
    print(df[['date','open','high','low','close','volume','amount','pct_chg']].to_string(index=False))
finally:
    bs.logout()
