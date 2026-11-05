"""修复 600604 的 10-23~10-27 被错误写成 600603 的数据（新浪交叉验证）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import akshare as ak

LOCAL_PATH = "data/raw/600604.parquet"

# 新浪拉正确数据
df_new = ak.stock_zh_a_daily(symbol="sh600604", start_date="20261023", end_date="20261027", adjust="qfq")
df_new['date'] = pd.to_datetime(df_new['date'])
df_new['volume'] = df_new['volume'].astype(int)
df_new['amount'] = df_new['amount'].astype(float)

# 读本地
local = pd.read_parquet(LOCAL_PATH)
print(f"本地列: {list(local.columns)}")
print(f"本地修复前 10-22 之后:")
print(local[local['date'] >= '2026-10-22'][['date','open','high','low','close','volume','amount','pct_chg']].to_string(index=False))

# 计算 pct_chg
prev_close = local[local['date'] == '2026-10-22']['close'].iloc[0]
df_new = df_new.sort_values('date').reset_index(drop=True)
df_new['pct_chg'] = None
for i in range(len(df_new)):
    if i == 0:
        df_new.loc[i, 'pct_chg'] = (df_new.loc[i, 'close'] - prev_close) / prev_close * 100
    else:
        df_new.loc[i, 'pct_chg'] = (df_new.loc[i, 'close'] - df_new.loc[i-1, 'close']) / df_new.loc[i-1, 'close'] * 100

# 只保留本地已有的列
common_cols = [c for c in local.columns if c in df_new.columns]
common_cols = ['date'] + [c for c in common_cols if c != 'date']
print(f"共同列: {common_cols}")
df_new = df_new[common_cols]

# 备份
import shutil
shutil.copy(LOCAL_PATH, LOCAL_PATH + '.bak_600604fix')

# 替换 10-23 之后的行
local_new = local[local['date'] < '2026-10-23'].copy()
result = pd.concat([local_new, df_new], ignore_index=True)
result = result.sort_values('date').reset_index(drop=True)
result.to_parquet(LOCAL_PATH, index=False)

print(f"\n修复后 10-22 之后:")
chk = pd.read_parquet(LOCAL_PATH)
print(chk[chk['date'] >= '2026-10-22'][['date','open','high','low','close','volume','amount','pct_chg']].to_string(index=False))
print(f"总行数: {len(chk)} (原 {len(local)})")
