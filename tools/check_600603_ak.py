"""用akshare交叉验证600603的10-23~10-27真实数据"""
import akshare as ak
import pandas as pd

# 600603 是上交所
df = ak.stock_zh_a_hist(symbol="600603", period="daily", start_date="20261020", end_date="20261027", adjust="qfq")
print("akshare 600603 前复权:")
print(df.to_string(index=False))
print()

df2 = ak.stock_zh_a_hist(symbol="600604", period="daily", start_date="20261020", end_date="20261027", adjust="qfq")
print("akshare 600604 前复权:")
print(df2.to_string(index=False))
