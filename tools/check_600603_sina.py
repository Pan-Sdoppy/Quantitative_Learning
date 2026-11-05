"""用akshare新浪接口交叉验证600603"""
import akshare as ak
import pandas as pd

try:
    df = ak.stock_zh_a_daily(symbol="sh600603", start_date="20261020", end_date="20261027", adjust="qfq")
    print("新浪 600603 前复权:")
    print(df.to_string(index=False))
except Exception as e:
    print(f"新浪600603失败: {e}")

try:
    df2 = ak.stock_zh_a_daily(symbol="sh600604", start_date="20261020", end_date="20261027", adjust="qfq")
    print("\n新浪 600604 前复权:")
    print(df2.to_string(index=False))
except Exception as e:
    print(f"新浪600604失败: {e}")
