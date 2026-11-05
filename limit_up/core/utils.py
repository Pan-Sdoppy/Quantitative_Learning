"""涨停板专项研究 - 路径配置"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 复用主板数据（只读）
RAW_DIR = PROJECT_ROOT.parent / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [PROCESSED_DIR, MODEL_DIR, RESULT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 涨停阈值（主板 10%，留 0.5% 余量）
LIMIT_UP_PCT = 9.5
# 次新股上市初期无涨跌幅限制，豁免
MIN_LIST_DAYS = 60
