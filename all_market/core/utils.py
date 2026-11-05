"""
通用工具模块：日志、路径、GPU检测、异常捕获
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============ 路径配置 ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, RESULT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "quant") -> logging.Logger:
    """获取带文件+控制台输出的logger"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_DIR / f"{name}_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def detect_gpu() -> dict:
    """
    检测可用GPU设备，返回设备信息字典。
    LightGBM GPU在部分Windows环境下会原生崩溃，默认关闭（CPU训练此数据量也很快）。
    MLP使用PyTorch GPU，稳定可靠。
    """
    info = {"lightgbm_gpu": False, "torch_device": "cpu", "torch_gpu_name": None, "torch_gpu_mem_gb": 0}

    # PyTorch GPU 检测（可靠，不会崩溃）
    try:
        import torch
        if torch.cuda.is_available():
            info["torch_device"] = "cuda"
            info["torch_gpu_name"] = torch.cuda.get_device_name(0)
            info["torch_gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass

    return info


def safe_div(a, b):
    """安全除法，除零返回0；保持pandas Series类型"""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        a = pd.Series(a) if not isinstance(a, pd.Series) else a
        b = pd.Series(b) if not isinstance(b, pd.Series) else b
        return a.divide(b, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0)
    return np.where(b == 0, 0, np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0))
