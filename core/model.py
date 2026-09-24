"""
模型模块：LightGBM(GPU) + XGBoost + MLP(PyTorch GPU) 集成
- 1天持仓，动态卖出：尾盘收盘卖(1d_close)
- 支持断点续训、早停、GPU自动检测
- 置信度 = 横截面排名百分位 [0, 1]
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import MODEL_DIR, get_logger, detect_gpu

warnings.filterwarnings("ignore")
logger = get_logger("model")

HORIZONS = ["1d_close"]
GPU_INFO = detect_gpu()


# ============================================================
# LightGBM 模型
# ============================================================
class LGBModel:
    """LightGBM 回归模型，GPU加速"""

    def __init__(self, horizon: int, use_gpu: bool = None):
        self.horizon = horizon
        self.use_gpu = use_gpu if use_gpu is not None else GPU_INFO["lightgbm_gpu"]
        self.model = None
        self.feature_names = None

    def _get_params(self):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 100,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": -1,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        if self.use_gpu:
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0
            logger.info(f"LightGBM 使用 GPU 加速 ({self.horizon})")
        else:
            logger.info(f"LightGBM 使用 CPU ({self.horizon})")
        return params

    def train(self, X_train, y_train, X_val=None, y_val=None, num_rounds=2000):
        import lightgbm as lgb
        import gc
        # 转float32减少内存占用
        X_train_f32 = np.asarray(X_train, dtype=np.float32)
        y_train_f32 = np.asarray(y_train, dtype=np.float32)
        train_data = lgb.Dataset(X_train_f32, label=y_train_f32)
        del X_train_f32, y_train_f32
        valid_sets = [train_data]
        valid_names = ["train"]
        val_data = None
        if X_val is not None and y_val is not None:
            X_val_f32 = np.asarray(X_val, dtype=np.float32)
            y_val_f32 = np.asarray(y_val, dtype=np.float32)
            val_data = lgb.Dataset(X_val_f32, label=y_val_f32, reference=train_data)
            del X_val_f32, y_val_f32
            valid_sets.append(val_data)
            valid_names.append("val")
        gc.collect()

        callbacks = [lgb.log_evaluation(period=200)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=True))

        # 尝试GPU训练，失败自动回退CPU
        try:
            self.model = lgb.train(
                self._get_params(),
                train_data,
                num_boost_round=num_rounds,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
        except Exception as e:
            if self.use_gpu:
                logger.warning(f"LightGBM GPU训练失败({e})，回退CPU模式")
                self.use_gpu = False
                self.model = lgb.train(
                    self._get_params(),
                    train_data,
                    num_boost_round=num_rounds,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks,
                )
            else:
                raise
        finally:
            # 训练完成后释放Dataset内存
            del train_data, valid_sets, valid_names, callbacks
            if val_data is not None:
                del val_data
            gc.collect()
        logger.info(
            f"LightGBM 训练完成，最佳迭代: {self.model.best_iteration}, device={'gpu' if self.use_gpu else 'cpu'}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        meta = {"feature_names": self.feature_names, "horizon": self.horizon,
                "use_gpu": self.use_gpu, "best_iteration": self.model.best_iteration}
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: Path):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=str(path))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names")
            self.horizon = meta.get("horizon", self.horizon)
        return self

        return self


# ============================================================
# XGBoost 模型
# ============================================================
class XGBModel:
    """XGBoost 回归模型，GPU加速"""

    def __init__(self, horizon: int, use_gpu: bool = None):
        self.horizon = horizon
        self.use_gpu = use_gpu if use_gpu is not None else GPU_INFO.get("torch_cuda", False)
        self.model = None
        self.feature_names = None

    def _get_params(self):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_jobs": -1,
            "seed": 42,
        }
        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
            logger.info(f"XGBoost 使用 GPU 加速 ({self.horizon})")
        else:
            params["tree_method"] = "hist"
            logger.info(f"XGBoost 使用 CPU ({self.horizon})")
        return params

    def train(self, X_train, y_train, X_val=None, y_val=None, num_rounds=2000):
        import xgboost as xgb
        import gc
        # 转float32减少内存占用
        X_train_f32 = np.asarray(X_train, dtype=np.float32)
        y_train_f32 = np.asarray(y_train, dtype=np.float32)
        dtrain = xgb.DMatrix(X_train_f32, label=y_train_f32)
        del X_train_f32, y_train_f32
        evals = [(dtrain, "train")]
        dval = None
        if X_val is not None and y_val is not None:
            X_val_f32 = np.asarray(X_val, dtype=np.float32)
            y_val_f32 = np.asarray(y_val, dtype=np.float32)
            dval = xgb.DMatrix(X_val_f32, label=y_val_f32)
            del X_val_f32, y_val_f32
            evals.append((dval, "val"))
        gc.collect()
        try:
            self.model = xgb.train(
                self._get_params(), dtrain,
                num_boost_round=num_rounds, evals=evals,
                early_stopping_rounds=100, verbose_eval=200,
            )
        except Exception as e:
            if self.use_gpu:
                logger.warning(f"XGBoost GPU训练失败({e})，回退CPU模式")
                self.use_gpu = False
                self.model = xgb.train(
                    self._get_params(), dtrain,
                    num_boost_round=num_rounds, evals=evals,
                    early_stopping_rounds=100, verbose_eval=200,
                )
            else:
                raise
        finally:
            # 训练完成后释放DMatrix内存
            del dtrain, evals
            if dval is not None:
                del dval
            gc.collect()
        logger.info(
            f"XGBoost 训练完成，最佳迭代: {self.model.best_iteration}, device={'gpu' if self.use_gpu else 'cpu'}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型未训练")
        import xgboost as xgb
        return self.model.predict(xgb.DMatrix(X), iteration_range=(0, self.model.best_iteration + 1))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        meta = {"feature_names": self.feature_names, "horizon": self.horizon,
                "use_gpu": self.use_gpu, "best_iteration": self.model.best_iteration}
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: Path):
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names")
            self.horizon = meta.get("horizon", self.horizon)
        return self


# ============================================================
# MLP 模型 (PyTorch)
# ============================================================
class MLPModel:
    """多层感知机回归模型，PyTorch GPU加速"""

    def __init__(self, horizon: int, input_dim: int, use_gpu: bool = None):
        self.horizon = horizon
        self.input_dim = input_dim
        self.use_gpu = use_gpu if use_gpu is not None else (GPU_INFO["torch_device"] == "cuda")
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None
        self.best_epoch = None  # 最佳epoch（早停时记录，用于100%数据训练继承）

    def _build_model(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.SiLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return Net(self.input_dim)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=80, batch_size=4096, lr=1e-3):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        import gc
        import random

        # 固定全部随机种子，保证MLP训练可复现
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 确保float32，避免float64内存爆炸
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train.values if hasattr(y_train, "values") else y_train, dtype=np.float32)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val.values if hasattr(y_val, "values") else y_val, dtype=np.float32)

        # 标准化（原地修改，避免额外大数组）
        self.scaler_mean = np.nanmean(X_train, axis=0).astype(np.float32)
        self.scaler_std = (np.nanstd(X_train, axis=0) + 1e-8).astype(np.float32)
        X_train -= self.scaler_mean
        X_train /= self.scaler_std
        if X_val is not None:
            X_val -= self.scaler_mean
            X_val /= self.scaler_std

        # 转为tensor（移到GPU后删除numpy数组释放内存）
        n_samples = len(X_train)
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)
        del X_train, y_train
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
            del X_val, y_val
        else:
            X_val_t = None
            y_val_t = None
        gc.collect()
        torch.cuda.empty_cache() if self.device == "cuda" else None
        dataset = TensorDataset(X_t, y_t)
        _g = torch.Generator()
        _g.manual_seed(42)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=_g)

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.HuberLoss(delta=0.05)

        best_val_loss = float("inf")
        best_state = None
        patience = 15
        no_improve = 0

        logger.info(f"MLP 训练开始: device={self.device}, 样本={n_samples}, epochs={epochs}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            # 验证
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_v = self.model(X_val_t).cpu().numpy()
                    y_v = y_val_t.cpu().numpy()
                    val_loss = np.mean((pred_v - y_v) ** 2)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = epoch + 1
                    no_improve = 0
                else:
                    no_improve += 1
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch + 1}/{epochs} train_loss={total_loss / len(loader):.6f} "
                                f"val_mse={val_loss:.6f} best={best_val_loss:.6f}")
                if no_improve >= patience:
                    logger.info(f"  早停于 epoch {epoch + 1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch + 1}/{epochs} loss={total_loss / len(loader):.6f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        # 训练完成后释放GPU数据
        del X_t, y_t, dataset, loader
        if X_val_t is not None:
            del X_val_t, y_val_t
        gc.collect()
        torch.cuda.empty_cache() if self.device == "cuda" else None
        logger.info(f"MLP 训练完成 ({self.horizon})")
        return self

    def predict(self, X):
        import torch
        if self.model is None:
            raise ValueError("模型未训练")
        X = np.asarray(X, dtype=np.float32)
        X_s = (X - self.scaler_mean) / self.scaler_std
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_s).to(self.device)
            pred = self.model(X_t).cpu().numpy()
        return pred

    def save(self, path: Path):
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "horizon": self.horizon,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "best_epoch": self.best_epoch,
        }
        torch.save(state, str(path))

    def load(self, path: Path):
        import torch
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        self.input_dim = state["input_dim"]
        self.horizon = state.get("horizon", self.horizon)
        self.feature_names = state.get("feature_names")
        self.scaler_mean = state["scaler_mean"]
        self.scaler_std = state["scaler_std"]
        self.best_epoch = state.get("best_epoch", None)
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(state["model_state"])
        return self


# ============================================================
# 集成模型
# ============================================================
class EnsembleModel:
    """LightGBM + XGBoost + MLP 三模型加权集成"""

    def __init__(self, horizon: int, lgb_weight: float = 0.35,
                 xgb_weight: float = 0.35, mlp_weight: float = 0.30):
        self.horizon = horizon
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.mlp_weight = mlp_weight
        self.lgb = LGBModel(horizon)
        self.xgb = XGBModel(horizon)
        self.mlp = None
        self.feature_names = None

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None,
              fixed_lgb_rounds=None, fixed_xgb_rounds=None, fixed_mlp_epochs=None):
        """训练集成模型
        fixed_lgb_rounds/xgb_rounds/mlp_epochs: 固定迭代次数（从回测模型继承），为None则用早停
        """
        import gc
        if feature_names is not None:
            self.feature_names = [c for c in feature_names if c is not None]
        elif hasattr(X_train, "columns"):
            self.feature_names = [c for c in list(X_train.columns) if c is not None]
        else:
            self.feature_names = None
        # 统一转float32 numpy数组，减少内存占用
        X_tr = np.asarray(X_train.values if hasattr(X_train, "values") else X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)
        X_vl = None
        y_vl = None
        if X_val is not None and y_val is not None:
            X_vl = np.asarray(X_val.values if hasattr(X_val, "values") else X_val, dtype=np.float32)
            y_vl = np.asarray(y_val, dtype=np.float32)
        gc.collect()
        logger.info(
            f"训练数据: {X_tr.shape[0]}样本, {X_tr.shape[1]}特征, float32, 内存≈{X_tr.nbytes / 1024 ** 2:.0f}MB")

        # LightGBM
        self.lgb.feature_names = self.feature_names
        lgb_rounds = fixed_lgb_rounds if fixed_lgb_rounds is not None else 2000
        self.lgb.train(X_tr, y_tr, X_vl if fixed_lgb_rounds is None else None,
                       y_vl if fixed_lgb_rounds is None else None, num_rounds=lgb_rounds)
        gc.collect()

        # XGBoost
        self.xgb.feature_names = self.feature_names
        xgb_rounds = fixed_xgb_rounds if fixed_xgb_rounds is not None else 2000
        self.xgb.train(X_tr, y_tr, X_vl if fixed_xgb_rounds is None else None,
                       y_vl if fixed_xgb_rounds is None else None, num_rounds=xgb_rounds)
        gc.collect()

        # MLP
        input_dim = X_tr.shape[1]
        self.mlp = MLPModel(self.horizon, input_dim)
        self.mlp.feature_names = self.feature_names
        mlp_epochs = fixed_mlp_epochs if fixed_mlp_epochs is not None else 80
        self.mlp.train(X_tr, y_tr, X_vl if fixed_mlp_epochs is None else None,
                       y_vl if fixed_mlp_epochs is None else None, epochs=mlp_epochs)
        gc.collect()

        # 训练完成后释放原始数据
        del X_tr, y_tr
        if X_vl is not None:
            del X_vl, y_vl
        gc.collect()

        return self

    def predict(self, X):
        # 特征兼容：旧模型训练时含被错误标准化的 high_low_ratio（已修复不再作为因子），
        # 预测时自动补0（训练时该特征标准化后均值为0，补0影响最小），避免 shape check 报错
        import numpy as np
        if hasattr(X, "columns") and self.feature_names is not None:
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                for c in missing:
                    X = X.copy()
                    X[c] = 0.0
                X = X[self.feature_names]
        elif not hasattr(X, "columns") and self.feature_names is not None:
            n_model = len(self.feature_names)
            if X.shape[1] < n_model:
                pad = np.zeros((X.shape[0], n_model - X.shape[1]), dtype=X.dtype)
                X = np.hstack([X, pad])
        X_arr = X.values if hasattr(X, "values") else X
        pred_lgb = self.lgb.predict(X_arr)
        pred_xgb = self.xgb.predict(X_arr)
        pred_mlp = self.mlp.predict(X_arr)
        return (self.lgb_weight * pred_lgb +
                self.xgb_weight * pred_xgb +
                self.mlp_weight * pred_mlp)

    def save(self, tag: str = "latest"):
        base = MODEL_DIR / tag
        base.mkdir(parents=True, exist_ok=True)
        self.lgb.save(base / f"lgb_{self.horizon}.txt")
        self.xgb.save(base / f"xgb_{self.horizon}.json")
        self.mlp.save(base / f"mlp_{self.horizon}.pt")
        meta = {
            "horizon": self.horizon,
            "lgb_weight": self.lgb_weight,
            "xgb_weight": self.xgb_weight,
            "mlp_weight": self.mlp_weight,
            "feature_names": self.feature_names,
            "saved_at": datetime.now().isoformat(),
        }
        with open(base / f"ensemble_{self.horizon}.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"集成模型已保存: {base} ({self.horizon})")

    def load(self, tag: str = "latest"):
        base = MODEL_DIR / tag
        self.lgb.load(base / f"lgb_{self.horizon}.txt")
        xgb_path = base / f"xgb_{self.horizon}.json"
        if xgb_path.exists():
            self.xgb.load(xgb_path)
        else:
            logger.warning("XGBoost模型不存在，将仅使用LGB+MLP")
            self.xgb_weight = 0
            self.lgb_weight = 0.5
            self.mlp_weight = 0.5
        meta_path = base / f"ensemble_{self.horizon}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names")
            self.lgb_weight = meta.get("lgb_weight", 0.35)
            self.xgb_weight = meta.get("xgb_weight", 0.35 if xgb_path.exists() else 0)
            self.mlp_weight = meta.get("mlp_weight", 0.30)
        input_dim = len(self.feature_names) if self.feature_names else 100
        self.mlp = MLPModel(self.horizon, input_dim)
        self.mlp.load(base / f"mlp_{self.horizon}.pt")
        logger.info(
            f"集成模型已加载: {base} ({self.horizon}, LGB={self.lgb_weight}, XGB={self.xgb_weight}, MLP={self.mlp_weight})")
        return self


def compute_confidence(predictions: np.ndarray) -> np.ndarray:
    """
    将预测收益率转为置信度 [0, 1]。
    使用横截面排名百分位，保留4位小数。
    """
    if len(predictions) == 0:
        return np.array([])
    rank = pd.Series(predictions).rank(pct=True).values
    return np.round(rank, 4)
