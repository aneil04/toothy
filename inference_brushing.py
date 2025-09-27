# inference_brushing.py
import json
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ---------- Label mapping ----------
ID2NAME = {
    1: "closed_left",
    2: "closed_mid",
    3: "closed_right",
    4: "open_left_down",
    5: "open_mid_down",
    6: "open_right_down",
    7: "open_left_up",
    8: "open_mid_up",
    9: "open_right_up",
}


# ---------- Minimal MLP (must match training) ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], num_classes: int = 9, dropout: float = 0.10):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------- Featurization (same as simplified training) ----------
DEFAULT_FEATURE_COLS = (
    "tb_mid_x","tb_mid_y",
    "hw_x","hw_y",
    "hm_x","hm_y",
    "ht_x","ht_y",
    "is_smiling",
)

def featurize_row(
    r: Dict[str, Union[int, float]],
    feature_cols: Tuple[str, ...] = DEFAULT_FEATURE_COLS,
    use_relative_diffs: bool = False,
) -> List[float]:
    tbx, tby = r["tb_mid_x"], r["tb_mid_y"]
    hwx, hwy = r["hw_x"], r["hw_y"]
    hmx, hmy = r["hm_x"], r["hm_y"]
    htx, hty = r["ht_x"], r["ht_y"]
    smile    = r["is_smiling"]

    feats = [
        tbx, tby,
        hwx, hwy,
        hmx, hmy,
        htx, hty,
        float(smile),
    ]

    if use_relative_diffs:
        feats += [
            tbx - hwx, tby - hwy,
            tbx - hmx, tby - hmy,
            tbx - htx, tby - hty,
            hmx - hwx, hmy - hwy,   # wrist->middle
            htx - hmx, hty - hmy,   # middle->tips
        ]
    return feats


# ---------- Loader helpers ----------
def load_scaler(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads StandardScaler params saved by training:
    {"mean": [...], "scale": [...]}
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    mean = np.array(data["mean"], dtype=np.float32)
    scale = np.array(data["scale"], dtype=np.float32)
    return mean, scale


def build_model(
    model_path: str,
    in_dim: int,
    hidden: Tuple[int, ...] = (64, 32),
    dropout: float = 0.10,
    device: str = "cpu",
) -> MLP:
    model = MLP(in_dim=in_dim, hidden=hidden, dropout=dropout)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------- Inference APIs ----------
@torch.no_grad()
def predict_one(
    row: Dict[str, Union[int, float]],
    model: MLP,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    feature_cols: Tuple[str, ...] = DEFAULT_FEATURE_COLS,
    use_relative_diffs: bool = False,
    device: str = "cpu",
) -> Dict[str, Union[int, str, List[float]]]:
    """
    Returns:
      {
        "pred_id": int in [1..9],
        "pred_name": str,
        "probs": List[float] length 9  (softmax probabilities in class-id order 1..9)
      }
    """
    x = np.array(featurize_row(row, feature_cols, use_relative_diffs), dtype=np.float32)
    # Standardize (same StandardScaler params used at training time)
    x = (x - scaler_mean) / scaler_scale
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    logits = model(x_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred0 = int(probs.argmax())             # 0..8
    pred_id = pred0 + 1                     # 1..9
    return {
        "pred_id": pred_id,
        "pred_name": ID2NAME[pred_id],
        "probs": probs.tolist(),
    }


@torch.no_grad()
def predict_batch(
    rows: List[Dict[str, Union[int, float]]],
    model: MLP,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    feature_cols: Tuple[str, ...] = DEFAULT_FEATURE_COLS,
    use_relative_diffs: bool = False,
    device: str = "cpu",
) -> List[Dict[str, Union[int, str, List[float]]]]:
    mats = [featurize_row(r, feature_cols, use_relative_diffs) for r in rows]
    X = np.asarray(mats, dtype=np.float32)
    X = (X - scaler_mean) / scaler_scale
    X_t = torch.from_numpy(X).to(device)
    logits = model(X_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    out = []
    for p in probs:
        pred0 = int(p.argmax())
        pred_id = pred0 + 1
        out.append({
            "pred_id": pred_id,
            "pred_name": ID2NAME[pred_id],
            "probs": p.tolist(),
        })
    return out
