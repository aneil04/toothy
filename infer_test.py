# example_infer.py
from inference_brushing import (
    load_scaler, build_model, predict_one, DEFAULT_FEATURE_COLS
)

# Paths from your training script output
MODEL_PATH = "toothy_mlp.pt"
SCALER_PATH = "toothy_scaler.json"

# Load scaler + model (make sure hidden sizes & dropout match training)
mean, scale = load_scaler(SCALER_PATH)
in_dim = len(DEFAULT_FEATURE_COLS)                # + extras if you used use_relative_diffs=True (then add 10)
model = build_model(MODEL_PATH, in_dim=in_dim, hidden=(64, 32), dropout=0.10, device="cpu")

# One sample (keys must match training)
row = {
    "tb_mid_x": 12.3, "tb_mid_y": -8.1,
    "hw_x": -30.0, "hw_y": -20.0,
    "hm_x": -10.0, "hm_y": -5.0,
    "ht_x": 5.0,   "ht_y": 2.0,
    "is_smiling": 0
}

res = predict_one(
    row=row,
    model=model,
    scaler_mean=mean,
    scaler_scale=scale,
    feature_cols=DEFAULT_FEATURE_COLS,
    use_relative_diffs=False,   # set True if you trained with relative diffs ON
    device="cpu",
)

print(res)
# -> {'pred_id': 5, 'pred_name': 'open_mid_down', 'probs': [ ... 9 floats ... ]}
