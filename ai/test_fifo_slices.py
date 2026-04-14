import os, pickle, numpy as np
from predict_fifo_utils import prepare_fifo_features

m_path = os.path.join("retrieval", "results", "trained_models", "SplitFIFO_area_model.pkl")
with open(m_path, "rb") as f: m = pickle.load(f)

slices = [
    {"depth": 512, "ram_style": "block", "impl_style": "vivado"},
    {"depth": 256, "ram_style": "distributed", "impl_style": "rtl"},
    {"depth": 16, "ram_style": "distributed", "impl_style": "rtl"}
]

for i, s in enumerate(slices):
    feat = prepare_fifo_features(depth=s["depth"], in_width=8, ram_style=s["ram_style"], impl_style=s["impl_style"], bits=1, simd=8)
    X_arr = np.array([float(feat.get(f, 0.0)) for f in m["feature_names"]], dtype=np.float32).reshape(1, -1)
    preds = np.maximum(0, m["model"].predict(X_arr)[0])
    print(f"\nFatia {i} (depth={s['depth']}, {s['ram_style']}, {s['impl_style']}):")
    print(f"  -> LUTs: {np.round(preds[0])}")
    print(f"  -> SRLs: {np.round(preds[2])}")
    print(f"  -> FFs: {np.round(preds[3])}")
    print(f"  -> RAMB36: {np.round(preds[4] * 2) / 2}")
    print(f"  -> RAMB18: {np.round(preds[5] * 2) / 2}")
