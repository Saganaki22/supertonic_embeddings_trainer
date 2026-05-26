import json
import numpy as np
from pathlib import Path

REF = Path(__file__).parent / "reference_styles"
voices = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]

styles = {}
for v in voices:
    with open(REF / f"{v}.json") as f:
        styles[v] = json.load(f)

print("=" * 60)
print("DIMENSIONS & STRUCTURE")
print("=" * 60)
for v in ["F1", "M1"]:
    for key in ("style_ttl", "style_dp"):
        d = styles[v][key]
        arr = np.array(d["data"])
        print(f"  {v} {key}:")
        print(f"    dims:          {d['dims']}")
        print(f"    data type:     list of {type(d['data']).__name__}, depth={type(d['data'][0]).__name__}")
        print(f"    flat numel:    {arr.size}")
        print(f"    actual shape:  {arr.shape}")
        print()

print("=" * 60)
print("PER-VOICE STATS")
print("=" * 60)
for v in voices:
    for key in ("style_ttl", "style_dp"):
        arr = np.array(styles[v][key]["data"]).flatten()
        print(
            f"  {v:3s} {key:10s}: "
            f"min={arr.min():+.4f}  max={arr.max():+.4f}  "
            f"mean={arr.mean():+.4f}  std={arr.std():.4f}  "
            f"norm={np.linalg.norm(arr):.2f}"
        )

print()
print("=" * 60)
print("CROSS-VOICE COSINE SIMILARITY")
print("=" * 60)
for key in ("style_ttl", "style_dp"):
    print(f"\n--- {key} ---")
    vecs = {}
    for v in voices:
        vecs[v] = np.array(styles[v][key]["data"]).flatten()
    header = "      " + "  ".join(f"{v:>6s}" for v in voices)
    print(header)
    for a in voices:
        vals = []
        for b in voices:
            va, vb = vecs[a], vecs[b]
            cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10)
            vals.append(f"{cos:6.3f}")
        print(f"  {a:3s}  " + "  ".join(vals))

print()
print("=" * 60)
print("GENDER GROUP ANALYSIS (F avg vs M avg)")
print("=" * 60)
for key in ("style_ttl", "style_dp"):
    f_vecs = [np.array(styles[v][key]["data"]).flatten() for v in ["F1", "F2", "F3", "F4", "F5"]]
    m_vecs = [np.array(styles[v][key]["data"]).flatten() for v in ["M1", "M2", "M3", "M4", "M5"]]
    f_avg = np.mean(f_vecs, axis=0)
    m_avg = np.mean(m_vecs, axis=0)
    cos = np.dot(f_avg, m_avg) / (np.linalg.norm(f_avg) * np.linalg.norm(m_avg) + 1e-10)
    l2 = np.linalg.norm(f_avg - m_avg)
    print(f"\n--- {key} ---")
    print(f"  F_avg norm: {np.linalg.norm(f_avg):.4f}  M_avg norm: {np.linalg.norm(m_avg):.4f}")
    print(f"  F vs M cos: {cos:.6f}")
    print(f"  F vs M L2:  {l2:.4f}")
    diff = np.abs(f_avg - m_avg)
    top_dims = np.argsort(diff)[::-1][:10]
    print(f"  Top-10 differing dimensions: {top_dims.tolist()}")
    print(f"  Top-10 diff magnitudes:       {[f'{diff[d]:.4f}' for d in top_dims]}")

print()
print("=" * 60)
print("PER-DIMENSION VARIANCE (how many dims carry identity)")
print("=" * 60)
for key in ("style_ttl", "style_dp"):
    all_vecs = np.array([np.array(styles[v][key]["data"]).flatten() for v in voices])
    per_dim_var = np.var(all_vecs, axis=0)
    total_var = np.sum(per_dim_var)
    sorted_var = np.sort(per_dim_var)[::-1]
    cumvar = np.cumsum(sorted_var) / total_var
    n90 = np.searchsorted(cumvar, 0.90) + 1
    n95 = np.searchsorted(cumvar, 0.95) + 1
    n99 = np.searchsorted(cumvar, 0.99) + 1
    print(f"\n--- {key} ---")
    print(f"  Total dims:      {per_dim_var.size}")
    print(f"  Dims for 90% var: {n90} ({n90/per_dim_var.size*100:.1f}%)")
    print(f"  Dims for 95% var: {n95} ({n95/per_dim_var.size*100:.1f}%)")
    print(f"  Dims for 99% var: {n99} ({n99/per_dim_var.size*100:.1f}%)")
    print(f"  Zero-var dims:    {np.sum(per_dim_var < 1e-10)}")
    print(f"  Low-var dims (<1e-6): {np.sum(per_dim_var < 1e-6)}")
