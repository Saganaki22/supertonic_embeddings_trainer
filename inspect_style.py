import json
import sys
import argparse
from pathlib import Path
import numpy as np


def inspect_style(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    print(f"File: {path}")
    print(f"Top-level keys: {list(data.keys())}")
    print()

    for key in data:
        entry = data[key]
        if isinstance(entry, dict) and "dims" in entry and "data" in entry:
            dims = entry["dims"]
            flat_data = np.array(entry["data"]).flatten()
            print(f"  {key}:")
            print(f"    dims:      {dims}")
            print(f"    numel:     {flat_data.size}")
            print(f"    dtype:     float32 (inferred)")
            print(f"    min:       {flat_data.min():.6f}")
            print(f"    max:       {flat_data.max():.6f}")
            print(f"    mean:      {flat_data.mean():.6f}")
            print(f"    std:       {flat_data.std():.6f}")
            print(f"    has_nan:   {np.isnan(flat_data).any()}")
            print(f"    has_inf:   {np.isinf(flat_data).any()}")
            if flat_data.size <= 20:
                print(f"    values:    {flat_data.tolist()}")
            else:
                print(f"    first_10:  {flat_data[:10].tolist()}")
            print()
        else:
            print(f"  {key}: {type(entry).__name__} = {entry}")
            print()


def compare_styles(path_a: str, path_b: str):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print()

    for key in set(list(a.keys()) + list(b.keys())):
        if key not in a:
            print(f"  {key}: MISSING in A")
            continue
        if key not in b:
            print(f"  {key}: MISSING in B")
            continue

        ea, eb = a[key], b[key]
        if not (isinstance(ea, dict) and "data" in ea):
            print(f"  {key}: {ea} vs {eb}")
            continue

        va = np.array(ea["data"]).flatten()
        vb = np.array(eb["data"]).flatten()

        if va.shape != vb.shape:
            print(f"  {key}: SHAPE MISMATCH {va.shape} vs {vb.shape}")
            continue

        diff = va - vb
        print(f"  {key}:")
        print(f"    dims:        {ea['dims']}")
        print(f"    l2_dist:     {np.linalg.norm(diff):.6f}")
        print(f"    cos_sim:     {np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10):.6f}")
        print(f"    max_abs_diff:{np.abs(diff).max():.6f}")
        print(f"    mean_diff:   {diff.mean():.6f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect Supertonic style JSON files")
    parser.add_argument("files", nargs="+", help="Path(s) to style JSON(s)")
    parser.add_argument("--compare", action="store_true", help="Compare two style files")
    args = parser.parse_args()

    if args.compare and len(args.files) >= 2:
        compare_styles(args.files[0], args.files[1])
    else:
        for p in args.files:
            inspect_style(p)
            print()


if __name__ == "__main__":
    main()
