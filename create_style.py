import json
import argparse
from pathlib import Path
import numpy as np


def load_style(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_style(style: dict, path: str):
    with open(path, "w") as f:
        json.dump(style, f)


def inject_embedding(style: dict, key: str, data: np.ndarray, dims: list[int]):
    style[key] = {
        "dims": dims,
        "data": data.flatten().tolist(),
    }
    return style


def blend_styles(style_a: dict, style_b: dict, alpha: float = 0.5) -> dict:
    result = {}
    for key in ("style_ttl", "style_dp"):
        if key not in style_a or key not in style_b:
            continue
        da = np.array(style_a[key]["data"], dtype=np.float32)
        db = np.array(style_b[key]["data"], dtype=np.float32)
        blended = (1 - alpha) * da + alpha * db
        result[key] = {
            "dims": style_a[key]["dims"],
            "data": blended.tolist(),
        }
    return result


def interpolate_style(style: dict, target_key: str, new_data: np.ndarray, blend: float = 0.5) -> dict:
    if target_key not in style:
        raise KeyError(f"{target_key} not found in style")
    original = np.array(style[target_key]["data"], dtype=np.float32)
    if original.shape != new_data.shape:
        if original.size == new_data.size:
            new_data = new_data.reshape(original.shape)
        else:
            raise ValueError(f"Shape mismatch: original={original.shape}, new={new_data.shape}")
    mixed = (1 - blend) * original + blend * new_data
    style[target_key]["data"] = mixed.tolist()
    return style


def clone_dims(source_style: dict, target_style: dict) -> dict:
    for key in ("style_ttl", "style_dp"):
        if key in source_style and key in target_style:
            src_data = np.array(source_style[key]["data"], dtype=np.float32)
            tgt_dims = target_style[key]["dims"]
            tgt_numel = int(np.prod(tgt_dims))
            if src_data.size == tgt_numel:
                target_style[key]["data"] = src_data.tolist()
            else:
                print(f"  {key}: numel mismatch src={src_data.size} vs tgt={tgt_numel}, reshaping/padding")
                flat = src_data.flatten()
                if flat.size < tgt_numel:
                    padded = np.zeros(tgt_numel, dtype=np.float32)
                    padded[:flat.size] = flat
                    target_style[key]["data"] = padded.tolist()
                else:
                    target_style[key]["data"] = flat[:tgt_numel].tolist()
            target_style[key]["dims"] = tgt_dims
    return target_style


def main():
    parser = argparse.ArgumentParser(description="Inject/modify Supertonic style JSONs")
    sub = parser.add_subparsers(dest="cmd")

    p_blend = sub.add_parser("blend", help="Blend two style JSONs")
    p_blend.add_argument("style_a")
    p_blend.add_argument("style_b")
    p_blend.add_argument("--alpha", type=float, default=0.5)
    p_blend.add_argument("-o", "--output", default="blended.style.json")

    p_inject = sub.add_parser("inject", help="Inject embedding data into a style JSON")
    p_inject.add_argument("target")
    p_inject.add_argument("--key", choices=["style_ttl", "style_dp", "both"], default="both")
    p_inject.add_argument("--source", help="Source style JSON to steal embeddings from")
    p_inject.add_argument("--blend", type=float, default=1.0, help="Blend ratio (1.0=full replace)")
    p_inject.add_argument("-o", "--output", default=None)

    p_clone = sub.add_parser("clone", help="Clone embedding dims from source into target structure")
    p_clone.add_argument("source")
    p_clone.add_argument("target")
    p_clone.add_argument("-o", "--output", default="cloned.style.json")

    args = parser.parse_args()

    if args.cmd == "blend":
        a = load_style(args.style_a)
        b = load_style(args.style_b)
        blended = blend_styles(a, b, args.alpha)
        save_style(blended, args.output)
        print(f"Blended (alpha={args.alpha}): {args.output}")

    elif args.cmd == "inject":
        target = load_style(args.target)
        if args.source:
            source = load_style(args.source)
            src_data = np.array(source[args.key]["data"] if args.key != "both" else source["style_ttl"]["data"], dtype=np.float32)
            keys = [args.key] if args.key != "both" else ["style_ttl", "style_dp"]
            for k in keys:
                data = np.array(source[k]["data"], dtype=np.float32)
                target = interpolate_style(target, k, data, args.blend)
                print(f"  Injected {k} (blend={args.blend})")
        out = args.output or args.target
        save_style(target, out)
        print(f"Saved: {out}")

    elif args.cmd == "clone":
        source = load_style(args.source)
        target = load_style(args.target)
        cloned = clone_dims(source, target)
        save_style(cloned, args.output)
        print(f"Cloned: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
