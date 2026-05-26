import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path


class MappingNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_pairs(data_dir: str):
    pairs = []
    data_path = Path(data_dir)
    for ext in ("*.npy", "*.pt"):
        for f in data_path.glob(ext):
            name = f.stem
            style_path = data_path / f"{name}.style.json"
            if style_path.exists():
                pairs.append((str(f), str(style_path)))
    return pairs


def build_training_dataset(pairs: list[tuple[str, str]], key: str = "style_ttl"):
    src_list, tgt_list = [], []
    for emb_path, style_path in pairs:
        if emb_path.endswith(".npy"):
            emb = np.load(emb_path)
        else:
            emb = torch.load(emb_path, weights_only=True).numpy()
        with open(style_path) as f:
            style = json.load(f)
        tgt = np.array(style[key]["data"], dtype=np.float32)
        src_list.append(emb.flatten())
        tgt_list.append(tgt.flatten())
    return np.array(src_list, dtype=np.float32), np.array(tgt_list, dtype=np.float32)


def train_mapping(
    data_dir: str,
    key: str = "style_ttl",
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    save_path: str = "mapping_ttl.pt",
):
    pairs = collect_pairs(data_dir)
    if not pairs:
        print(f"No training pairs found in {data_dir}")
        print("Expected: <name>.npy or <name>.pt (source embedding) + <name>.style.json (target)")
        return

    print(f"Found {len(pairs)} training pairs")
    src, tgt = build_training_dataset(pairs, key)
    print(f"Source shape: {src.shape}, Target shape: {tgt.shape}")

    model = MappingNetwork(src.shape[1], tgt.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    src_t = torch.from_numpy(src)
    tgt_t = torch.from_numpy(tgt)

    for epoch in range(epochs):
        model.train()
        pred = model(src_t)
        loss = criterion(pred, tgt_t)
        optimizer.zero_grad()
        loss.item()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.6f}")

    torch.save({
        "model": model.state_dict(),
        "input_dim": src.shape[1],
        "output_dim": tgt.shape[1],
        "hidden_dim": hidden_dim,
        "key": key,
    }, save_path)
    print(f"Saved mapping network: {save_path}")


def apply_mapping(model_path: str, embedding_path: str, output_path: str | None = None):
    ckpt = torch.load(model_path, weights_only=False)
    model = MappingNetwork(ckpt["input_dim"], ckpt["output_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    if embedding_path.endswith(".npy"):
        emb = np.load(embedding_path)
    else:
        emb = torch.load(embedding_path, weights_only=True).numpy()

    with torch.no_grad():
        mapped = model(torch.from_numpy(emb.flatten().unsqueeze(0)))

    key = ckpt.get("key", "style_ttl")
    style = {
        key: {
            "dims": [1, mapped.shape[-1]],
            "data": mapped.squeeze().numpy().tolist(),
        }
    }

    out = output_path or str(Path(embedding_path).with_suffix(f".mapped_{key}.json"))
    with open(out, "w") as f:
        json.dump(style, f)
    print(f"Mapped embedding saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Mapping network for latent space conversion")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train mapping network")
    p_train.add_argument("data_dir", help="Directory with paired embeddings + style JSONs")
    p_train.add_argument("--key", choices=["style_ttl", "style_dp"], default="style_ttl")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--hidden-dim", type=int, default=512)
    p_train.add_argument("-o", "--output", default=None)

    p_apply = sub.add_parser("apply", help="Apply trained mapping to an embedding")
    p_apply.add_argument("model_path", help="Trained mapping .pt file")
    p_apply.add_argument("embedding", help="Source embedding .npy or .pt")
    p_apply.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        save_path = args.output or f"mapping_{args.key}.pt"
        train_mapping(args.data_dir, args.key, args.epochs, args.lr, args.hidden_dim, save_path)
    elif args.cmd == "apply":
        apply_mapping(args.model_path, args.embedding, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
