"""
Train custom transformer embedding với Triplet Loss.
Dùng legal_triplets_dataset đã có sẵn.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # chạy trực tiếp được từ thư mục gốc repo

from src.custom_transformer.architecture import LegalTransformerEncoder
from src.custom_transformer.tokenizer import LegalTokenizer, build_tokenizer


CONFIG = {
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ff_dim": 512,
    "max_len": 128,
    "dropout": 0.1,
    "batch_size": 32,
    "epochs": 15,
    "lr": 3e-4,
    "triplet_margin": 0.3,
    "dataset_path": "data/triplets/legal_triplets_dataset",
    "test_dataset_path": "data/triplets/test_triplets_dataset",
    "tokenizer_path": "models/custom-tokenizer/vocab.json",
    "model_output": "models/custom-transformer/model.pt",
}


# Dataset
# ------------------------------------------------------------------

class TripletDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer: LegalTokenizer, max_len: int = 128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _encode(self, text: str) -> tuple[list[int], list[int]]:
        enc = self.tokenizer.encode_plus(text, self.max_len)
        return enc["input_ids"], enc["attention_mask"]

    def __getitem__(self, idx):
        row = self.data[idx]
        a_ids, a_mask = self._encode(row["anchor"])
        p_ids, p_mask = self._encode(row["positive"])
        n_ids, n_mask = self._encode(row["negative"])
        return (
            torch.tensor(a_ids), torch.tensor(a_mask),
            torch.tensor(p_ids), torch.tensor(p_mask),
            torch.tensor(n_ids), torch.tensor(n_mask),
        )


def train(config: dict = CONFIG):
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Tokenizer
    tok_path = config["tokenizer_path"]
    if Path(tok_path).exists():
        tokenizer = LegalTokenizer()
        tokenizer.load(tok_path)
    else:
        tokenizer = build_tokenizer()

    vocab_size = len(tokenizer)

    # Dataset
    print(f"Loading dataset from {config['dataset_path']}...")
    hf_ds = load_from_disk(config["dataset_path"])
    print(f"  {len(hf_ds)} triplets")

    dataset = TripletDataset(hf_ds, tokenizer, config["max_len"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Test dataset (nếu có)
    test_loader = None
    test_path = config.get("test_dataset_path", "")
    if test_path and Path(test_path).exists():
        hf_test = load_from_disk(test_path)
        test_dataset = TripletDataset(hf_test, tokenizer, config["max_len"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        print(f"  {len(hf_test)} test triplets loaded")

    # Model
    model = LegalTransformerEncoder(
        vocab_size=vocab_size,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_len=config["max_len"],
        dropout=config["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss & optimizer
    criterion = nn.TripletMarginLoss(margin=config["triplet_margin"], p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    # Training loop
    print("\n" + "=" * 50)
    print("Training Custom Transformer Embedding")
    print("=" * 50)

    train_log, test_log = [], []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            a_ids, a_mask, p_ids, p_mask, n_ids, n_mask = [t.to(device) for t in batch]

            anchor_emb   = model(a_ids, a_mask)
            positive_emb = model(p_ids, p_mask)
            negative_emb = model(n_ids, n_mask)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        train_log.append({"epoch": epoch, "loss": avg_loss})

        # Test loss
        test_loss_str = ""
        if test_loader:
            model.eval()
            with torch.no_grad():
                t_total = sum(
                    criterion(model(a.to(device), am.to(device)),
                              model(p.to(device), pm.to(device)),
                              model(n.to(device), nm.to(device))).item()
                    for a, am, p, pm, n, nm in test_loader
                )
            avg_test = t_total / len(test_loader)
            test_log.append({"epoch": epoch, "loss": avg_test})
            test_loss_str = f" | Test Loss: {avg_test:.4f}"

        print(f"Epoch {epoch:02d}/{config['epochs']} | Loss: {avg_loss:.4f}{test_loss_str} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Lưu loss log
    with open("training/logs/loss_log_custom.json", "w") as f:
        json.dump(train_log, f, indent=2)
    if test_log:
        with open("training/logs/loss_log_test_custom.json", "w") as f:
            json.dump(test_log, f, indent=2)
        print("Test loss log → training/logs/loss_log_test_custom.json")

    # Save
    out_path = Path(config["model_output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    # Save config cùng model để load lại
    cfg_path = out_path.parent / "config.json"
    with open(cfg_path, "w") as f:
        json.dump({**config, "vocab_size": vocab_size}, f, indent=2)

    print(f"\nModel saved → {out_path}")
    print(f"Config saved → {cfg_path}")
    return model, tokenizer


if __name__ == "__main__":
    train()
