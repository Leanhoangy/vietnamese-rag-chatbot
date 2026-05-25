"""
Train Custom Transformer với per-step loss logging.
Output: loss_log_custom.json → dùng để vẽ biểu đồ thật.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

from src.custom_transformer.tokenizer import LegalTokenizer, build_tokenizer
from src.custom_transformer.architecture import LegalTransformerEncoder

CONFIG = {
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ff_dim": 512,
    "max_len": 128,
    "dropout": 0.1,
    "batch_size": 16,
    "epochs": 5,
    "lr": 3e-4,
    "triplet_margin": 0.3,
    "dataset_path": "legal_triplets_dataset",
    "tokenizer_path": "models/custom-tokenizer/vocab.json",
    "model_output": "models/custom-transformer/model.pt",
}
LOG_PATH = "loss_log_custom.json"

DEVICE = torch.device("cpu")


class TripletDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _encode(self, text):
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


# ── Load tokenizer & dataset ───────────────────────────────────────────────────
tokenizer = LegalTokenizer()
tokenizer.load(CONFIG["tokenizer_path"])

hf_ds   = load_from_disk(CONFIG["dataset_path"])
dataset = TripletDataset(hf_ds, tokenizer, CONFIG["max_len"])
loader  = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

steps_per_epoch = len(loader)
total_steps     = steps_per_epoch * CONFIG["epochs"]
print(f"Dataset: {len(hf_ds)} triplets | Steps/epoch: {steps_per_epoch} | Total: {total_steps}")

# ── Model ──────────────────────────────────────────────────────────────────────
model = LegalTransformerEncoder(
    vocab_size=len(tokenizer),
    hidden_dim=CONFIG["hidden_dim"],
    num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_layers"],
    ff_dim=CONFIG["ff_dim"],
    max_len=CONFIG["max_len"],
    dropout=CONFIG["dropout"],
).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.TripletMarginLoss(margin=CONFIG["triplet_margin"], p=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

# ── Training loop với per-step logging ────────────────────────────────────────
loss_log    = []
global_step = 0

print(f"\n=== Training Custom Transformer với loss logging ===")

for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    epoch_loss = 0.0

    for batch in loader:
        a_ids, a_mask, p_ids, p_mask, n_ids, n_mask = [t.to(DEVICE) for t in batch]

        a_emb = model(a_ids, a_mask)
        p_emb = model(p_ids, p_mask)
        n_emb = model(n_ids, n_mask)

        loss = criterion(a_emb, p_emb, n_emb)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1
        loss_val = loss.item()
        epoch_loss += loss_val

        loss_log.append({"step": global_step, "epoch": epoch, "loss": round(loss_val, 6)})

        if global_step % 10 == 0 or global_step == 1:
            print(f"  Epoch {epoch}/{CONFIG['epochs']} | Step {global_step:3d}/{total_steps} | Loss: {loss_val:.4f}")

    scheduler.step()
    avg = epoch_loss / len(loader)
    print(f"  --- Epoch {epoch} done | Avg Loss: {avg:.4f} ---")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = Path(CONFIG["model_output"])
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out_path)

with open(LOG_PATH, "w") as f:
    json.dump(loss_log, f, indent=2)

print(f"\n✓ Loss log saved: {LOG_PATH} ({len(loss_log)} entries)")
print(f"✓ Model saved: {CONFIG['model_output']}")
