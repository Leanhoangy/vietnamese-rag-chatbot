"""
Train Fine-tuned E5 với loss logging thực tế.
Output: loss_log.json → dùng để vẽ biểu đồ thật.
Manual training loop — bypasses HuggingFace Trainer (avoids MPS auto-dispatch).
"""
import os, sys, json, random
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

DATASET_PATH = "legal_triplets_dataset"
OUTPUT_DIR   = "models/finetuned-embedder"
LOG_PATH     = "loss_log.json"
EPOCHS       = 5
BATCH_SIZE   = 16
LR           = 2e-5
MARGIN       = 0.3
DEVICE       = torch.device("cpu")

# ── Load dataset ───────────────────────────────────────────────────────────────
ds = load_from_disk(DATASET_PATH)
triplets = [(r["anchor"], r["positive"], r["negative"]) for r in ds]
steps_per_epoch = len(triplets) // BATCH_SIZE
total_steps = steps_per_epoch * EPOCHS
print(f"Dataset: {len(triplets)} triplets | Steps/epoch: {steps_per_epoch} | Total: {total_steps}")

# ── Model (force CPU) ──────────────────────────────────────────────────────────
model = SentenceTransformer("intfloat/multilingual-e5-base", device=DEVICE)
model = model.to(DEVICE)
model.train()

# Pull out the tokenizer and the underlying transformer module
tokenizer   = model.tokenizer
transformer = model[0].auto_model  # XLMRobertaModel

# ── Encode with gradients (mean-pool + L2-norm) ────────────────────────────────
def encode_grad(texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = transformer(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(emb, p=2, dim=1)

# ── Optimiser + warmup scheduler ──────────────────────────────────────────────
optimizer    = AdamW(transformer.parameters(), lr=LR)
warmup_steps = int(total_steps * 0.1)
scheduler    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

# ── Training loop ──────────────────────────────────────────────────────────────
loss_log    = []
global_step = 0

print(f"\n=== Training Fine-tuned E5 với loss logging ===")

for epoch in range(1, EPOCHS + 1):
    random.shuffle(triplets)

    for i in range(0, len(triplets) - BATCH_SIZE + 1, BATCH_SIZE):
        batch = triplets[i: i + BATCH_SIZE]
        a_emb = encode_grad([t[0] for t in batch])
        p_emb = encode_grad([t[1] for t in batch])
        n_emb = encode_grad([t[2] for t in batch])

        d_pos = F.pairwise_distance(a_emb, p_emb)
        d_neg = F.pairwise_distance(a_emb, n_emb)
        loss  = F.relu(d_pos - d_neg + MARGIN).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if global_step < warmup_steps:
            scheduler.step()

        global_step += 1
        loss_val = loss.item()
        loss_log.append({"step": global_step, "epoch": epoch, "loss": round(loss_val, 6)})

        if global_step % 10 == 0 or global_step == 1:
            print(f"  Epoch {epoch}/{EPOCHS} | Step {global_step:3d}/{total_steps} | Loss: {loss_val:.4f}")

    print(f"  --- Epoch {epoch} done ---")

# ── Save model ─────────────────────────────────────────────────────────────────
model.eval()
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save(OUTPUT_DIR)

# ── Save log ───────────────────────────────────────────────────────────────────
with open(LOG_PATH, "w") as f:
    json.dump(loss_log, f, indent=2)

print(f"\n✓ Loss log saved: {LOG_PATH} ({len(loss_log)} entries)")
print(f"✓ Model saved: {OUTPUT_DIR}")
