"""
Transformer Encoder nhỏ tự xây từ đầu bằng PyTorch.
Dùng để tạo sentence embeddings cho văn bản pháp lý tiếng Việt.

Kiến trúc:
    Token Embedding + Positional Encoding
    → TransformerEncoderLayer × num_layers
    → Mean Pooling (bỏ qua [PAD])
    → L2 Normalize
    → vector 128 chiều
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LegalTransformerEncoder(nn.Module):
    """
    Transformer encoder nhỏ cho sentence embedding.
    Output: vector 128 chiều đã normalize L2.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,        # (batch, seq_len)
        attention_mask: torch.Tensor,   # (batch, seq_len)  1=real, 0=pad
    ) -> torch.Tensor:
        # Embedding + positional encoding
        x = self.token_embedding(input_ids)      # (batch, seq_len, hidden)
        x = self.pos_encoding(x)

        # TransformerEncoder dùng src_key_padding_mask: True = bỏ qua
        pad_mask = attention_mask == 0           # (batch, seq_len)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Mean pooling — chỉ tính trên các token thực (không phải PAD)
        mask = attention_mask.unsqueeze(-1).float()   # (batch, seq_len, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # L2 normalize để dùng cosine similarity
        return F.normalize(x, p=2, dim=-1)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-mode wrapper (no grad)."""
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)


# ------------------------------------------------------------------
# Helper: load model đã train
# ------------------------------------------------------------------

def load_custom_model(
    model_path: str = "models/custom-transformer/model.pt",
    vocab_size: int = 8000,
) -> LegalTransformerEncoder:
    model = LegalTransformerEncoder(vocab_size=vocab_size)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Custom transformer loaded ← {model_path}")
    return model
