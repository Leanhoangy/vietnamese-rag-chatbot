"""
Pipeline train custom transformer embedding từ đầu.
Bước 1: Chuẩn bị data (nếu chưa có)
Bước 2: Augment triplets (nếu chưa có)
Bước 3: Build tokenizer
Bước 4: Train custom transformer
"""

import sys
from pathlib import Path


def prepare_data():
    print("\n" + "=" * 60)
    print("BƯỚC 1: Chuẩn bị training data")
    print("=" * 60)

    if Path("legal_triplets_dataset").exists():
        from datasets import load_from_disk
        ds = load_from_disk("legal_triplets_dataset")
        print(f"✓ Dataset đã có — {len(ds)} triplets. Bỏ qua.")
        return True

    from src.data_preparation import LegalDataPreparator
    preparator = LegalDataPreparator(test_set_path="test_set.json", k_hard_negatives=3)
    dataset = preparator.prepare_all(save_to_disk=True)
    if not dataset:
        print("✗ Tạo dataset thất bại")
        return False
    print(f"✓ Đã tạo {len(dataset)} triplets")
    return True


def augment_data():
    print("\n" + "=" * 60)
    print("BƯỚC 2: Augment triplets")
    print("=" * 60)

    from datasets import load_from_disk
    ds = load_from_disk("legal_triplets_dataset")

    # Nếu đã augment rồi (nhiều hơn 300 triplets) thì bỏ qua
    if len(ds) > 300:
        print(f"✓ Dataset đã augment ({len(ds)} triplets). Bỏ qua.")
        return True

    from src.augment_triplets import augment_dataset
    augment_dataset()
    return True


def build_tokenizer():
    print("\n" + "=" * 60)
    print("BƯỚC 3: Build tokenizer")
    print("=" * 60)

    if Path("models/custom-tokenizer/vocab.json").exists():
        print("✓ Tokenizer đã có. Bỏ qua.")
        return True

    from src.custom_transformer.tokenizer import build_tokenizer as _build
    _build()
    return True


def train_transformer():
    print("\n" + "=" * 60)
    print("BƯỚC 4: Train custom transformer")
    print("=" * 60)

    from src.custom_transformer.trainer import train
    train()
    return True


def main():
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Custom Transformer - Training Pipeline" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")

    if not prepare_data():
        sys.exit(1)

    if not augment_data():
        sys.exit(1)

    if not build_tokenizer():
        sys.exit(1)

    if not train_transformer():
        sys.exit(1)

    print("\n✓ Hoàn thành!")
    print("  - Model: models/custom-transformer/model.pt")
    print("  - Tokenizer: models/custom-tokenizer/vocab.json")
    print("  - Chạy evaluation: python evaluate.py")


if __name__ == "__main__":
    main()
