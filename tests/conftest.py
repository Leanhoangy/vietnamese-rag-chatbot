"""Test chunker/glossary mà không cần đọc data/raw hay tải model.

src.chunker đọc toàn bộ văn bản ở cấp module (from .loader import data), nên thay src.loader
bằng một module rỗng trước khi import.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_fake_loader = types.ModuleType("src.loader")
_fake_loader.data = []
sys.modules.setdefault("src.loader", _fake_loader)
