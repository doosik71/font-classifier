"""`data/dataset`의 폰트 중심 PNG를 읽어 `data/dataset-2`의 글자 중심
chunk PNG로 재배치하는 배치 스크립트.

기존 `construct-dataset.py`는 "폰트 하나 = 세로로 2,350글자를 붙인 PNG"
구조를 만든다. 이 스크립트는 그 결과를 다시 읽어 "여러 폰트 x 연속된 몇
개 글자 = chunk PNG" 구조로 바꾼다. 이렇게 하면 같은 글자를 여러 폰트로
묶는 배치 구성이 쉬워지고, 폰트 학습에서 content(글자) 편차를 줄이는 데
유리하다.

실행:
    uv run python scripts/construct-dataset-2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.char_extract import CHAR_SIZE
from font_classifier.dataset_loader import DATASET_DIR
from font_classifier.font_dataset import HANGUL_TABLE, SCAN_DIR

DATASET2_DIR = SCAN_DIR.parent / "dataset-2"
INDEX_PATH = DATASET2_DIR / "index.json"
DEFAULT_CHUNK_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack data/dataset into char-major chunk PNGs under data/dataset-2."
    )
    parser.add_argument("--source-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--target-dir", type=Path, default=DATASET2_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def _load_font_index(dataset_dir: Path) -> list[dict]:
    index_path = dataset_dir / "index.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    ids = sorted(entry["id"] for entry in entries)
    if ids != list(range(1, len(entries) + 1)):
        raise ValueError(
            f"{index_path}: 'id' must form a contiguous 1..N range "
            f"(N={len(entries)})"
        )
    entries.sort(key=lambda entry: entry["id"])
    return entries


def _decode_font(dataset_dir: Path, entry: dict) -> np.ndarray:
    image = Image.open(dataset_dir / entry["file"])
    image.load()
    arr = np.array(image, dtype=np.uint8)
    expected_shape = (len(HANGUL_TABLE) * CHAR_SIZE, CHAR_SIZE)
    if arr.shape != expected_shape:
        raise ValueError(
            f"{entry['file']}: unexpected image shape {arr.shape}, "
            f"expected {expected_shape}"
        )
    return arr.reshape(len(HANGUL_TABLE), CHAR_SIZE, CHAR_SIZE)


def _write_index(target_dir: Path, chunk_size: int, fonts: list[dict], chunks: list[dict]) -> None:
    (target_dir / "index.json").write_text(
        json.dumps(
            {
                "format": "char_chunks_v1",
                "char_size": CHAR_SIZE,
                "chunk_size": chunk_size,
                "num_chars": len(HANGUL_TABLE),
                "fonts": [
                    {"id": entry["id"], "font_name": entry["font_name"]}
                    for entry in fonts
                ],
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    fonts = _load_font_index(args.source_dir)
    if not fonts:
        raise ValueError(
            f"No font entries found in {args.source_dir}. "
            "Run scripts/construct-dataset.py first."
        )

    target_dir = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    num_fonts = len(fonts)
    num_chars = len(HANGUL_TABLE)
    chunks: list[dict] = []

    for chunk_id, char_start in enumerate(range(0, num_chars, args.chunk_size)):
        char_end = min(num_chars, char_start + args.chunk_size) - 1
        chunk_len = char_end - char_start + 1
        print(
            f"Building chunk {chunk_id:04d} "
            f"(chars {char_start}..{char_end}, {chunk_len} glyphs per font)..."
        )

        canvas = np.full(
            (num_fonts * CHAR_SIZE, chunk_len * CHAR_SIZE),
            255,
            dtype=np.uint8,
        )

        for font_pos, entry in enumerate(fonts):
            glyphs = _decode_font(args.source_dir, entry)[char_start:char_end + 1]
            top = font_pos * CHAR_SIZE
            for offset, glyph in enumerate(glyphs):
                left = offset * CHAR_SIZE
                canvas[top:top + CHAR_SIZE, left:left + CHAR_SIZE] = glyph

        file_name = f"{chunk_id:04d}.png"
        Image.fromarray(canvas, mode="L").save(target_dir / file_name)
        chunks.append({
            "chunk_id": chunk_id,
            "char_start": char_start,
            "char_end": char_end,
            "file": file_name,
        })
        _write_index(target_dir, args.chunk_size, fonts, chunks)

    print(f"Done: {len(chunks)} chunk(s) written to {target_dir}")


if __name__ == "__main__":
    main()
