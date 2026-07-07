"""이미 만들어진 `data/dataset`에서 흰 배경(255)만 있는 빈 칸 PNG를 삭제하는
일회성 마이그레이션 스크립트.

`construct-dataset.py`는 이제 흰 배경만 있는 빈 칸의 PNG를 아예 만들지 않고,
`FontGlyphDataset`도 폰트 디렉터리에 실제로 존재하는 PNG만 표본으로 삼는다.
그러나 이 변경 이전에 만들어진 데이터셋에는 빈 칸이 순백 PNG로 남아 있어,
새 로더가 그 흰 칸까지 유효 글자로 잘못 포함하게 된다. 이 스크립트는 재추출
없이 기존 폰트 디렉터리를 훑어 전부 흰색(min == 255)인 PNG만 삭제해, 새
construct-dataset로 다시 만든 것과 같은 상태로 맞춘다.

파일명 규칙(`{char_idx:04d}.png`)은 그대로 두고 빈 칸 파일만 지우므로, 남은
파일 번호로 글자를 복원하는 로더 규칙은 그대로 유지된다.

실행:
    uv run python scripts/migrate-drop-blank-glyphs.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.font_dataset import SCAN_DIR

DATASET_DIR = SCAN_DIR.parent / "dataset"
INDEX_PATH = DATASET_DIR / "index.json"

# 흰 칸 판정용으로 PNG를 읽어들이는 스레드 수(construct-dataset의 사전 스캔과
# 같은 정도의 I/O이므로 넉넉히 둔다).
WORKERS = 8


def _is_blank(path: Path) -> bool:
    with Image.open(path) as image:
        return int(np.asarray(image).min()) == 255


def _clean_font_dir(font_dir: Path) -> tuple[int, int]:
    """폰트 디렉터리 하나에서 순백 PNG를 삭제하고 (검사한 수, 삭제한 수)를
    돌려준다."""

    scanned = 0
    removed = 0
    for path in font_dir.glob("*.png"):
        scanned += 1
        if _is_blank(path):
            path.unlink()
            removed += 1
    return scanned, removed


def main() -> None:
    if not INDEX_PATH.exists():
        print(f"[ERROR] {INDEX_PATH} not found - nothing to migrate.")
        return

    entries = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    font_dirs = [DATASET_DIR / entry["dir"] for entry in entries]
    n = len(font_dirs)
    print(f"Scanning {n} font(s) in {DATASET_DIR} for blank glyphs...")

    total_scanned = 0
    total_removed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for pos, (scanned, removed) in enumerate(
            pool.map(_clean_font_dir, font_dirs), start=1
        ):
            total_scanned += scanned
            total_removed += removed
            if pos == n or pos % 200 == 0:
                print(f"  processed {pos}/{n} font(s), "
                      f"{total_removed} blank glyph(s) removed so far...")

    print(f"Done: removed {total_removed} blank glyph(s) out of "
          f"{total_scanned} scanned across {n} font(s).")


if __name__ == "__main__":
    main()
