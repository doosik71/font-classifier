"""data/annotation 정보를 이용해 data/scan의 zip/jpg에서 완성형 한글
2,350자 낱글자 영상을 추출하고, 폰트 하나당 세로로 이어붙인 PNG 파일
(64 x 150,400)로 저장하는 배치 스크립트.

폰트마다 2,350개의 개별 파일을 만들면 3천종이 넘는 폰트에서 6백만 건이
넘는 파일이 생겨 저장소와 메모리 처리에 부담이 크므로, 폰트 단위로 한
파일에 글자를 모아 저장한다.

폰트 목록은 `font_classifier.font_dataset.build_font_entries()`가 만드는
알파벳(가나다)순 목록을 그대로 사용하며, 그 순서대로 앞에서부터 처리한다.
한 폰트에서 글자 추출 실패(annotation 없음, 영상 로드 실패, 또는 폰트가
그 글자를 지원하지 않아 빈 칸으로 인쇄된 경우)가 5회 이상 발생하면 품질이
낮다고 보고 해당 폰트 전체를 건너뛴다.

실행:
    uv run python scripts/construct-dataset.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.char_extract import CHAR_SIZE, extract_char_cell
from font_classifier.font_dataset import (
    HANGUL_TABLE, FontEntry, SCAN_DIR, build_font_entries,
)
from font_classifier.grid_autocorrect import GridParams

DATASET_DIR = SCAN_DIR.parent / "dataset"
INDEX_PATH = DATASET_DIR / "index.json"

# 한 폰트에서 이 개수 이상 글자 추출에 실패하면 해당 폰트를 건너뛴다.
MAX_FAILURES = 5

_zip_cache: dict[str, zipfile.ZipFile] = {}


def _get_zip(zip_name: str) -> zipfile.ZipFile:
    zip_file = _zip_cache.get(zip_name)
    if zip_file is None:
        zip_file = zipfile.ZipFile(SCAN_DIR / zip_name)
        _zip_cache[zip_name] = zip_file
    return zip_file


def _load_rotated_page_image(page: dict) -> Image.Image | None:
    """font-dataset-browser.py의 동일 함수와 같은 로직(annotation의
    zip/entry에서 영상을 읽고 저장된 회전각만큼 보정한다)."""

    try:
        zip_file = _get_zip(page["zip"])
        data = zip_file.read(page["entry"])
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"[ERROR] {page.get('font_name')}: failed to open image "
              f"{page['entry']} ({exc})")
        return None

    angle = float(page.get("rotation_deg", 0.0))
    if angle:
        fill = (255, 255, 255) if image.mode == "RGB" else 255
        image = image.rotate(angle, resample=Image.BICUBIC,
                              expand=False, fillcolor=fill)
    return image


def _build_font_image(entry: FontEntry) -> Image.Image | None:
    """entry의 2,350자를 HANGUL_TABLE 순서대로 세로로 이어붙인 흑백 영상을
    만든다. 추출 실패가 MAX_FAILURES 이상이면 즉시 중단하고 None을
    반환한다(해당 폰트는 건너뛴다). 추출에 실패한 낱칸은 흰 배경(255)
    그대로 남긴다.
    """

    canvas = Image.new("L", (CHAR_SIZE, CHAR_SIZE * len(HANGUL_TABLE)), color=255)
    page_image_cache: dict[str, Image.Image | None] = {}
    failures = 0

    for idx, char in enumerate(HANGUL_TABLE):
        page = entry.char_pages.get(idx)
        glyph = None

        if page is None:
            failures += 1
            print(f"[WARNING] {entry.font_name}: no annotation for "
                  f"'{char}' (idx={idx}).")
        else:
            image_name = page["image_name"]
            if image_name not in page_image_cache:
                page_image_cache[image_name] = _load_rotated_page_image(page)
            image = page_image_cache[image_name]

            if image is None:
                failures += 1
            else:
                params = GridParams(**page["grid"])
                local_row, local_col = divmod(
                    idx - page["first_char_index"], params.cols)
                glyph = extract_char_cell(image, params, local_row, local_col)
                if glyph is None:
                    failures += 1
                    print(f"[WARNING] {entry.font_name}: '{char}' "
                          f"(idx={idx}) looks like a blank cell.")

        if failures >= MAX_FAILURES:
            print(f"[SKIP] Font '{entry.font_name}': {failures} char(s) "
                  "failed to extract - skipping this font.")
            return None

        if glyph is not None:
            canvas.paste(glyph, (0, idx * CHAR_SIZE))

    return canvas


def _write_index(index: list[dict]) -> None:
    INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    entries = build_font_entries()
    print(f"Found {len(entries)} font(s) with annotation.")

    index: list[dict] = []
    next_id = 1

    for position, entry in enumerate(entries, start=1):
        print(f"Processing '{entry.font_name}' "
              f"({position}/{len(entries)})...")
        image = _build_font_image(entry)
        if image is None:
            continue

        file_name = f"{next_id:04d}.png"
        image.save(DATASET_DIR / file_name)
        index.append({
            "id": next_id,
            "font_name": entry.font_name,
            "file": file_name,
        })
        next_id += 1
        _write_index(index)  # 중간에 중단되어도 지금까지 결과를 보존한다.

    print(f"Done: {len(index)} font(s) written to {DATASET_DIR}")


if __name__ == "__main__":
    main()
