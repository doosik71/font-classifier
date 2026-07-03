"""data/annotation의 annotation 정보를 폰트 단위로 모아, 완성형 2,350자
표의 각 글자가 어느 페이지(annotation)의 어느 칸에 있는지 알려주는 순수
로직. Tkinter나 다른 GUI 상태에 의존하지 않는다.

`scripts/font-dataset-browser.py`(GUI)가 사용한다.

annotation은 계속 추가되는 중이므로, 폰트 하나(2,350자)에서 일부 글자의
annotation이 아직 없는 것은 정상이다. 다만 페이지 전체가 통째로 누락되어
`MISSING_PAGE_THRESHOLD`자 이상을 표시할 수 없다면, 그 폰트는 검증 대상
목록에서 제외한다(README/AGENTS 지침에 따라 이유를 콘솔에 출력한다).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCAN_DIR = Path(__file__).resolve().parent.parent / "data" / "scan"
ANNOTATION_DIR = SCAN_DIR.parent / "annotation"

# 폰트 하나(2,350자)에서 annotation이 없는 글자가 이 개수 이상이면 페이지
# 전체가 누락된 것으로 보고 목록에서 제외한다.
MISSING_PAGE_THRESHOLD = 100

_REQUIRED_PAGE_FIELDS = ("zip", "entry", "image_name",
                         "first_char_index", "last_char_index", "grid")


def build_hangul_table() -> list[str]:
    """KS X 1001(완성형) 한글 2,350자를 코드/인쇄 순서대로 생성한다.

    scan-font-browser.py의 같은 이름 함수와 동일한 로직이다(설명은
    docs/scan-font-browser.md 2.3절 참고). 두 스크립트가 별도 실행
    단위이므로 작은 순수 함수를 그대로 복제해 불필요한 모듈 결합을
    피했다.
    """

    table = []
    for code in range(0xAC00, 0xD7A4):
        ch = chr(code)
        try:
            ch.encode("iso2022_kr")
        except UnicodeEncodeError:
            continue
        table.append(ch)
    return table


HANGUL_TABLE = build_hangul_table()

# 초성/중성/종성(받침 없음 포함) 개수. docs/model-design.md 3.7절의 조합 공식
# `code = 0xAC00 + (cho*NUM_JUNG + jung)*NUM_JONG + jong`에서 쓰는 상수와 같다.
NUM_CHO = 19
NUM_JUNG = 21
NUM_JONG = 28


def decompose_hangul_syllable(char: str) -> tuple[int, int, int]:
    """완성형 한글 음절 하나를 (초성, 중성, 종성) 인덱스로 분해한다.

    docs/model-design.md 3.7절의 산술 공식을 그대로 따른다. 학습 라벨 생성과
    추론 디코딩이 항상 같은 공식을 쓰도록 이 한 곳에만 구현하고 재사용한다.
    """

    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):
        raise ValueError(f"Not a modern Hangul syllable: {char!r}")

    index = code - 0xAC00
    cho, remainder = divmod(index, NUM_JUNG * NUM_JONG)
    jung, jong = divmod(remainder, NUM_JONG)
    return cho, jung, jong


@dataclass
class FontEntry:
    font_name: str
    pages: list[dict]
    char_pages: dict[int, dict]
    missing_count: int

    @property
    def is_complete(self) -> bool:
        return self.missing_count == 0


def _load_all_pages() -> list[dict]:
    pages = []
    for path in sorted(ANNOTATION_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ERROR] Failed to read {path.name} ({exc})")
            continue

        if any(field not in data for field in _REQUIRED_PAGE_FIELDS):
            print(f"[ERROR] {path.name}: missing required fields, skipping.")
            continue

        pages.append(data)
    return pages


def build_font_entries() -> list[FontEntry]:
    """data/annotation을 읽어 폰트별로 묶은 `FontEntry` 목록을 만든다.

    같은 `font_name`의 페이지를 image_name(8자리 일련번호) 오름차순으로
    정렬한 뒤, 앞에서부터 순서대로 글자 인덱스를 채운다. 한 페이지가
    중복 스캔되어 같은 글자 인덱스가 여러 페이지에 걸쳐 있으면, 먼저
    나온(=image_name이 더 작은) 페이지가 그 글자를 차지한다
    (`dict.setdefault`가 첫 배정을 유지).
    """

    by_font: dict[str, list[dict]] = {}
    for page in _load_all_pages():
        by_font.setdefault(page["font_name"], []).append(page)

    entries: list[FontEntry] = []
    for font_name, pages in by_font.items():
        pages.sort(key=lambda page: page["image_name"])

        char_pages: dict[int, dict] = {}
        for page in pages:
            first_idx = page["first_char_index"]
            last_idx = page["last_char_index"]
            for idx in range(first_idx, last_idx + 1):
                char_pages.setdefault(idx, page)

        missing_count = len(HANGUL_TABLE) - len(char_pages)
        if missing_count >= MISSING_PAGE_THRESHOLD:
            print(
                f"[ERROR] Font '{font_name}': missing annotation for "
                f"{missing_count} characters (likely a whole missing page) - "
                "excluded from the list."
            )
            continue

        if missing_count > 0:
            print(
                f"[WARNING] Font '{font_name}': missing annotation for "
                f"{missing_count} characters, but still included (annotation "
                "is still being filled in)."
            )

        entries.append(
            FontEntry(
                font_name=font_name,
                pages=pages,
                char_pages=char_pages,
                missing_count=missing_count,
            )
        )

    entries.sort(key=lambda entry: entry.font_name)
    return entries
