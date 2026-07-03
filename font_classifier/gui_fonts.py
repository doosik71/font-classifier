"""Tkinter GUI에서 한글 레이블을 그릴 때 쓸 폰트를 플랫폼 독립적으로 고르는 헬퍼.

`dataset-browser.py`, `font-dataset-browser.py` 등이 한글 글자 레이블을 그릴 때
`Malgun Gothic`(Windows 전용)을 하드코딩하면 우분투/맥에서는 해당 폰트가 없어
한글이 없는 기본 폰트로 폴백되어 글자가 깨진다. 이 모듈은 실제로 설치된
폰트 목록(`tkinter.font.families()`)에서 한글을 지원하는 폰트를 우선순위대로
찾아 반환한다.
"""

from __future__ import annotations

from tkinter import font as tkfont

# 플랫폼별 대표 한글 폰트를 우선순위대로 나열한다. 앞쪽일수록 우선.
# Windows: Malgun Gothic / macOS: Apple SD Gothic Neo, AppleGothic
# Linux: Noto Sans CJK KR, NanumGothic, Baekmuk Gulim
_PREFERRED_KOREAN_FONTS = (
    "Malgun Gothic",
    "Apple SD Gothic Neo",
    "AppleGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "NanumGothic",
    "NanumBarunGothic",
    "Baekmuk Gulim",
    "UnDotum",
)

# 한 번 확인한 결과를 재사용한다(폰트 목록 조회 비용 절감).
_cached_family: str | None = None


def korean_font_family(root=None) -> str:
    """설치된 폰트 중 한글을 지원하는 폰트 패밀리 이름을 반환한다.

    우선순위 목록에서 실제 설치된 첫 폰트를 고르고, 하나도 없으면
    Tk 기본 폰트(`TkDefaultFont`) 패밀리로 폴백한다.

    `tkinter.font`는 Tk 루트가 있어야 동작하므로 GUI를 만든 뒤에 호출해야 한다.
    `root`를 넘기면 해당 인터프리터의 폰트 목록을 사용한다.
    """
    global _cached_family
    if _cached_family is not None:
        return _cached_family

    available = set(tkfont.families(root=root)) if root is not None \
        else set(tkfont.families())

    for family in _PREFERRED_KOREAN_FONTS:
        if family in available:
            _cached_family = family
            return family

    # 우선순위 폰트가 하나도 없으면 기본 폰트 패밀리로 폴백한다.
    _cached_family = tkfont.nametofont("TkDefaultFont").actual("family")
    return _cached_family
