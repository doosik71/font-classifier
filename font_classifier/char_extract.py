"""annotation의 격자 좌표를 이용해 스캔 영상 한 칸에서 글자 하나를 잘라내
64x64 정규화 영상으로 만드는 순수 로직.

`scripts/font-dataset-browser.py`(GUI)가 사용한다. Tkinter나 다른 GUI
상태에 의존하지 않으며, 필요한 값은 모두 인자로 받는다.

정규화 규칙:

1. 칸(cell) 영역을 흑백으로 잘라내고, Otsu 방법으로 그 칸에 맞는 이진화
   임계값을 계산한다(칸마다 스캔 농도가 조금씩 다를 수 있으므로 고정
   임계값 대신 칸별로 적응적으로 계산한다). 이 임계값은 잉크 바운딩
   박스를 찾는 용도로만 쓰고, 최종 결과 영상은 이진화하지 않는다 —
   화면 표시는 그레이스케일이 원본에 더 가깝고 보기에도 자연스럽다.
2. 그 임계값으로 잉크 픽셀의 바운딩 박스를 구한다. 잉크 픽셀이 거의 없으면
   (해당 글자에 대응하는 폰트가 없어 칸이 비어 있는 경우) `None`을 반환한다.
3. 바운딩 박스만큼만 흑백(그레이스케일) 그대로 잘라내, 긴 변이 64px가
   되도록 종횡비를 유지한 채 확대 또는 축소한다. 원본보다 작으면
   확대도 한다 — 그래야 원본에 작게 인쇄된 폰트(예: YDEchoM)도 다른
   폰트와 비슷한 시각적 크기로 비교할 수 있다. 가로/세로에 같은
   배율을 적용하므로 종횡비와, 글자 크기 대비 획 두께의 상대적인
   비율은 그대로 유지된다(절대 픽셀 두께는 배율만큼 함께 변한다).
4. 64x64 흰 배경(255) 캔버스 중앙에 붙여 넣는다. 종횡비를 유지해야 하는
   이유는 한글 폰트 형태 분류에서 글자의 종횡비와 획 두께가 중요한
   특징이기 때문이다(README 참고).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .grid_autocorrect import GridParams

CHAR_SIZE = 64

# 칸 안의 잉크 픽셀이 이 개수 이하이면 해당 글자에 대응하는 폰트가 없어
# 빈 칸으로 인쇄된 것으로 본다(grid_autocorrect.collect_cell_offsets와
# 같은 기준).
MIN_INK_PIXELS = 5


def otsu_threshold(gray: np.ndarray) -> int:
    """그레이스케일 배열에 Otsu 방법으로 이진화 임계값을 계산한다.

    클래스 간 분산(between-class variance)이 최대가 되는 임계값을
    0~255 중에서 찾는다. 칸이 거의 단색(빈 칸 등)이라 분산이 0에
    가까우면 중간값 127을 그대로 반환한다.
    """

    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = float(gray.size)
    sum_total = float(np.dot(np.arange(256), hist))

    sum_below = 0.0
    weight_below = 0.0
    best_threshold = 127
    best_variance = 0.0

    for t in range(256):
        weight_below += hist[t]
        if weight_below == 0:
            continue
        weight_above = total - weight_below
        if weight_above == 0:
            break

        sum_below += t * hist[t]
        mean_below = sum_below / weight_below
        mean_above = (sum_total - sum_below) / weight_above

        variance = weight_below * weight_above * (mean_below - mean_above) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    return best_threshold


def extract_char_cell(
    image: Image.Image, params: GridParams, row: int, col: int
) -> Image.Image | None:
    """회전 보정이 이미 적용된 영상에서 (row, col) 칸의 글자를 잘라 64x64
    그레이스케일 영상으로 정규화한다. 칸에 글자가 없으면(빈 칸) `None`을
    반환한다.
    """

    left = max(0, int(round(params.origin_x + col * params.cell_w)))
    top = max(0, int(round(params.origin_y + row * params.cell_h)))
    right = min(image.width, int(round(params.origin_x + (col + 1) * params.cell_w)))
    bottom = min(image.height, int(round(params.origin_y + (row + 1) * params.cell_h)))
    if right <= left or bottom <= top:
        return None

    cell = image.crop((left, top, right, bottom)).convert("L")
    gray = np.asarray(cell)
    threshold = otsu_threshold(gray)

    ys, xs = np.nonzero(gray < threshold)
    if len(xs) <= MIN_INK_PIXELS:
        return None

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    glyph = cell.crop(bbox)

    scale = CHAR_SIZE / max(glyph.width, glyph.height)
    new_size = (max(1, round(glyph.width * scale)),
                max(1, round(glyph.height * scale)))
    glyph = glyph.resize(new_size, Image.LANCZOS)

    canvas = Image.new("L", (CHAR_SIZE, CHAR_SIZE), color=255)
    paste_x = (CHAR_SIZE - glyph.width) // 2
    paste_y = (CHAR_SIZE - glyph.height) // 2
    canvas.paste(glyph, (paste_x, paste_y))
    return canvas
