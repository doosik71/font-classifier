"""스캔 영상에서 격자 시작 좌표와 회전 보정 각도를 자동으로 추정하는 순수 로직.

`scripts/scan-font-browser.py`(GUI)와 `scripts/auto-correct-annotation.py`
(배치 도구)가 함께 사용하는 공용 모듈이다. Tkinter나 다른 GUI 상태에
의존하지 않으며, 필요한 값(격자 파라미터)은 모두 인자로 받는다.

원리와 검증 근거는 `docs/scan-font-browser.md` 2.9절을 참고하라.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# 자동 보정 시 격자 시작 좌표(origin_x/origin_y)를 탐색하는 범위(px)
AUTO_CORRECT_RANGE = range(-20, 21)

# 영상 맨 위 폰트 이름 인쇄 영역의 높이(px). origin_y 추정 시 이 영역은
# 잉크 프로젝션에서 제외한다 (실측 결과 폰트 이름은 y=130 안쪽에 있다).
TITLE_AREA_HEIGHT = 180

# 칸 왼쪽/오른쪽(또는 위/아래) 절반의 평균 바운딩 박스 편차 차이가 이
# 값(px)을 넘어야 회전 보정이 필요하다고 판단한다. 실제로 기울어지지
# 않은 영상에서도 폰트 자체의 비대칭 잉크 분포 때문에 1~3px 정도 차이가
# 흔히 나타나므로, 그보다 확실히 큰 값으로 잡는다.
ROTATION_BIAS_THRESHOLD_PX = 2.0

# sample.jpg(1654x2338)를 실측하여 얻은 기본 격자 좌표. 모든 스캔 영상은
# 동일한 인쇄 양식을 사용하므로 기본값으로 대부분의 영상에 들어맞지만,
# 스캔 결과물의 미세한 어긋남을 확인/보정할 수 있도록 조정 가능하다.
DEFAULT_GRID = {
    "cols": 20,
    "rows": 25,
    "origin_x": 8.0,
    "origin_y": 241.0,
    "cell_w": 78.76,
    "cell_h": 78.52,
}


@dataclass
class GridParams:
    cols: int
    rows: int
    origin_x: float
    origin_y: float
    cell_w: float
    cell_h: float


# 칸별 바운딩 박스 측정 결과: (row, col, dx, dy, 칸중심x, 칸중심y)
CellRecord = tuple[int, int, float, float, float, float]


def estimate_origin_and_rotation(
    image: Image.Image, params: GridParams
) -> tuple[float, float, float, Image.Image | None]:
    """격자 시작 좌표와 회전 보정 각도를 함께 추정한다.

    1) 회전 없음(0도)을 가정하고 `estimate_origin_stage`로 시작 좌표를
       구한다. 이때 계산되는 칸별 바운딩 박스 편차를 그대로 재사용해
       `estimate_rotation_deg`로 회전 여부/각도를 판단한다 (추가로
       픽셀을 다시 훑지 않으므로 이 단계는 거의 공짜다).
    2) 회전이 없다고 판단되면(ROTATION_BIAS_THRESHOLD_PX 미만) 1)의
       결과를 그대로 반환한다. 회전이 있다고 판단되면 그 각도로 영상을
       회전시킨 뒤 `estimate_origin_stage`를 한 번 더 실행해, 회전이
       반영된 상태에서 시작 좌표를 다시 구한다. 이 재실행이 비용이 드는
       부분이라 회전이 필요할 때만 수행한다.

    회전 추정은 작은 각도(대략 2도 이하)를 가정한다. 회전이 크면 글자가
    이웃 칸으로 넘어가 칸별 바운딩 박스 측정 자체가 부정확해지므로, 한
    번 보정한 뒤 다시 반복 추정하지는 않는다.

    반환값의 마지막 항목은 회전을 적용한 결과 이미지다(회전이 없었다면
    `None`). 호출자가 이미 계산된 회전 이미지를 화면 표시 등에 재사용해
    같은 회전을 두 번 계산하지 않도록 하기 위한 것이며, 필요 없다면
    무시해도 된다.
    """

    origin_x, origin_y, records = estimate_origin_stage(image, params)
    rotation_deg = estimate_rotation_deg(records, params)

    if rotation_deg is None:
        return origin_x, origin_y, 0.0, None

    # 소수 둘째 자리로 반올림한 뒤 일관되게 사용해야 호출자가 저장하는
    # 값과 정확히 일치하고, 캐시를 이용해 같은 각도로 다시 회전시키지
    # 않을 수 있다.
    rotation_deg = round(rotation_deg, 2)
    fill = (255, 255, 255) if image.mode == "RGB" else 255
    rotated_image = image.rotate(rotation_deg, resample=Image.BICUBIC, expand=False, fillcolor=fill)
    origin_x, origin_y, _ = estimate_origin_stage(rotated_image, params)
    return origin_x, origin_y, rotation_deg, rotated_image


def estimate_origin_stage(
    image: Image.Image, params: GridParams
) -> tuple[float, float, list[CellRecord]]:
    """주어진 영상(회전 보정 여부와 무관)에서 1차(위상)+2차(바운딩 박스)로 시작 좌표를 구한다.

    1차 보정은 칸 크기(cell_w/cell_h)는 맞다고 가정하고, 스캔 영상의
    이동(translate) 오차로 어긋난 시작 좌표만 DEFAULT_GRID 기준
    -20~+20px 범위에서 보정한다. 칸 간격과 같은 주기(cell_w 또는
    cell_h)의 이산 푸리에 계수를 프로젝션에서 구하면 그 위상으로 글자
    획이 몰려 있는 위치(peak, 칸 중심)를 정확히 찾을 수 있고, 거기서 반
    칸을 빼면 칸 경계(valley)가 되는 격자 시작 좌표를 얻는다.

    origin_y를 구할 때 쓰는 행(row) 프로젝션은 열 방향으로는 격자 전체
    폭을 사용하므로, 영상 맨 위의 폰트 이름 인쇄 영역까지 함께 더해진다.
    이 영역은 격자와 무관한 잡음이므로 TITLE_AREA_HEIGHT만큼 프로젝션을
    0으로 지운 뒤 위상을 계산한다.

    1차 보정은 "칸 전체에 걸친 평균적인 주기"만 맞추므로, 글자가 칸
    안에서 위/아래 또는 좌/우로 치우쳐 인쇄된 폰트(칸의 기하학적 중심과
    실제 글자 바운딩 박스 중심이 다른 경우)에는 잘 맞지 않는다. 2차
    보정은 1차 보정 결과로 나눈 각 칸에서 실제 글자 바운딩 박스를 구해
    칸 중심과의 차이를 칸별로 계산하고, 그 평균만큼 시작 좌표를 한 번 더
    옮긴다. 이때 모은 칸별 편차(records)는 회전 각도 추정에도 재사용된다.
    """

    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    ink = (gray < 200).astype(np.float64)

    def phase_origin(profile: np.ndarray, cell: float, base: float) -> float:
        coords = np.arange(len(profile))
        freq = 2 * np.pi / cell
        coeff = np.sum(profile * np.exp(-1j * freq * coords))
        peak_pos = (-np.angle(coeff) / freq) % cell
        candidate = peak_pos - cell / 2
        origin = candidate + cell * round((base - candidate) / cell)
        low, high = AUTO_CORRECT_RANGE.start, AUTO_CORRECT_RANGE.stop - 1
        return min(max(origin, base + low), base + high)

    y_lo = max(0, int(params.origin_y) - 25)
    y_hi = min(height, int(params.origin_y + params.rows * params.cell_h) + 25)
    col_profile = ink[y_lo:y_hi, :].sum(axis=0)
    origin_x = phase_origin(col_profile, params.cell_w, params.origin_x)

    x_lo = max(0, int(params.origin_x) - 25)
    x_hi = min(width, int(params.origin_x + params.cols * params.cell_w) + 25)
    row_profile = ink[:, x_lo:x_hi].sum(axis=1)
    row_profile[:TITLE_AREA_HEIGHT] = 0
    origin_y = phase_origin(row_profile, params.cell_h, params.origin_y)

    records = collect_cell_offsets(ink, origin_x, origin_y, params)
    origin_x, origin_y = refine_origin_by_bbox(records, origin_x, origin_y, params)
    return origin_x, origin_y, records


def collect_cell_offsets(
    ink: np.ndarray, origin_x: float, origin_y: float, params: GridParams
) -> list[CellRecord]:
    """칸별로 글자 바운딩 박스 중심과 칸 중심의 차이(dx, dy)를 모은다.

    결과는 `(row, col, dx, dy, 칸중심x, 칸중심y)` 튜플 목록이며, 시작
    좌표 2차 보정과 회전 각도 추정이 이 결과를 함께 사용한다. 잉크
    픽셀이 4개 이하인 칸(마지막 페이지의 빈 칸 등)은 제외한다.
    """

    height, width = ink.shape
    records: list[CellRecord] = []

    for row in range(params.rows):
        top = max(0, int(round(origin_y + row * params.cell_h)))
        bottom = min(height, int(round(origin_y + (row + 1) * params.cell_h)))
        if bottom <= top:
            continue
        for col in range(params.cols):
            left = max(0, int(round(origin_x + col * params.cell_w)))
            right = min(width, int(round(origin_x + (col + 1) * params.cell_w)))
            if right <= left:
                continue

            cell_ys, cell_xs = np.nonzero(ink[top:bottom, left:right])
            if len(cell_xs) < 5:
                continue  # 빈 칸(마지막 페이지 등)은 제외

            bbox_center_x = (cell_xs.min() + cell_xs.max()) / 2
            bbox_center_y = (cell_ys.min() + cell_ys.max()) / 2
            dx = bbox_center_x - (right - left) / 2
            dy = bbox_center_y - (bottom - top) / 2
            records.append((row, col, dx, dy, (left + right) / 2, (top + bottom) / 2))

    return records


def refine_origin_by_bbox(
    records: list[CellRecord], origin_x: float, origin_y: float, params: GridParams
) -> tuple[float, float]:
    """칸별 글자 바운딩 박스 중심과 칸 중심의 평균 차이로 시작 좌표를 2차 보정한다."""

    if len(records) < params.cols * params.rows * 0.3:
        return origin_x, origin_y

    offsets_x = [r[2] for r in records]
    offsets_y = [r[3] for r in records]
    mean_dx = float(np.clip(np.mean(offsets_x), -params.cell_w / 2, params.cell_w / 2))
    mean_dy = float(np.clip(np.mean(offsets_y), -params.cell_h / 2, params.cell_h / 2))
    return origin_x + mean_dx, origin_y + mean_dy


def estimate_rotation_deg(records: list[CellRecord], params: GridParams) -> float | None:
    """칸별 바운딩 박스 편차의 좌/우, 상/하 그룹 차이로 회전 보정 각도를 추정한다.

    영상이 작은 각도 θ만큼 기울어져 있으면, 칸 중심 대비 글자 바운딩
    박스 중심의 편차가 영상 안에서의 위치에 따라 선형으로 달라진다:
    세로 편차(dy)는 가로 위치에, 가로 편차(dx)는 세로 위치에 비례해
    커진다. 그래서 칸을 왼쪽/오른쪽 절반으로 나눠 dy 평균 차이를 보면
    θ를, 위/아래 절반으로 나눠 dx 평균 차이를 보면 역시 θ를 독립적으로
    추정할 수 있고, 이 값을 그대로 `rotation_deg`에 넣으면(부호 반전
    없이) 기울기가 보정된다 — 합성 회전(알고 있는 각도만큼 실제로 돌린
    뒤 다시 추정)으로 검증했다.

    기울지 않은 영상에서도 폰트 자체의 비대칭 잉크 분포 때문에 1~3px
    정도의 좌/우(또는 상/하) 편차 차이가 흔히 나타나므로,
    ROTATION_BIAS_THRESHOLD_PX를 넘지 않으면 회전이 없다고 보고 `None`을
    반환한다(불필요한 재추정을 피한다).
    """

    if len(records) < params.cols * params.rows * 0.3:
        return None

    data = np.array(records)
    row_idx, col_idx, dx, dy, ccx, ccy = data.T

    left = col_idx < params.cols / 2
    right = ~left
    top = row_idx < params.rows / 2
    bottom = ~top
    if left.sum() < 5 or right.sum() < 5 or top.sum() < 5 or bottom.sum() < 5:
        return None

    dy_diff = dy[right].mean() - dy[left].mean()
    dx_diff = dx[bottom].mean() - dx[top].mean()
    if abs(dy_diff) < ROTATION_BIAS_THRESHOLD_PX and abs(dx_diff) < ROTATION_BIAS_THRESHOLD_PX:
        return None

    theta_y = dy_diff / (ccx[right].mean() - ccx[left].mean())
    theta_x = -dx_diff / (ccy[bottom].mean() - ccy[top].mean())
    return float(np.degrees((theta_y + theta_x) / 2))
