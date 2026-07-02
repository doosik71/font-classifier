"""data/annotation의 기존 annotation에 자동 격자/회전 보정을 일괄 적용하는
텍스트 기반 배치 도구.

scan-font-browser.py의 자동 보정 기능(grid_autocorrect 모듈, 원리는
docs/scan-font-browser.md 2.9절 참고)이 만들어지기 전에, 사람이 직접 격자
시작 좌표와 회전 보정 각도를 입력해 저장한 annotation이 data/annotation에
1,000개 넘게 있다. 이 도구는 각 annotation이 가리키는 원본 스캔 영상에
자동 보정을 다시 적용해 보고,

- 기존 값과 자동 보정 값의 차이가 작으면 자동 보정 값으로 바로 갱신하고,
- 차이가 크면 영상 파일 이름/기존 값/자동 보정 값/차이를 화면에 출력한
  뒤 적용 여부를 물어본다.

실행 전 data/annotation 폴더 전체를 타임스탬프가 붙은 폴더로 백업한다.

실행:
    uv run python scripts/auto-correct-annotation.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.grid_autocorrect import (
    DEFAULT_GRID, GridParams, estimate_origin_and_rotation,
)

SCAN_DIR = Path(__file__).resolve().parent.parent / "data" / "scan"
ANNOTATION_DIR = SCAN_DIR.parent / "annotation"

# 기존 값과 자동 보정 값의 차이가 이 값을 넘으면(px 또는 도) 화면에
# 출력하고 적용 여부를 물어본다. 그 이하면 자동 보정 값으로 바로 갱신한다.
ORIGIN_DIFF_THRESHOLD_PX = 20.0
ROTATION_DIFF_THRESHOLD_DEG = 1.5

PROGRESS_INTERVAL = 50


def backup_annotation_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ANNOTATION_DIR.parent / f"annotation_backup_{timestamp}"
    shutil.copytree(ANNOTATION_DIR, backup_dir)
    return backup_dir


def build_search_params(annotation_grid: dict) -> GridParams:
    """annotation에 저장된 칸 크기는 유지하되, 시작 좌표는 항상 표준
    기본값에서 새로 탐색하도록 GridParams를 만든다.

    시작 좌표를 기존(수동) 값에서부터 탐색하면 자동 보정 결과가 기존
    값에 가깝게 편향되어 버려 비교 의미가 없어진다. scan-font-browser.py
    가 새 영상에 자동 보정을 적용할 때와 똑같이 DEFAULT_GRID의 시작
    좌표에서 독립적으로 다시 추정해야 제대로 된 비교가 된다. 칸 크기
    (cell_w/cell_h)와 열/행 수는 모든 영상이 같은 인쇄 양식을 쓰므로
    사실상 항상 DEFAULT_GRID와 같지만, 혹시 개별 영상에서 수동으로
    조정된 값이 있다면 그 값을 존중한다.
    """

    return GridParams(
        cols=int(annotation_grid.get("cols", DEFAULT_GRID["cols"])),
        rows=int(annotation_grid.get("rows", DEFAULT_GRID["rows"])),
        origin_x=float(DEFAULT_GRID["origin_x"]),
        origin_y=float(DEFAULT_GRID["origin_y"]),
        cell_w=float(annotation_grid.get("cell_w", DEFAULT_GRID["cell_w"])),
        cell_h=float(annotation_grid.get("cell_h", DEFAULT_GRID["cell_h"])),
    )


def load_image(zip_cache: dict[str, zipfile.ZipFile], zip_name: str, entry: str) -> Image.Image:
    zip_file = zip_cache.get(zip_name)
    if zip_file is None:
        zip_file = zipfile.ZipFile(SCAN_DIR / zip_name)
        zip_cache[zip_name] = zip_file

    data = zip_file.read(entry)
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("y 또는 n으로 답해주세요.")


def process_one(path: Path, zip_cache: dict[str, zipfile.ZipFile]) -> str:
    """annotation 파일 하나를 처리하고 결과 상태 문자열을 반환한다.

    반환값: "updated"(자동 적용) / "confirmed"(물어본 뒤 적용) /
    "skipped"(물어본 뒤 건너뜀) / "unchanged"(차이 없음) / "error"(처리 실패)
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  오류: {path.name} 읽기 실패 ({exc})")
        return "error"

    zip_name = data.get("zip")
    entry = data.get("entry")
    if not zip_name or not entry:
        print(f"  오류: {path.name}에 zip/entry 정보가 없습니다.")
        return "error"

    grid = data.get("grid", {})
    manual_origin_x = float(grid.get("origin_x", DEFAULT_GRID["origin_x"]))
    manual_origin_y = float(grid.get("origin_y", DEFAULT_GRID["origin_y"]))
    manual_rotation = float(data.get("rotation_deg", 0.0))

    try:
        image = load_image(zip_cache, zip_name, entry)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"  오류: {path.name} 영상을 열 수 없습니다 ({zip_name}/{entry}): {exc}")
        return "error"

    params = build_search_params(grid)
    auto_origin_x, auto_origin_y, auto_rotation, _ = estimate_origin_and_rotation(
        image, params)

    diff_x = auto_origin_x - manual_origin_x
    diff_y = auto_origin_y - manual_origin_y
    diff_rot = auto_rotation - manual_rotation

    if abs(diff_x) < 1e-6 and abs(diff_y) < 1e-6 and abs(diff_rot) < 1e-6:
        return "unchanged"

    large_diff = (
        abs(diff_x) > ORIGIN_DIFF_THRESHOLD_PX
        or abs(diff_y) > ORIGIN_DIFF_THRESHOLD_PX
        or abs(diff_rot) > ROTATION_DIFF_THRESHOLD_DEG
    )

    if large_diff:
        print(f"\n{path.name}  ({zip_name} / {entry})")
        print(
            f"  기존 값   : origin_x={manual_origin_x:.2f} origin_y={manual_origin_y:.2f} "
            f"rotation_deg={manual_rotation:.2f}"
        )
        print(
            f"  자동 보정 : origin_x={auto_origin_x:.2f} origin_y={auto_origin_y:.2f} "
            f"rotation_deg={auto_rotation:.2f}"
        )
        print(
            f"  차이      : dx={diff_x:+.2f} dy={diff_y:+.2f} d회전={diff_rot:+.2f}")
        if not ask_yes_no("자동 보정 값을 적용할까요?"):
            return "skipped"

    data.setdefault("grid", dict(grid))
    data["grid"]["origin_x"] = round(auto_origin_x, 3)
    data["grid"]["origin_y"] = round(auto_origin_y, 3)
    data["rotation_deg"] = round(auto_rotation, 2)
    path.write_text(json.dumps(data, ensure_ascii=False,
                    indent=2), encoding="utf-8")

    return "confirmed" if large_diff else "updated"


def main() -> None:
    if not ANNOTATION_DIR.exists():
        print(f"{ANNOTATION_DIR} 폴더가 없습니다.")
        return

    paths = sorted(ANNOTATION_DIR.glob("*.json"))
    if not paths:
        print(f"{ANNOTATION_DIR} 안에 annotation 파일이 없습니다.")
        return

    print(f"총 {len(paths)}개의 annotation 파일을 검사합니다.")
    backup_dir = backup_annotation_dir()
    print(f"원본을 다음 폴더에 백업했습니다: {backup_dir}")

    zip_cache: dict[str, zipfile.ZipFile] = {}
    counts = {"updated": 0, "confirmed": 0,
              "skipped": 0, "unchanged": 0, "error": 0}

    for i, path in enumerate(paths, 1):
        counts[process_one(path, zip_cache)] += 1
        if i % PROGRESS_INTERVAL == 0 or i == len(paths):
            print(f"진행 {i}/{len(paths)} ...")

    for zip_file in zip_cache.values():
        zip_file.close()

    print("\n=== 완료 ===")
    print(f"자동 적용        : {counts['updated']}건")
    print(f"확인 후 적용     : {counts['confirmed']}건")
    print(f"확인 후 건너뜀   : {counts['skipped']}건")
    print(f"변화 없음        : {counts['unchanged']}건")
    print(f"오류             : {counts['error']}건")
    print(f"백업 위치        : {backup_dir}")


if __name__ == "__main__":
    main()
