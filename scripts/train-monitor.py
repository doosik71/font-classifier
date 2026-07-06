"""학습 진행 상황 실시간 모니터. `data/checkpoints/metrics.jsonl`을
주기적으로 폴링해 변경을 감지하고, 손실·정확도 곡선을 matplotlib 창에
실시간으로 그린다.

`scripts/train-model.py`가 `--log-every` 스텝마다 `--checkpoint-dir`
(기본 `data/checkpoints`)의 `metrics.jsonl`에 한 줄짜리 JSON을 append하는데,
이 모니터는 그 파일을 읽기만 한다(학습 프로세스와 독립적으로,
아무것도 쓰지 않는다).

- 파일 감시는 `data/checkpoints/metrics.jsonl` 한 경로만 대상으로 한다.
- 폴링은 파일의 (mtime, size)를 비교해 변경이 있을 때만 다시 읽고
  다시 그린다.

자세한 사용법/설계 근거는 docs/train-monitor.md 참고.

실행:
    uv run python scripts/train-monitor.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "data" / "checkpoints"
METRICS_FILENAME = "metrics.jsonl"
SERIES_COLOR = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

# 각 패널은 서로 관련된 metric 몇 개를 한 축에 그린다. 같은 패널 안의
# metric은 선 스타일로 구분한다. 여기 나열한 key가 metrics.jsonl에 없으면
# 해당 선을 그냥 그리지 않는다.
PANELS = [
    {"title": "Total loss", "ylabel": "loss",
     "metrics": [("loss", "-", "loss")]},
    {"title": "Jamo loss (CE)", "ylabel": "loss",
     "metrics": [("loss_jamo", "-", "jamo")]},
    {"title": "Font loss", "ylabel": "loss",
     "metrics": [("loss_font", "-", "font"),
                 ("loss_font_warm", ":", "warm"),
                 ("loss_font_trn", "--", "trn")]},
    {"title": "Jamo accuracy", "ylabel": "acc",
     "metrics": [("syllable_acc", "-", "syllable"),
                 ("cho_acc", ":", "cho"),
                 ("jung_acc", "-.", "jung"),
                 ("jong_acc", (0, (1, 3)), "jong")]},
    {"title": "Font top-1 accuracy", "ylabel": "acc", "alpha_twin": True,
     "metrics": [("font_acc", "-", "top1")]},
    {"title": "Font top-5 accuracy", "ylabel": "acc", "alpha_twin": True,
     "metrics": [("font_top5_acc", "-", "top5")]},
]
GRID_ROWS, GRID_COLS = 2, 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live-plot training metrics from data/checkpoints/metrics.jsonl.")
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR,
                         help=f"이 폴더의 {METRICS_FILENAME} 파일을 찾아 그린다")
    parser.add_argument("--interval", type=float, default=5.0,
                         help="파일 변경을 확인하는 폴링 주기(초)")
    parser.add_argument("--x-axis", choices=["step", "epoch"], default="step",
                         help="가로축으로 쓸 값 (metrics.jsonl의 step/epoch)")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    """metrics.jsonl 한 파일을 dict 리스트로 읽는다. 학습 프로세스가 마지막
    줄을 쓰는 도중이라 JSON이 깨져 있을 수 있으므로, 파싱 실패한 줄은 조용히
    건너뛴다(다음 폴링에서 완성된 줄로 다시 읽힌다)."""

    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def series(records: list[dict], xkey: str, ykey: str) -> tuple[list, list]:
    """`records`에서 (xkey, ykey)가 모두 있는 점만 뽑아 (xs, ys)로 돌려준다.
    metric마다 등장 시점이 다를 수 있어(예: 나중에 추가된 font_top5_acc)
    metric별로 독립적으로 x/y 쌍을 만든다."""

    xs: list = []
    ys: list = []
    for record in records:
        if xkey in record and ykey in record and record[ykey] is not None:
            xs.append(record[xkey])
            ys.append(record[ykey])
    return xs, ys


class Monitor:
    """metrics.jsonl 폴링 + 다시 그리기 상태를 담는다."""

    def __init__(self, checkpoints_dir: Path, xkey: str) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.metrics_path = checkpoints_dir / METRICS_FILENAME
        self.xkey = xkey
        self.stat: tuple[int, int] | None = None
        self.data: list[dict] = []
        self._first = True

        self.fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 8))
        self.axes = list(axes.flat)
        # alpha_twin이 있는 패널은 twin 축을 미리 하나 만들어 재사용한다
        # (다시 그릴 때마다 새로 만들면 축이 계속 쌓인다).
        self.twins: list = []
        for ax, panel in zip(self.axes, PANELS):
            self.twins.append(ax.twinx() if panel.get("alpha_twin") else None)
        try:
            self.fig.canvas.manager.set_window_title("train-monitor")
        except Exception:
            pass

    def poll(self) -> bool:
        """파일을 확인하고 변경이 있으면 self.data를 갱신한다. 다시 그릴
        필요가 있으면 True를 돌려준다."""

        try:
            stat = self.metrics_path.stat()
        except OSError:
            if self.data or self.stat is not None:
                self.data = []
                self.stat = None
                return True
            return False

        key = (stat.st_mtime_ns, stat.st_size)
        if self.stat != key:
            self.data = load_records(self.metrics_path)
            self.stat = key
            return True
        return False

    def redraw(self) -> None:
        records = self.data
        for ax, twin, panel in zip(self.axes, self.twins, PANELS):
            ax.clear()
            if twin is not None:
                # Axes.clear()는 twin의 y축을 기본값(왼쪽)으로 되돌리므로,
                # twinx()가 걸어 둔 오른쪽 배치를 매번 다시 지정해 준다.
                twin.clear()
                twin.yaxis.set_label_position("right")
                twin.yaxis.set_ticks_position("right")
                twin.set_visible(False)
            drew_twin = False
            for ykey, style, _label in panel["metrics"]:
                xs, ys = series(records, self.xkey, ykey)
                if xs:
                    width = 1.6 if style == "-" else 1.0
                    ax.plot(xs, ys, linestyle=style, color=SERIES_COLOR, linewidth=width)
            if panel.get("alpha_twin") and twin is not None:
                xs, ys = series(records, self.xkey, "alpha")
                if xs:
                    twin.plot(xs, ys, linestyle="-", color=SERIES_COLOR,
                              linewidth=0.9, alpha=0.35)
                    drew_twin = True

            ax.set_title(panel["title"], fontsize=10)
            ax.set_xlabel(self.xkey)
            ax.set_ylabel(panel["ylabel"])
            ax.grid(True, alpha=0.3)
            # 같은 패널에 metric이 여러 개면 선 스타일 -> metric 이름 범례를 붙인다.
            if len(panel["metrics"]) > 1:
                handles = [Line2D([0], [0], color="0.3", linestyle=style)
                           for _key, style, _label in panel["metrics"]]
                labels = [label for _key, _style, label in panel["metrics"]]
                ax.legend(handles, labels, fontsize=7, loc="best")
            if drew_twin:
                twin.set_ylim(-0.02, 1.05)
                twin.set_ylabel("alpha (curriculum)", fontsize=8, color="0.4")
                twin.set_visible(True)

        self._draw_suptitle(records)
        self.fig.tight_layout(rect=(0, 0, 1, 0.94))

    def _draw_suptitle(self, records: list[dict]) -> None:
        now = time.strftime("%H:%M:%S")
        if records:
            last = records[-1]
            step = last.get("step", "?")
            epoch = last.get("epoch", "?")
            status = f"epoch {epoch}   step {step}"
        else:
            status = (f"no metrics found at {self.metrics_path} "
                      "- will appear once training starts")
        self.fig.suptitle(f"train-monitor   updated {now}   |   {status}", fontsize=11)

    def update(self, _frame) -> None:
        if self.poll() or self._first:
            self._first = False
            self.redraw()


def main() -> None:
    args = parse_args()

    backend = matplotlib.get_backend()
    if backend.lower() == "agg":
        print("[train-monitor] 경고: 비대화형 backend(agg)라 창을 띄울 수 "
              "없습니다. 디스플레이가 있는 환경에서 실행하세요.")

    print(f"[train-monitor] watching {args.checkpoints_dir / METRICS_FILENAME} "
          f"(interval {args.interval}s, x-axis {args.x_axis})")

    monitor = Monitor(args.checkpoints_dir, args.x_axis)
    # FuncAnimation 객체는 참조가 사라지면 GC되어 갱신이 멈추므로 변수에 묶어 둔다.
    anim = FuncAnimation(  # noqa: F841 - keep a reference alive
        monitor.fig, monitor.update,
        interval=max(1, int(args.interval * 1000)), cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
