"""학습 진행 상황 실시간 모니터. `data/checkpoints/*/metrics.jsonl`을
주기적으로 폴링해 변경을 감지하고, 각 run(예: `v1`, `v2`)의 손실·정확도
곡선을 matplotlib 창에 겹쳐 그려 v1 방법과 v2 방법을 나란히 비교할 수
있게 한다.

`scripts/train-model-v1.py`/`train-model-v2.py`가 `--log-every` 스텝마다
`--checkpoint-dir`(각각 `data/checkpoints/v1`, `.../v2`)의 `metrics.jsonl`에
한 줄짜리 JSON을 append하는데(각 스크립트 1.5절), 이 모니터는 그 파일들을
읽기만 한다(학습 프로세스와 독립적으로, 아무것도 쓰지 않는다).

- run 구분은 `metrics.jsonl`이 들어 있는 **하위 폴더 이름**으로 한다
  (`data/checkpoints/<run>/metrics.jsonl` → run 이름 `<run>`). 같은 패널
  안에서 run은 색으로, 서로 다른 metric은 선 스타일로 구분한다.
- 폴링은 각 파일의 (mtime, size)를 비교해 변경이 있을 때만 다시 읽고
  다시 그린다. 학습 중 새 run 폴더가 생기면 다음 폴링에서 자동으로
  잡힌다.

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

# 각 패널은 서로 관련된 metric 몇 개를 한 축에 그린다. run은 색으로,
# 같은 패널 안의 metric은 선 스타일로 구분한다. 여기 나열한 key가
# metrics.jsonl에 없으면(예: v1에는 loss_font_warm/trn/alpha가 없다)
# 그 run에서는 해당 선을 그냥 그리지 않는다.
PANELS = [
    {"title": "Total loss", "ylabel": "loss",
     "metrics": [("loss", "-", "loss")]},
    {"title": "Jamo loss (CE)", "ylabel": "loss",
     "metrics": [("loss_jamo", "-", "jamo")]},
    {"title": "Font loss", "ylabel": "loss",
     # 그림 안 텍스트는 matplotlib 기본 폰트가 한글을 못 그려 tofu(□)가 되므로
     # 영어로 둔다(콘솔/문서는 한글). v1=softmax, v2=sigmoid라 스케일이 다르다.
     "note": "v1(softmax) vs v2(sigmoid): scales differ - don't compare absolute values",
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
        description="Live-plot training metrics from data/checkpoints/*/metrics.jsonl.")
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR,
                         help="이 폴더 아래 <run>/metrics.jsonl 들을 찾아 그린다")
    parser.add_argument("--interval", type=float, default=5.0,
                         help="파일 변경을 확인하는 폴링 주기(초)")
    parser.add_argument("--x-axis", choices=["step", "epoch"], default="step",
                         help="가로축으로 쓸 값 (metrics.jsonl의 step/epoch)")
    parser.add_argument("--runs", nargs="*", default=None,
                         help="그릴 run 이름을 제한한다(예: --runs v1 v2). "
                              "생략하면 발견되는 모든 run을 그린다")
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


def assign_colors(run_names: list[str]) -> dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {name: cycle[i % len(cycle)] for i, name in enumerate(run_names)}


class Monitor:
    """metrics.jsonl 폴링 + 다시 그리기 상태를 담는다."""

    def __init__(self, checkpoints_dir: Path, xkey: str, runs: list[str] | None) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.xkey = xkey
        self.run_filter = set(runs) if runs else None
        self.stat: dict[Path, tuple[int, int]] = {}   # path -> (mtime_ns, size)
        self.data: dict[str, list[dict]] = {}          # run 이름 -> records
        self.colors: dict[str, str] = {}
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

        files = sorted(self.checkpoints_dir.glob("*/metrics.jsonl"))
        changed = False
        seen_runs: set[str] = set()
        for path in files:
            run = path.parent.name
            if self.run_filter is not None and run not in self.run_filter:
                continue
            seen_runs.add(run)
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (stat.st_mtime_ns, stat.st_size)
            if self.stat.get(path) != key:
                self.data[run] = load_records(path)
                self.stat[path] = key
                changed = True

        for run in list(self.data):
            if run not in seen_runs:
                del self.data[run]
                changed = True

        run_names = sorted(self.data)
        if set(run_names) != set(self.colors):
            self.colors = assign_colors(run_names)
            changed = True
        return changed

    def redraw(self) -> None:
        run_names = sorted(self.data)
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
            for run in run_names:
                records = self.data[run]
                color = self.colors[run]
                for ykey, style, _label in panel["metrics"]:
                    xs, ys = series(records, self.xkey, ykey)
                    if xs:
                        width = 1.6 if style == "-" else 1.0
                        ax.plot(xs, ys, linestyle=style, color=color, linewidth=width)
                if panel.get("alpha_twin") and twin is not None:
                    xs, ys = series(records, self.xkey, "alpha")
                    if xs:
                        twin.plot(xs, ys, linestyle="-", color=color,
                                  linewidth=0.9, alpha=0.35)
                        drew_twin = True

            ax.set_title(panel["title"], fontsize=10)
            ax.set_xlabel(self.xkey)
            ax.set_ylabel(panel["ylabel"])
            ax.grid(True, alpha=0.3)
            if panel.get("note"):
                ax.text(0.01, 0.02, panel["note"], transform=ax.transAxes,
                        fontsize=6.5, color="0.4", va="bottom")
            # 같은 패널에 metric이 여러 개면 선 스타일 → metric 이름 범례를
            # (색과 무관한 회색 견본으로) 붙인다. run(색) 범례는 그림 상단에
            # 따로 둔다.
            if len(panel["metrics"]) > 1:
                handles = [Line2D([0], [0], color="0.3", linestyle=style)
                           for _key, style, _label in panel["metrics"]]
                labels = [label for _key, _style, label in panel["metrics"]]
                ax.legend(handles, labels, fontsize=7, loc="best")
            if drew_twin:
                twin.set_ylim(-0.02, 1.05)
                twin.set_ylabel("alpha (curriculum)", fontsize=8, color="0.4")
                twin.set_visible(True)

        self._draw_run_legend(run_names)
        self._draw_suptitle(run_names)
        self.fig.tight_layout(rect=(0, 0, 1, 0.94))

    def _draw_run_legend(self, run_names: list[str]) -> None:
        for legend in list(self.fig.legends):
            legend.remove()
        if not run_names:
            return
        handles = [Line2D([0], [0], color=self.colors[run], linewidth=2)
                   for run in run_names]
        self.fig.legend(handles, run_names, loc="upper right",
                        ncol=len(run_names), fontsize=9, title="run")

    def _draw_suptitle(self, run_names: list[str]) -> None:
        now = time.strftime("%H:%M:%S")
        if run_names:
            parts = []
            for run in run_names:
                records = self.data[run]
                last = records[-1] if records else {}
                step = last.get("step", "?")
                parts.append(f"{run}: step {step}")
            status = "   ".join(parts)
        else:
            status = (f"no metrics found under {self.checkpoints_dir}/*/metrics.jsonl "
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

    print(f"[train-monitor] watching {args.checkpoints_dir}/*/metrics.jsonl "
          f"(interval {args.interval}s, x-axis {args.x_axis})")

    monitor = Monitor(args.checkpoints_dir, args.x_axis, args.runs)
    # FuncAnimation 객체는 참조가 사라지면 GC되어 갱신이 멈추므로 변수에 묶어 둔다.
    anim = FuncAnimation(  # noqa: F841 - keep a reference alive
        monitor.fig, monitor.update,
        interval=max(1, int(args.interval * 1000)), cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
