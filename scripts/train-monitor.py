"""학습 진행 상황 실시간 모니터. 체크포인트 폴더의 `metrics.jsonl`을
주기적으로 폴링해 변경을 감지하고, 손실·정확도 곡선을 tkinter 창에 embed된
matplotlib 그래프로 실시간으로 그린다.

`scripts/train-model.py`가 `--log-every` 스텝마다 `--checkpoint-dir`
(기본 `data/checkpoints`)의 `metrics.jsonl`에 한 줄짜리 JSON을 append하는데,
이 모니터는 그 파일을 읽기만 한다(학습 프로세스와 독립적으로,
아무것도 쓰지 않는다).

여러 하이퍼파라메터로 동시에 학습할 때는 각 실험이 서로 다른 체크포인트
폴더(`data/checkpoints/<run>`)에 기록한다. 이 모니터는 `--checkpoints-dir`
아래에서 `metrics.jsonl`을 가진 폴더(루트 자신 + 바로 아래 하위 폴더)를
찾아 상단 콤보박스에 나열하고, 실행 중에 그 콤보박스로 볼 폴더를
전환할 수 있다.

- 폴링은 현재 선택된 폴더의 `metrics.jsonl` 한 경로만 대상으로 하며,
  파일의 (mtime, size)를 비교해 변경이 있을 때만 다시 읽고 다시 그린다.
- 후보 폴더 목록도 주기적으로 다시 스캔해, 학습이 새로 시작되어 폴더가
  생기면 콤보박스에 자동으로 나타난다.

자세한 사용법/설계 근거는 docs/train-monitor.md 참고.

실행:
    uv run python scripts/train-monitor.py
"""

from __future__ import annotations

import argparse
import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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
        description="Live-plot training metrics; switch checkpoint folders at runtime.")
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR,
                         help=f"이 폴더(자신 + 바로 아래 하위 폴더) 중 "
                              f"{METRICS_FILENAME}이 있는 폴더를 골라 그린다")
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
    """하나의 metrics.jsonl 폴링 + 다시 그리기 상태를 담는다. 어느 폴더를
    볼지는 `set_folder`로 실행 중에 바꿀 수 있다(그림/축은 그대로 재사용)."""

    def __init__(self, xkey: str) -> None:
        self.xkey = xkey
        self.checkpoints_dir: Path | None = None
        self.metrics_path: Path | None = None
        self.stat: tuple[int, int] | None = None
        self.data: list[dict] = []

        self.fig = Figure(figsize=(15, 8))
        axes = self.fig.subplots(GRID_ROWS, GRID_COLS)
        self.axes = list(axes.flat)
        # alpha_twin이 있는 패널은 twin 축을 미리 하나 만들어 재사용한다
        # (다시 그릴 때마다 새로 만들면 축이 계속 쌓인다).
        self.twins: list = []
        for ax, panel in zip(self.axes, PANELS):
            self.twins.append(ax.twinx() if panel.get("alpha_twin") else None)

    def set_folder(self, checkpoints_dir: Path | None) -> None:
        """볼 폴더를 바꾼다. stat/데이터를 비워 다음 poll에서 새 파일을
        처음부터 다시 읽게 한다."""

        self.checkpoints_dir = checkpoints_dir
        self.metrics_path = (
            checkpoints_dir / METRICS_FILENAME if checkpoints_dir else None)
        self.stat = None
        self.data = []

    def poll(self) -> bool:
        """파일을 확인하고 변경이 있으면 self.data를 갱신한다. 다시 그릴
        필요가 있으면 True를 돌려준다."""

        if self.metrics_path is None:
            if self.data or self.stat is not None:
                self.data = []
                self.stat = None
                return True
            return False

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
        folder = self.checkpoints_dir.name if self.checkpoints_dir else "-"
        if records:
            last = records[-1]
            step = last.get("step", "?")
            epoch = last.get("epoch", "?")
            status = f"epoch {epoch}   step {step}"
        elif self.metrics_path is None:
            status = "no checkpoint folder selected"
        else:
            status = (f"no metrics found at {self.metrics_path} "
                      "- will appear once training starts")
        self.fig.suptitle(
            f"train-monitor [{folder}]   updated {now}   |   {status}", fontsize=11)


class MonitorApp:
    """tkinter 창: 상단에 폴더 선택 콤보박스, 그 아래 matplotlib 캔버스.
    주기적으로 후보 폴더를 다시 스캔하고, 현재 폴더의 metrics를 폴링한다."""

    def __init__(self, root: tk.Tk, monitor: Monitor,
                 scan_root: Path, interval: float) -> None:
        self.root = root
        self.monitor = monitor
        self.scan_root = Path(scan_root)
        self.interval_ms = max(200, int(interval * 1000))
        self.folders: list[tuple[str, Path]] = []
        self._running = True

        top = ttk.Frame(root, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Checkpoint folder:").pack(side=tk.LEFT)
        self.folder_var = tk.StringVar()
        self.combo = ttk.Combobox(top, textvariable=self.folder_var,
                                  state="readonly", width=48)
        self.combo.pack(side=tk.LEFT, padx=(6, 0))
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_selection())

        self.canvas = FigureCanvasTkAgg(monitor.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 초기 목록/선택을 채우고(변경이 없어도 첫 그리기를 하도록) 폴링 시작.
        self._refresh_folders()
        if self.monitor.metrics_path is None:
            self.monitor.redraw()
            self.canvas.draw_idle()
        self._tick()

    def _discover(self) -> list[tuple[str, Path]]:
        """scan_root 자신과 바로 아래 하위 폴더 중 metrics.jsonl이 있는
        폴더를 (라벨, 경로)로 모은다. 루트 자신은 "<이름> (base)"로 구분한다."""

        found: list[tuple[str, Path]] = []
        root = self.scan_root
        if (root / METRICS_FILENAME).exists():
            found.append((f"{root.name} (base)", root))
        try:
            subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            subdirs = []
        for sub in subdirs:
            if (sub / METRICS_FILENAME).exists():
                found.append((sub.name, sub))
        return found

    def _refresh_folders(self) -> None:
        """후보 폴더를 다시 스캔한다. 목록이 바뀌었을 때만 콤보박스를
        갱신하고, 현재 선택이 사라졌으면 첫 폴더로 되돌린다."""

        found = self._discover()
        if found == self.folders:
            return
        self.folders = found
        labels = [label for label, _path in found]
        self.combo["values"] = labels
        if self.folder_var.get() not in labels:
            self.folder_var.set(labels[0] if labels else "")
            self._apply_selection()

    def _apply_selection(self) -> None:
        label = self.folder_var.get()
        path = dict(self.folders).get(label)
        self.monitor.set_folder(path)
        self.monitor.redraw()
        self.canvas.draw_idle()
        self.root.title(f"train-monitor - {label}" if label else "train-monitor")

    def _tick(self) -> None:
        if not self._running:
            return
        self._refresh_folders()
        if self.monitor.poll():
            self.monitor.redraw()
            self.canvas.draw_idle()
        self.root.after(self.interval_ms, self._tick)

    def _on_close(self) -> None:
        self._running = False
        self.root.destroy()


def main() -> None:
    args = parse_args()

    print(f"[train-monitor] scanning {args.checkpoints_dir} for "
          f"{METRICS_FILENAME} folders "
          f"(interval {args.interval}s, x-axis {args.x_axis})")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print("[train-monitor] 경고: tkinter 창을 열 수 없습니다 "
              f"(디스플레이가 있는 환경에서 실행하세요). 원인: {exc}")
        return

    root.title("train-monitor")
    monitor = Monitor(args.x_axis)
    MonitorApp(root, monitor, args.checkpoints_dir, args.interval)
    root.mainloop()


if __name__ == "__main__":
    main()
