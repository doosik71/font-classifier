"""`data/dataset`(construct-dataset.py가 폰트별로 미리 잘라 놓은 완성형
한글 2,350자 PNG)을 폰트 단위로 열람하는 Tkinter GUI 도구.

`font-dataset-browser.py`가 `data/annotation` + `data/scan`에서 매번 새로
격자를 찾고 글자를 잘라내 보여줬다면, 이 도구는 `construct-dataset.py`가
이미 64x64로 정규화해 세로로 이어붙여 둔 `data/dataset/<번호>.png`에서
정해진 오프셋(`idx*64`)으로 자르기만 하면 되므로 grid/rotation/Otsu 계산이
전혀 필요 없다. `data/dataset`을 읽기만 하며 아무것도 쓰지 않는다.

실행:
    uv run python scripts/dataset-browser.py
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.char_extract import CHAR_SIZE
from font_classifier.font_dataset import HANGUL_TABLE, SCAN_DIR
from font_classifier.gui_fonts import korean_font_family

DATASET_DIR = SCAN_DIR.parent / "dataset"
INDEX_PATH = DATASET_DIR / "index.json"

COLS = 20
LABEL_HEIGHT = 14
CELL_MARGIN = 6
CELL_PITCH_X = CHAR_SIZE + CELL_MARGIN
CELL_PITCH_Y = LABEL_HEIGHT + CHAR_SIZE + CELL_MARGIN

CANVAS_BACKGROUND = "#D9D9D9"
GLYPH_BORDER_COLOR = "#888888"
COLOR_BLANK_GLYPH = "#FF4444"


def _label(entry: dict) -> str:
    return f"[{entry['id']:04d}] {entry['font_name']}"


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        entries = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to read {INDEX_PATH} ({exc})")
        return []
    entries.sort(key=lambda entry: entry["font_name"])
    return entries


class DatasetBrowser(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Dataset Browser")
        self.geometry("1500x950")

        # 한글 레이블용 폰트를 플랫폼에 맞게 고른다(Malgun Gothic은 Windows 전용).
        self.label_font_family = korean_font_family(root=self)

        self.entries: list[dict] = load_index()
        self.tk_images: list[ImageTk.PhotoImage] = []  # 가비지 컬렉션 방지

        self.status_var = tk.StringVar(value="왼쪽에서 폰트를 선택하세요")

        self._build_widgets()
        self._load_font_list()

    # ---------------------------------------------------------------- UI --
    def _build_widgets(self) -> None:
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=260)
        paned.add(left, weight=0)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self.list_label_var = tk.StringVar(value="폰트 목록")
        ttk.Label(left, textvariable=self.list_label_var).pack(
            anchor=tk.W, padx=6, pady=(6, 0))

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.font_listbox = tk.Listbox(
            list_frame, exportselection=False, yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.font_listbox.yview)
        self.font_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.font_listbox.bind("<<ListboxSelect>>", self._on_font_selected)

        legend = "글자 셀 테두리 - 빨강: 이 자리의 글자를 추출하지 못함(빈 칸)"
        ttk.Label(left, text=legend, wraplength=240,
                  foreground="#555555").pack(anchor=tk.W, padx=6, pady=(0, 6))

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(
            canvas_frame, background=CANVAS_BACKGROUND, yscrollcommand=vbar.set)
        vbar.config(command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        status_bar = ttk.Label(
            self, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    # ------------------------------------------------------------- 데이터 --
    def _load_font_list(self) -> None:
        if not self.entries:
            self.status_var.set(
                f"{DATASET_DIR} 에 데이터셋이 없습니다 - "
                "먼저 construct-dataset.py를 실행하세요"
            )
            return

        for entry in self.entries:
            self.font_listbox.insert(tk.END, _label(entry))

        self.list_label_var.set(f"폰트 목록 ({len(self.entries)})")

    # --------------------------------------------------------------- 화면 --
    def _on_font_selected(self, event: object = None) -> None:
        selection = self.font_listbox.curselection()
        if not selection:
            return

        entry = self.entries[selection[0]]
        self.status_var.set(f"{_label(entry)} 불러오는 중...")
        self.update_idletasks()
        self._render_font(entry)

    def _render_font(self, entry: dict) -> None:
        self.canvas.delete("all")
        self.tk_images = []

        try:
            image = Image.open(DATASET_DIR / entry["file"])
            image.load()
        except OSError as exc:
            self.status_var.set(
                f"{_label(entry)}: 영상을 열 수 없습니다 ({exc})")
            return

        blank_count = 0

        for idx, char in enumerate(HANGUL_TABLE):
            row, col = divmod(idx, COLS)
            x = col * CELL_PITCH_X
            y = row * CELL_PITCH_Y

            label_top = y + CHAR_SIZE
            self.canvas.create_text(
                x + CHAR_SIZE / 2, label_top, text=char, fill="black",
                font=(self.label_font_family, 9), anchor=tk.N,
            )

            cell = image.crop((0, idx * CHAR_SIZE, CHAR_SIZE, (idx + 1) * CHAR_SIZE))
            if cell.getextrema() == (255, 255):
                blank_count += 1
                self._draw_placeholder(x, y)
                continue

            tk_image = ImageTk.PhotoImage(cell)
            self.tk_images.append(tk_image)
            self.canvas.create_image(x, y, anchor=tk.NW, image=tk_image)
            self.canvas.create_rectangle(
                x, y, x + CHAR_SIZE, y + CHAR_SIZE, outline=GLYPH_BORDER_COLOR
            )

        rows = -(-len(HANGUL_TABLE) // COLS)
        self.canvas.config(
            scrollregion=(0, 0, COLS * CELL_PITCH_X, rows * CELL_PITCH_Y)
        )
        self.canvas.yview_moveto(0)

        self.status_var.set(
            f"{_label(entry)} — 빈 칸(추출 실패) {blank_count}자"
        )

    def _draw_placeholder(self, x: float, y: float) -> None:
        self.canvas.create_rectangle(
            x, y, x + CHAR_SIZE, y + CHAR_SIZE,
            outline=COLOR_BLANK_GLYPH, width=1,
        )


def main() -> None:
    app = DatasetBrowser()
    app.mainloop()


if __name__ == "__main__":
    main()
