"""data/annotation의 annotation 정보를 이용해 data/scan 원본 영상에서
완성형 한글 2,350자 낱글자 영상을 추출/정규화하여 화면에 표시하는 GUI
도구.

scan-font-browser.py가 영상(페이지) 단위로 annotation을 입력/저장하는
도구라면, 이 도구는 그렇게 저장된 annotation이 실제로 올바른 낱글자
영상을 만들어내는지 폰트 단위로 검증하기 위한 것이다. data/annotation과
data/scan을 오직 읽기만 하며 아무것도 쓰지 않는다.

왼쪽에는 annotation이 존재하는 폰트 목록을(annotation이 상당량 누락된
폰트는 제외하고), 오른쪽에는 선택한 폰트의 2,350자를 원본 페이지와 같은
20열 격자로 이어붙여 스크롤 가능한 화면에 표시한다. 각 칸은 64x64로
정규화된 그레이스케일 영상이며, 그 아래에 이 자리에 있어야 할 글자를
레이블로 보여준다(이진화는 글자 바운딩 박스를 찾는 데만 쓰고 화면에는
표시하지 않는다 — 원리는 docs/font-dataset-browser.md 참고).

annotation이 없어 표시할 수 없는 글자나, annotation은 있지만 해당
폰트에 그 글자가 없어 빈 칸으로 인쇄된 경우는 화면에 빈 칸으로 표시하고
콘솔에 경고를 출력한다.

실행:
    uv run python scripts/font-dataset-browser.py
"""

from __future__ import annotations

import io
import sys
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.char_extract import CHAR_SIZE, extract_char_cell
from font_classifier.font_dataset import (
    HANGUL_TABLE, FontEntry, SCAN_DIR, build_font_entries,
)
from font_classifier.grid_autocorrect import GridParams
from font_classifier.gui_fonts import korean_font_family

COLS = 20
LABEL_HEIGHT = 14
CELL_MARGIN = 6
CELL_PITCH_X = CHAR_SIZE + CELL_MARGIN
CELL_PITCH_Y = LABEL_HEIGHT + CHAR_SIZE + CELL_MARGIN

COLOR_COMPLETE = "#1a7f37"
COLOR_PARTIAL = "#E08A00"
COLOR_MISSING_ANNOTATION = "#FF4444"
COLOR_BLANK_GLYPH = "#FF0000"

CANVAS_BACKGROUND = "#D9D9D9"
GLYPH_BORDER_COLOR = "#888888"


class FontDatasetBrowser(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Font Dataset Browser")
        self.geometry("1500x950")

        # 한글 레이블용 폰트를 플랫폼에 맞게 고른다(Malgun Gothic은 Windows 전용).
        self.label_font_family = korean_font_family(root=self)

        self.font_entries: list[FontEntry] = build_font_entries()
        self._zip_cache: dict[str, zipfile.ZipFile] = {}
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

        legend = (
            f"초록=annotation 완비  주황=일부 누락(포함)\n"
            f"글자 셀 테두리 - 연한 빨강: annotation 없음 / 진한 빨강: 빈 칸(글자 없음)"
        )
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
        if not self.font_entries:
            self.status_var.set(
                f"{SCAN_DIR.parent / 'annotation'} 에 사용 가능한 annotation이 없습니다"
            )
            return

        for entry in self.font_entries:
            self.font_listbox.insert(tk.END, entry.font_name)
            color = COLOR_COMPLETE if entry.is_complete else COLOR_PARTIAL
            self.font_listbox.itemconfig(tk.END, foreground=color)

        self.list_label_var.set(f"폰트 목록 ({len(self.font_entries)})")

    def _get_zip(self, zip_name: str) -> zipfile.ZipFile:
        zip_file = self._zip_cache.get(zip_name)
        if zip_file is None:
            zip_file = zipfile.ZipFile(SCAN_DIR / zip_name)
            self._zip_cache[zip_name] = zip_file
        return zip_file

    def _load_rotated_page_image(self, page: dict) -> Image.Image | None:
        try:
            zip_file = self._get_zip(page["zip"])
            data = zip_file.read(page["entry"])
            image = Image.open(io.BytesIO(data))
            image.load()
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            print(
                f"[ERROR] {page.get('font_name')}: failed to open image "
                f"{page['entry']} ({exc})"
            )
            return None

        angle = float(page.get("rotation_deg", 0.0))
        if angle:
            fill = (255, 255, 255) if image.mode == "RGB" else 255
            image = image.rotate(angle, resample=Image.BICUBIC,
                                  expand=False, fillcolor=fill)
        return image

    # --------------------------------------------------------------- 화면 --
    def _on_font_selected(self, event: object = None) -> None:
        selection = self.font_listbox.curselection()
        if not selection:
            return

        entry = self.font_entries[selection[0]]
        self.status_var.set(f"{entry.font_name} 불러오는 중...")
        self.update_idletasks()
        self._render_font(entry)

    def _render_font(self, entry: FontEntry) -> None:
        self.canvas.delete("all")
        self.tk_images = []

        page_image_cache: dict[str, Image.Image | None] = {}
        missing_annotation = 0
        blank_glyph = 0

        for idx, char in enumerate(HANGUL_TABLE):
            row, col = divmod(idx, COLS)
            x = col * CELL_PITCH_X
            y = row * CELL_PITCH_Y

            label_top = y + CHAR_SIZE
            self.canvas.create_text(
                x + CHAR_SIZE / 2, label_top, text=char, fill="black",
                font=(self.label_font_family, 9), anchor=tk.N,
            )

            page = entry.char_pages.get(idx)
            glyph = None

            if page is None:
                missing_annotation += 1
                self._draw_placeholder(x, y, COLOR_MISSING_ANNOTATION)
                continue

            image_name = page["image_name"]
            if image_name not in page_image_cache:
                page_image_cache[image_name] = self._load_rotated_page_image(
                    page)
            image = page_image_cache[image_name]
            if image is None:
                self._draw_placeholder(x, y, COLOR_MISSING_ANNOTATION)
                continue

            params = GridParams(**page["grid"])
            local_row, local_col = divmod(
                idx - page["first_char_index"], params.cols)
            glyph = extract_char_cell(image, params, local_row, local_col)

            if glyph is None:
                blank_glyph += 1
                print(
                    f"[WARNING] {entry.font_name}: '{char}' (idx={idx}) looks "
                    "like a blank cell - this font seems to lack this character."
                )
                self._draw_placeholder(x, y, COLOR_BLANK_GLYPH)
                continue

            tk_image = ImageTk.PhotoImage(glyph)
            self.tk_images.append(tk_image)
            self.canvas.create_image(x, y, anchor=tk.NW, image=tk_image)
            self.canvas.create_rectangle(
                x, y, x + CHAR_SIZE, y + CHAR_SIZE, outline=GLYPH_BORDER_COLOR
            )

        if missing_annotation:
            print(
                f"[WARNING] {entry.font_name}: {missing_annotation} character(s) "
                "could not be displayed due to missing annotation."
            )

        rows = -(-len(HANGUL_TABLE) // COLS)
        self.canvas.config(
            scrollregion=(0, 0, COLS * CELL_PITCH_X, rows * CELL_PITCH_Y)
        )
        self.canvas.yview_moveto(0)

        self.status_var.set(
            f"{entry.font_name} — annotation 없음 {missing_annotation}자, "
            f"빈 칸(글자 없음) {blank_glyph}자"
        )

    def _draw_placeholder(self, x: float, y: float, color: str) -> None:
        self.canvas.create_rectangle(
            x, y, x + CHAR_SIZE, y + CHAR_SIZE, outline=color, width=1
        )


def main() -> None:
    app = FontDatasetBrowser()
    app.mainloop()


if __name__ == "__main__":
    main()
