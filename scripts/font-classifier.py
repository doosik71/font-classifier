"""학습된 폰트 인식 모델(`data/checkpoints/latest.pt`)로 낱글자 하나의
한글/폰트를 인식해 결과를 보여 주는 Tkinter GUI 앱.

README의 최종 목표("사용자가 제시한 영상에서 한글 글자를 추출하고 폰트를
인식")의 낱글자 단위 버전이다. 입력 글자는 두 가지 방법으로 고른다.

- **데이터셋**: `data/dataset`의 폰트를 왼쪽 목록에서 고르고, 2,350자 격자에서
  글자 칸을 클릭한다. 이미 64x64로 정규화된 글리프이며 정답(폰트/글자)을
  알고 있으므로 화면에 함께 표시한다.
- **파일**: 임의의 영상을 연 뒤, 글자 하나를 감싸는 사각형을 드래그해
  고른다(다중 글자 사진에서 한 글자만 선택). 박스를 그리지 않으면 영상
  전체가 낱글자 하나라고 보고 그대로 쓴다. 고른 영역은 학습 때와 똑같은
  `char_extract.normalize_glyph`(Otsu 최소 바운딩 박스 + 64x64 정규화)로
  글자를 뽑는다.

인식은 `model.encode()` 한 번으로 초/중/종성 logits과 폰트 logits을 얻어
- 한글: 제한 디코딩(2,350자 표)과 개방 디코딩(11,172자)을 모두 표시하고,
- 폰트: softmax 상위 10위 폰트 이름과 확률을 표시하며, "인식된 글자를 그
  폰트로 쓴 글리프"를 `data/dataset`에서 읽어 함께 보여 준다(예: 명조체
  "가"로 인식하면 명조체의 "가" 글리프를 10위까지 나란히 표시).

증강(augmentation) 적용 여부는 체크박스로 켜고 끌 수 있다 — 켜면
`dataset_loader`의 학습용 augmentation(약한 세트)을 그대로 한 번 적용해
사진 노이즈에 대한 강건성을 눈으로 확인할 수 있다.

실행:
    uv run python scripts/font-classifier.py
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.char_extract import CHAR_SIZE, normalize_glyph
# `_augment`는 dataset_loader의 학습용 augmentation을 그대로 재사용한다 -
# 인식 입력을 학습 입력과 똑같은 경로로 변형해야 "증강 켜기"가 실제 학습
# 도메인을 반영하기 때문이다(모듈 밖 공개 래퍼를 새로 두는 대신 같은 패키지
# 안에서 직접 가져온다).
from font_classifier.dataset_loader import AugmentConfig, _augment
from font_classifier.font_dataset import HANGUL_TABLE, SCAN_DIR
from font_classifier.gui_fonts import korean_font_family
from font_classifier.model import (
    FontRecognitionModel, decode_open, decode_restricted,
)

DATASET_DIR = SCAN_DIR.parent / "dataset"
INDEX_PATH = DATASET_DIR / "index.json"
CHECKPOINT_PATH = DATASET_DIR.parent / "checkpoints" / "latest.pt"

TOP_K = 10
CHAR_TO_INDEX = {ch: idx for idx, ch in enumerate(HANGUL_TABLE)}

# 데이터셋 글자 격자(왼쪽 데이터셋 탭)의 셀 배치. dataset-browser.py와 같은 값.
GRID_COLS = 20
LABEL_HEIGHT = 14
CELL_MARGIN = 6
CELL_PITCH_X = CHAR_SIZE + CELL_MARGIN
CELL_PITCH_Y = LABEL_HEIGHT + CHAR_SIZE + CELL_MARGIN

CANVAS_BACKGROUND = "#D9D9D9"
GLYPH_BORDER_COLOR = "#888888"
COLOR_BLANK = "#FF4444"
COLOR_SELECT = "#2277EE"
COLOR_CORRECT = "#118822"
COLOR_WRONG = "#CC2222"
COLOR_GT_ROW = "#DFF3E0"

# 파일 탭에서 원본 영상을 화면에 맞춰 축소해 보여줄 최대 한 변 길이(px).
FILE_VIEW_MAX = 620
# 입력 글리프 미리보기 배율(64px -> 192px). 픽셀을 있는 그대로 보이도록 확대.
INPUT_PREVIEW_SCALE = 3


# --------------------------------------------------------------------------
# 모델 / 데이터 로딩
# --------------------------------------------------------------------------

def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to read {INDEX_PATH} ({exc})")
        return []


def load_model(device: torch.device) -> tuple[FontRecognitionModel, int]:
    """`latest.pt`에서 폰트 인식 모델을 복원한다. 폰트 클래스 수는 체크포인트의
    폰트 분류기 가중치 크기에서 직접 읽어 데이터셋 재확인 없이도 맞춘다."""

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state = checkpoint["model"]
    num_classes = state["font_head.classifier.weight"].shape[0]
    model = FontRecognitionModel(num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, num_classes


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class FontClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Font Classifier")
        self.geometry("1780x1000")

        self.kfamily = korean_font_family(root=self)
        self._default_bg = self.cget("background")  # tk 위젯 기본 배경색

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.entries = load_index()
        # 폰트 logit 인덱스 i  <->  index.json의 id = i+1 (dataset_loader의
        # font_label = id-1 규칙). id로 폰트 이름/글리프 파일을 찾는다.
        self.id_to_entry = {entry["id"]: entry for entry in self.entries}
        self.name_sorted = sorted(self.entries, key=lambda e: e["font_name"])

        self.augment_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="")

        # 데이터셋 소스 상태
        self.dataset_font: dict | None = None      # 선택된 폰트 index.json 항목
        self.dataset_char_idx: int | None = None   # 선택된 글자(HANGUL_TABLE 인덱스)
        self.grid_images: list[ImageTk.PhotoImage] = []
        self.selection_marker: int | None = None

        # 파일 소스 상태
        self.file_image: Image.Image | None = None      # 원본 PIL 영상
        self.file_view_scale = 1.0                        # 원본->화면 축소 배율
        self.file_photo: ImageTk.PhotoImage | None = None
        self.file_box: tuple[int, int, int, int] | None = None  # 원본 좌표계 박스
        self._drag_start: tuple[float, float] | None = None
        self._rubber_band: int | None = None

        # 결과 표시용 PhotoImage 참조 보관(가비지 컬렉션 방지)
        self.result_images: list[ImageTk.PhotoImage] = []
        # (font_id, char_idx) -> 글리프 PNG 캐시. 빈 칸(파일 없음)은 None으로
        # 캐시해 같은 칸을 반복해서 디스크에서 찾지 않는다.
        self._glyph_cache: dict[tuple[int, int], Image.Image | None] = {}

        self.model: FontRecognitionModel | None = None
        self.num_classes = 0

        self._build_widgets()
        self._load_model_or_disable()
        self._populate_font_list()

    # ------------------------------------------------------------ 위젯 구성 --
    def _build_widgets(self) -> None:
        outer = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        # 입력(왼쪽): 데이터셋/파일 탭
        input_frame = ttk.Frame(outer)
        outer.add(input_frame, weight=3)
        self.notebook = ttk.Notebook(input_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self._build_dataset_tab(self.notebook), text="데이터셋")
        self.notebook.add(self._build_file_tab(self.notebook), text="파일")

        # 결과(오른쪽)
        result_frame = ttk.Frame(outer)
        outer.add(result_frame, weight=2)
        self._build_result_pane(result_frame)

        # 하단 제어 바
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Checkbutton(bar, text="augmentation 적용", variable=self.augment_var).pack(
            side=tk.LEFT, padx=8, pady=6)
        self.recognize_btn = ttk.Button(bar, text="인식 실행", command=self._on_recognize)
        self.recognize_btn.pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Label(bar, textvariable=self.status_var, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    def _build_dataset_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)

        paned = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=240)
        paned.add(left, weight=0)
        self.font_list_label = tk.StringVar(value="폰트 목록")
        ttk.Label(left, textvariable=self.font_list_label).pack(anchor=tk.W, padx=6, pady=(6, 0))

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.font_listbox = tk.Listbox(
            list_frame, exportselection=False, yscrollcommand=scrollbar.set,
            font=(self.kfamily, 10))
        scrollbar.config(command=self.font_listbox.yview)
        self.font_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.font_listbox.bind("<<ListboxSelect>>", self._on_font_selected)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        ttk.Label(right, text="글자 칸을 클릭해 인식할 글자를 선택하세요").pack(
            anchor=tk.W, padx=6, pady=(6, 0))
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.grid_canvas = tk.Canvas(
            canvas_frame, background=CANVAS_BACKGROUND,
            yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        vbar.config(command=self.grid_canvas.yview)
        hbar.config(command=self.grid_canvas.xview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.grid_canvas.bind("<Button-1>", self._on_grid_click)
        self.grid_canvas.bind("<MouseWheel>", self._on_grid_wheel)
        return frame

    def _build_file_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)
        top = ttk.Frame(frame)
        top.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(top, text="영상 열기…", command=self._on_open_file).pack(side=tk.LEFT)
        ttk.Button(top, text="박스 지우기", command=self._clear_file_box).pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="글자 하나를 드래그로 감싸세요(박스 없으면 영상 전체를 낱글자로 처리)").pack(
            side=tk.LEFT, padx=6)

        self.file_canvas = tk.Canvas(frame, background=CANVAS_BACKGROUND,
                                     width=FILE_VIEW_MAX, height=FILE_VIEW_MAX)
        self.file_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.file_canvas.bind("<Button-1>", self._on_file_press)
        self.file_canvas.bind("<B1-Motion>", self._on_file_drag)
        self.file_canvas.bind("<ButtonRelease-1>", self._on_file_release)
        return frame

    def _build_result_pane(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="인식 결과", font=(self.kfamily, 12, "bold")).pack(
            anchor=tk.W, padx=8, pady=(8, 0))
        # 스크롤 가능한 결과 영역
        canvas = tk.Canvas(parent, highlightthickness=0)
        vbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.result_frame = ttk.Frame(canvas)
        self.result_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.result_frame, anchor=tk.NW)
        self._result_canvas = canvas

    # ---------------------------------------------------------- 모델 상태 --
    def _load_model_or_disable(self) -> None:
        if not CHECKPOINT_PATH.exists():
            self.recognize_btn.config(state=tk.DISABLED)
            self.status_var.set(f"체크포인트가 없습니다: {CHECKPOINT_PATH} — 먼저 학습하세요")
            return
        try:
            self.model, self.num_classes = load_model(self.device)
        except (OSError, KeyError, RuntimeError) as exc:
            self.recognize_btn.config(state=tk.DISABLED)
            self.status_var.set(f"모델 로드 실패: {exc}")
            return
        self.status_var.set(
            f"모델 로드 완료 ({self.num_classes}종 폰트, {self.device}). "
            "입력을 고른 뒤 '인식 실행'을 누르세요.")

    def _populate_font_list(self) -> None:
        if not self.entries:
            self.font_list_label.set("폰트 목록 (없음)")
            self.status_var.set(f"{DATASET_DIR}에 데이터셋이 없습니다 — construct-dataset.py 실행 필요")
            return
        for entry in self.name_sorted:
            self.font_listbox.insert(tk.END, f"[{entry['id']:04d}] {entry['font_name']}")
        self.font_list_label.set(f"폰트 목록 ({len(self.name_sorted)})")

    # ------------------------------------------------- 데이터셋 소스: 격자 --
    def _on_font_selected(self, event: object = None) -> None:
        selection = self.font_listbox.curselection()
        if not selection:
            return
        entry = self.name_sorted[selection[0]]
        self.dataset_font = entry
        self.dataset_char_idx = None
        self.selection_marker = None
        self._render_grid()
        self.status_var.set(f"'{entry['font_name']}' — 글자 칸을 클릭하세요")

    def _glyph(self, font_id: int, char_idx: int) -> Image.Image | None:
        """`data/dataset/<dir>/<char_idx:04d>.png`에서 글리프 하나를 읽는다.
        빈 칸은 `construct-dataset.py`가 파일을 만들지 않으므로, 파일이 없으면
        (그 폰트에 없는 글자이면) None을 돌려준다. 결과는 캐시한다."""

        key = (font_id, char_idx)
        if key in self._glyph_cache:
            return self._glyph_cache[key]
        entry = self.id_to_entry.get(font_id)
        image: Image.Image | None = None
        if entry is not None:
            try:
                image = Image.open(DATASET_DIR / entry["dir"] / f"{char_idx:04d}.png")
                image.load()
            except OSError:
                image = None  # 파일 없음 = 빈 칸(정상 경우)
        self._glyph_cache[key] = image
        return image

    def _render_grid(self) -> None:
        self.grid_canvas.delete("all")
        self.grid_images = []
        self.selection_marker = None
        if self.dataset_font is None:
            return
        font_id = self.dataset_font["id"]

        for idx, char in enumerate(HANGUL_TABLE):
            row, col = divmod(idx, GRID_COLS)
            x = col * CELL_PITCH_X
            y = row * CELL_PITCH_Y
            self.grid_canvas.create_text(
                x + CHAR_SIZE / 2, y + CHAR_SIZE, text=char, fill="black",
                font=(self.kfamily, 9), anchor=tk.N)
            cell = self._glyph(font_id, idx)
            if cell is None:  # 빈 칸: 이 폰트에 없는 글자(파일 없음)
                self.grid_canvas.create_rectangle(
                    x, y, x + CHAR_SIZE, y + CHAR_SIZE, outline=COLOR_BLANK)
                continue
            photo = ImageTk.PhotoImage(cell)
            self.grid_images.append(photo)
            self.grid_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.grid_canvas.create_rectangle(
                x, y, x + CHAR_SIZE, y + CHAR_SIZE, outline=GLYPH_BORDER_COLOR)

        rows = -(-len(HANGUL_TABLE) // GRID_COLS)
        self.grid_canvas.config(
            scrollregion=(0, 0, GRID_COLS * CELL_PITCH_X, rows * CELL_PITCH_Y))
        self.grid_canvas.xview_moveto(0)
        self.grid_canvas.yview_moveto(0)

    def _on_grid_wheel(self, event: tk.Event) -> None:
        self.grid_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_grid_click(self, event: tk.Event) -> None:
        if self.dataset_font is None:
            return
        cx = self.grid_canvas.canvasx(event.x)
        cy = self.grid_canvas.canvasy(event.y)
        col = int(cx // CELL_PITCH_X)
        row = int(cy // CELL_PITCH_Y)
        if col < 0 or col >= GRID_COLS or row < 0:
            return
        idx = row * GRID_COLS + col
        # 셀 아래 레이블 영역(글자 그림 밖)을 눌러도 같은 칸으로 본다.
        if idx >= len(HANGUL_TABLE):
            return
        self.dataset_char_idx = idx
        self._mark_selection(row, col)
        self.status_var.set(
            f"선택: '{self.dataset_font['font_name']}' / '{HANGUL_TABLE[idx]}'")

    def _mark_selection(self, row: int, col: int) -> None:
        if self.selection_marker is not None:
            self.grid_canvas.delete(self.selection_marker)
        x = col * CELL_PITCH_X
        y = row * CELL_PITCH_Y
        self.selection_marker = self.grid_canvas.create_rectangle(
            x - 1, y - 1, x + CHAR_SIZE + 1, y + CHAR_SIZE + 1,
            outline=COLOR_SELECT, width=3)

    # ------------------------------------------------------ 파일 소스: 박스 --
    def _on_open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="영상 선택",
            filetypes=[("이미지", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"),
                       ("모든 파일", "*.*")])
        if not path:
            return
        try:
            image = Image.open(path)
            image.load()
        except OSError as exc:
            messagebox.showerror("영상 열기 실패", str(exc))
            return
        self.file_image = image
        self.file_box = None
        self._render_file_image()
        self.status_var.set(f"영상 로드: {Path(path).name} — 글자를 드래그로 감싸거나 그대로 인식")

    def _render_file_image(self) -> None:
        self.file_canvas.delete("all")
        self._rubber_band = None
        image = self.file_image
        if image is None:
            return
        scale = min(1.0, FILE_VIEW_MAX / max(image.width, image.height))
        self.file_view_scale = scale
        disp = image.convert("RGB")
        if scale < 1.0:
            disp = disp.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS)
        self.file_photo = ImageTk.PhotoImage(disp)
        self.file_canvas.config(scrollregion=(0, 0, disp.width, disp.height))
        self.file_canvas.create_image(0, 0, anchor=tk.NW, image=self.file_photo)
        self._draw_file_box()

    def _draw_file_box(self) -> None:
        if self._rubber_band is not None:
            self.file_canvas.delete(self._rubber_band)
            self._rubber_band = None
        if self.file_box is None:
            return
        x0, y0, x1, y1 = self.file_box
        s = self.file_view_scale
        self._rubber_band = self.file_canvas.create_rectangle(
            x0 * s, y0 * s, x1 * s, y1 * s, outline=COLOR_SELECT, width=2)

    def _clear_file_box(self) -> None:
        self.file_box = None
        self._draw_file_box()

    def _on_file_press(self, event: tk.Event) -> None:
        if self.file_image is None:
            return
        self._drag_start = (self.file_canvas.canvasx(event.x),
                            self.file_canvas.canvasy(event.y))
        if self._rubber_band is not None:
            self.file_canvas.delete(self._rubber_band)
        self._rubber_band = self.file_canvas.create_rectangle(
            *self._drag_start, *self._drag_start, outline=COLOR_SELECT, width=2)

    def _on_file_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        cx = self.file_canvas.canvasx(event.x)
        cy = self.file_canvas.canvasy(event.y)
        self.file_canvas.coords(self._rubber_band, *self._drag_start, cx, cy)

    def _on_file_release(self, event: tk.Event) -> None:
        if self._drag_start is None or self.file_image is None:
            return
        cx = self.file_canvas.canvasx(event.x)
        cy = self.file_canvas.canvasy(event.y)
        (sx, sy) = self._drag_start
        self._drag_start = None
        # 화면 좌표 -> 원본 좌표. 작은 클릭(드래그 아님)은 박스 없음으로 본다.
        if abs(cx - sx) < 4 or abs(cy - sy) < 4:
            self.file_box = None
            self._draw_file_box()
            return
        s = self.file_view_scale
        x0, x1 = sorted((sx / s, cx / s))
        y0, y1 = sorted((sy / s, cy / s))
        x0 = max(0, min(self.file_image.width, int(round(x0))))
        x1 = max(0, min(self.file_image.width, int(round(x1))))
        y0 = max(0, min(self.file_image.height, int(round(y0))))
        y1 = max(0, min(self.file_image.height, int(round(y1))))
        self.file_box = (x0, y0, x1, y1)

    # --------------------------------------------------------------- 인식 --
    def _current_input(self) -> tuple[Image.Image, tuple[str, str] | None] | None:
        """활성 탭에 따라 인식할 64x64 글리프와 (있으면) 정답(폰트명, 글자)을
        만든다. 만들 수 없으면 사용자에게 알리고 None을 반환한다."""

        tab = self.notebook.index(self.notebook.select())
        if tab == 0:  # 데이터셋
            if self.dataset_font is None or self.dataset_char_idx is None:
                messagebox.showinfo("입력 필요", "폰트와 글자를 먼저 선택하세요.")
                return None
            idx = self.dataset_char_idx
            cell = self._glyph(self.dataset_font["id"], idx)
            if cell is None:
                messagebox.showinfo(
                    "빈 칸", "이 폰트에는 해당 글자가 없습니다(빈 칸). 다른 글자를 고르세요.")
                return None
            truth = (self.dataset_font["font_name"], HANGUL_TABLE[idx])
            return cell.convert("L"), truth

        # 파일
        if self.file_image is None:
            messagebox.showinfo("입력 필요", "먼저 영상을 여세요.")
            return None
        region = self.file_image
        if self.file_box is not None:
            region = self.file_image.crop(self.file_box)
        glyph = normalize_glyph(region)
        if glyph is None:
            messagebox.showinfo(
                "글자 없음", "선택한 영역에서 글자를 찾지 못했습니다. 박스를 다시 그려 보세요.")
            return None
        return glyph, None

    def _prepare_tensor(self, glyph: Image.Image) -> tuple[torch.Tensor, Image.Image]:
        """정규화된 64x64 글리프를 모델 입력 텐서로 만든다. 증강이 켜져 있으면
        학습용 augmentation을 한 번 적용하고, 실제로 모델에 들어간 영상을
        미리보기용 PIL로도 돌려준다."""

        arr = np.asarray(glyph.convert("L"), dtype=np.uint8)
        if self.augment_var.get():
            tensor = _augment(arr, AugmentConfig())  # (1,64,64), [0,1]
        else:
            tensor = torch.from_numpy(arr).to(torch.float32).div_(255.0).unsqueeze(0)
        shown = (tensor.squeeze(0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        preview = Image.fromarray(shown, mode="L")
        return tensor.unsqueeze(0).to(self.device), preview

    def _on_recognize(self) -> None:
        if self.model is None:
            return
        prepared = self._current_input()
        if prepared is None:
            return
        glyph, truth = prepared
        tensor, preview = self._prepare_tensor(glyph)

        with torch.no_grad():
            out = self.model.encode(tensor)
            restricted = decode_restricted(
                out.cho_logits, out.jung_logits, out.jong_logits)[0]
            open_char = decode_open(
                out.cho_logits, out.jung_logits, out.jong_logits)[0]
            probs = out.font_logits.softmax(dim=-1)[0].cpu()

        k = min(TOP_K, self.num_classes)
        top = probs.topk(k)
        top_indices = top.indices.tolist()
        top_probs = top.values.tolist()

        gt_rank = None
        gt_prob = None
        if truth is not None and self.dataset_font is not None:
            gt_index = self.dataset_font["id"] - 1
            if 0 <= gt_index < probs.numel():
                gt_prob = float(probs[gt_index])
                gt_rank = int((probs > probs[gt_index]).sum().item()) + 1

        self._render_results(preview, truth, restricted, open_char,
                             top_indices, top_probs, gt_rank, gt_prob)
        self.status_var.set(
            f"인식 완료 — 한글 '{restricted}', 폰트 top-1 "
            f"'{self.id_to_entry.get(top_indices[0] + 1, {}).get('font_name', '?')}'"
            f" ({top_probs[0] * 100:.1f}%)")

    # --------------------------------------------------------- 결과 렌더링 --
    def _render_results(
        self, preview: Image.Image, truth: tuple[str, str] | None,
        restricted: str, open_char: str,
        top_indices: list[int], top_probs: list[float],
        gt_rank: int | None, gt_prob: float | None,
    ) -> None:
        for child in self.result_frame.winfo_children():
            child.destroy()
        self.result_images = []
        kfont = (self.kfamily, 11)

        # 입력 글리프 미리보기
        head = ttk.Frame(self.result_frame)
        head.pack(fill=tk.X, anchor=tk.W, pady=(0, 8))
        big = preview.resize(
            (CHAR_SIZE * INPUT_PREVIEW_SCALE, CHAR_SIZE * INPUT_PREVIEW_SCALE),
            Image.NEAREST)
        photo = ImageTk.PhotoImage(big)
        self.result_images.append(photo)
        tk.Label(head, image=photo, borderwidth=1, relief=tk.SOLID).pack(side=tk.LEFT)
        info = ttk.Frame(head)
        info.pack(side=tk.LEFT, padx=12, anchor=tk.N)
        ttk.Label(info, text="입력 글자(모델에 들어간 영상)", font=kfont).pack(anchor=tk.W)
        if truth is not None:
            gt_id = self.dataset_font["id"] if self.dataset_font else None
            ttk.Label(info, text=f"정답 폰트: {truth[0]} (id={gt_id})",
                      font=kfont).pack(anchor=tk.W)
            ttk.Label(info, text=f"정답 글자: {truth[1]}", font=kfont).pack(anchor=tk.W)
        else:
            ttk.Label(info, text="정답: (파일 입력 — 알 수 없음)", font=kfont,
                      foreground="#666666").pack(anchor=tk.W)

        # 한글 인식 결과
        hangul = ttk.LabelFrame(self.result_frame, text="한글 인식")
        hangul.pack(fill=tk.X, anchor=tk.W, pady=6)
        truth_char = truth[1] if truth is not None else None
        self._hangul_line(hangul, "제한 디코딩(2,350자)", restricted, truth_char, kfont)
        self._hangul_line(hangul, "개방 디코딩(11,172자)", open_char, truth_char, kfont)

        # 폰트 top-k
        font_box = ttk.LabelFrame(self.result_frame, text=f"폰트 상위 {len(top_indices)}위")
        font_box.pack(fill=tk.X, anchor=tk.W, pady=6)
        gt_index = (self.dataset_font["id"] - 1) if (truth is not None and self.dataset_font) else None
        for rank, (logit_idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
            self._font_row(font_box, rank, logit_idx, prob, restricted,
                           is_gt=(logit_idx == gt_index), kfont=kfont)

        if truth is not None and gt_rank is not None and gt_rank > len(top_indices):
            ttk.Label(
                self.result_frame,
                text=f"정답 폰트 '{truth[0]}'는 {gt_rank}위 "
                     f"({(gt_prob or 0) * 100:.2f}%)로 상위 {len(top_indices)}위 밖입니다.",
                font=kfont, foreground=COLOR_WRONG).pack(anchor=tk.W, pady=(4, 0))

    def _hangul_line(self, parent: tk.Widget, label: str, char: str,
                     truth_char: str | None, kfont: tuple) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, anchor=tk.W, padx=6, pady=2)
        ttk.Label(row, text=f"{label}: ", font=kfont).pack(side=tk.LEFT)
        color = "black"
        suffix = ""
        if truth_char is not None:
            correct = (char == truth_char)
            color = COLOR_CORRECT if correct else COLOR_WRONG
            suffix = "  (O)" if correct else f"  (X) (정답 {truth_char})"
        tk.Label(row, text=char + suffix, font=(self.kfamily, 13, "bold"),
                 fg=color).pack(side=tk.LEFT)

    def _font_row(self, parent: tk.Widget, rank: int, logit_idx: int, prob: float,
                  recognized_char: str, is_gt: bool, kfont: tuple) -> None:
        bg = COLOR_GT_ROW if is_gt else self._default_bg
        row = tk.Frame(parent, bg=bg)
        row.pack(fill=tk.X, anchor=tk.W, padx=4, pady=1)

        tk.Label(row, text=f"{rank:2d}.", font=kfont, width=3, bg=bg).pack(side=tk.LEFT)

        # 인식된 글자를 이 폰트로 쓴 글리프
        glyph = self._font_glyph(logit_idx + 1, recognized_char)
        if glyph is not None:
            photo = ImageTk.PhotoImage(glyph)
            self.result_images.append(photo)
            tk.Label(row, image=photo, borderwidth=1, relief=tk.SOLID, bg=bg).pack(side=tk.LEFT)
        else:
            tk.Label(row, text="(없음)", width=8, height=4, relief=tk.SOLID,
                     fg=COLOR_BLANK, bg=bg, font=kfont).pack(side=tk.LEFT)

        entry = self.id_to_entry.get(logit_idx + 1)
        name = f"{entry['font_name']} (id={entry['id']})" if entry else f"id={logit_idx + 1}?"
        text = f"  {prob * 100:5.1f}%   {name}"
        if is_gt:
            text += "   ← 정답"
        tk.Label(row, text=text, font=kfont, anchor=tk.W, justify=tk.LEFT, bg=bg).pack(
            side=tk.LEFT, padx=6)

    def _font_glyph(self, font_id: int, char: str) -> Image.Image | None:
        """`char`(인식된 글자)를 `font_id` 폰트로 쓴 64x64 글리프를 데이터셋에서
        읽는다. 표 밖 글자이거나 그 폰트에 없으면 None."""

        idx = CHAR_TO_INDEX.get(char)
        if idx is None:
            return None
        cell = self._glyph(font_id, idx)
        if cell is None:
            return None
        return cell.copy()


def main() -> None:
    app = FontClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
