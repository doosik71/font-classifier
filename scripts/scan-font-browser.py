"""data/scan 폴더의 zip 파일에 담긴 스캔 영상을 열람하고 학습용 annotation을
작성하는 GUI 도구.

각 zip 파일에는 다양한 폰트로 완성형 한글 2,350자를 인쇄 후 스캔한 jpg 영상이
담겨 있다. 하나의 영상은 첫 줄에 폰트 이름이 인쇄되어 있고, 그 아래에 가로
20열 x 세로 25행(500자)의 격자로 글자가 배치되어 있다. 폰트 하나는 연속된
5장의 영상으로 구성되며 마지막 5번째 영상은 500자에 못 미칠 수 있다.

이 도구는 zip에서 jpg를 올바르게 추출하고 격자 좌표가 실제 글자 배치와
일치하는지 화면에서 확인하는 동시에, 각 영상의 폰트 이름/시작 글자/회전
보정 각도/격자 좌표를 data/annotation 폴더에 json으로 저장한다.

실행:
    uv run python scripts/scan-font-browser.py
"""

from __future__ import annotations

import io
import json
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from grid_autocorrect import DEFAULT_GRID, GridParams, estimate_origin_and_rotation

SCAN_DIR = Path(__file__).resolve().parent.parent / "data" / "scan"
ANNOTATION_DIR = SCAN_DIR.parent / "annotation"


def build_hangul_table() -> list[str]:
    """KS X 1001(완성형) 한글 2,350자를 코드/인쇄 순서대로 생성한다.

    이 순서는 유니코드 한글 음절(U+AC00~U+D7A3) 중 iso2022_kr로 인코딩
    가능한 글자만 코드값 순으로 골라낸 것과 같다(euc_kr/cp949는 확장
    11,172자 전체를 허용하므로 사용할 수 없다). 실제 스캔 영상(sample.jpg,
    001.zip의 첫 페이지)과 대조하여 순서가 정확히 일치함을 확인했다.
    """

    table = []
    for code in range(0xAC00, 0xD7A4):
        ch = chr(code)
        try:
            ch.encode("iso2022_kr")
        except UnicodeEncodeError:
            continue
        table.append(ch)
    return table


class ScanFontBrowser(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Scan Font Browser")
        self.geometry("1300x900")

        self.hangul_table = build_hangul_table()
        self.char_index = {ch: i for i, ch in enumerate(self.hangul_table)}

        self.zip_paths: list[Path] = []
        self.zip_path: Path | None = None
        self.zip_file: zipfile.ZipFile | None = None
        self.jpg_entries: list[str] = []

        self.current_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.display_info: tuple[int, int, float] | None = None
        self._rotated_cache: tuple[float,
                                   Image.Image, Image.Image] | None = None

        self.grid_vars = {
            key: tk.StringVar(value=str(value)) for key, value in DEFAULT_GRID.items()
        }
        self.show_grid_var = tk.BooleanVar(value=True)
        self.auto_correct_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="ZIP 파일을 선택하세요")

        self.font_name_var = tk.StringVar(value="")
        self.first_char_var = tk.StringVar(value="")
        self.last_char_info_var = tk.StringVar(value="마지막 글자: -")
        self.rotation_var = tk.StringVar(value="0.0")
        self.current_annotated = False

        self._build_widgets()
        self._bind_shortcuts()
        self._load_zip_list()

    # ---------------------------------------------------------------- UI --
    def _build_widgets(self) -> None:
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=240)
        paned.add(left, weight=0)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # 왼쪽: zip 선택 + jpg 목록
        ttk.Label(left, text="ZIP 파일").pack(anchor=tk.W, padx=6, pady=(6, 0))
        self.zip_combo = ttk.Combobox(left, state="readonly")
        self.zip_combo.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.zip_combo.bind("<<ComboboxSelected>>", self._on_zip_selected)

        self.list_label_var = tk.StringVar(value="이미지 목록")
        ttk.Label(left, textvariable=self.list_label_var).pack(
            anchor=tk.W, padx=6)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.image_listbox = tk.Listbox(
            list_frame, exportselection=False, yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.image_listbox.yview)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_selected)

        self._build_annotation_panel(left)

        # 오른쪽: 격자 조정 컨트롤 + 캔버스
        controls = ttk.Frame(right)
        controls.pack(fill=tk.X, padx=6, pady=6)

        self._add_spin(controls, "cols", "열", 0)
        self._add_spin(controls, "rows", "행", 1)
        self._add_spin(controls, "origin_x", "시작 X", 2)
        self._add_spin(controls, "origin_y", "시작 Y", 3)
        self._add_spin(controls, "cell_w", "칸 너비", 4)
        self._add_spin(controls, "cell_h", "칸 높이", 5)

        ttk.Checkbutton(
            controls, text="격자 표시", variable=self.show_grid_var, command=self._redraw
        ).grid(row=0, column=12, padx=(20, 4))
        ttk.Checkbutton(
            controls,
            text="자동 보정",
            variable=self.auto_correct_var,
            command=self._on_auto_correct_toggled,
        ).grid(row=0, column=13, padx=(4, 4))
        ttk.Button(controls, text="기본값", command=self._reset_grid).grid(
            row=0, column=14, padx=(4, 0)
        )

        self.canvas = tk.Canvas(right, background="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.canvas.bind("<Configure>", lambda event: self._redraw())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        status_bar = ttk.Label(
            self, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_annotation_panel(self, left: ttk.Frame) -> None:
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=6, pady=6)
        ttk.Label(left, text="Annotation", font=(
            "", 9, "bold")).pack(anchor=tk.W, padx=6)

        ttk.Label(left, text="폰트 이름").pack(anchor=tk.W, padx=6, pady=(6, 0))
        default_font = tkfont.nametofont("TkDefaultFont")
        font_name_attrs = default_font.actual()
        font_name_attrs["size"] = round(font_name_attrs["size"] * 1.5)
        font_name_font = tkfont.Font(**font_name_attrs)
        self.font_name_font = font_name_font  # 가비지 컬렉션 방지
        ttk.Entry(
            left, textvariable=self.font_name_var, font=font_name_font
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

        ttk.Label(left, text="첫 글자").pack(anchor=tk.W, padx=6)
        first_char_entry = ttk.Entry(
            left, textvariable=self.first_char_var, font=font_name_font, width=6)
        first_char_entry.pack(anchor=tk.W, padx=6, pady=(0, 4))
        first_char_entry.bind("<KeyRelease>", lambda event: self._redraw())

        ttk.Label(left, textvariable=self.last_char_info_var, wraplength=210).pack(
            anchor=tk.W, padx=6, pady=(0, 6)
        )

        ttk.Label(left, text="회전 보정 각도 (도, 반시계 +)").pack(anchor=tk.W, padx=6)
        rotation_spin = ttk.Spinbox(
            left,
            from_=-45,
            to=45,
            increment=0.1,
            textvariable=self.rotation_var,
            width=8,
            command=self._redraw,
        )
        rotation_spin.pack(anchor=tk.W, padx=6, pady=(0, 6))
        rotation_spin.bind("<Return>", lambda event: self._redraw())
        rotation_spin.bind("<FocusOut>", lambda event: self._redraw())

        ttk.Button(
            left, text="Annotation 저장 (Ctrl+S)", command=self._on_save_clicked
        ).pack(fill=tk.X, padx=6, pady=(0, 2))
        ttk.Button(
            left, text="다음 영상으로 이동 (Ctrl+N)", command=self._on_next_image_clicked
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-s>", lambda event: self._on_save_clicked())
        self.bind_all("<Control-S>", lambda event: self._on_save_clicked())
        self.bind_all(
            "<Control-n>", lambda event: self._on_next_image_clicked())
        self.bind_all(
            "<Control-N>", lambda event: self._on_next_image_clicked())
        self.bind_all(
            "<Control-l>", lambda event: self._on_save_and_next_clicked())
        self.bind_all(
            "<Control-L>", lambda event: self._on_save_and_next_clicked())

        nudges = [
            ("<Control-Left>", lambda: self._nudge_grid("origin_x", -0.5)),
            ("<Control-Right>", lambda: self._nudge_grid("origin_x", 0.5)),
            ("<Control-Up>", lambda: self._nudge_grid("origin_y", -0.5)),
            ("<Control-Down>", lambda: self._nudge_grid("origin_y", 0.5)),
            ("<Control-Alt-Left>", lambda: self._nudge_rotation(-0.1)),
            ("<Control-Alt-Right>", lambda: self._nudge_rotation(0.1)),
        ]
        for sequence, action in nudges:
            self.bind_all(sequence, lambda event, action=action: action())
            # Listbox의 기본 클래스 바인딩이 Ctrl+방향키(특히 Ctrl+Up/Down)를
            # 먼저 가로채 bind_all까지 이벤트가 전달되지 않으므로, 목록에
            # 포커스가 있을 때를 위해 직접 바인딩하고 기본 동작을 막는다.
            self.image_listbox.bind(
                sequence, lambda event, action=action: action() or "break")

    def _nudge_grid(self, key: str, delta: float) -> None:
        try:
            value = float(self.grid_vars[key].get())
        except ValueError:
            value = DEFAULT_GRID[key]
        self.grid_vars[key].set(str(round(value + delta, 3)))
        self._redraw()

    def _nudge_rotation(self, delta: float) -> None:
        self.rotation_var.set(str(round(self._get_rotation() + delta, 2)))
        self._redraw()

    def _add_spin(self, parent: ttk.Frame, key: str, label: str, column: int) -> None:
        ttk.Label(parent, text=label).grid(
            row=0, column=column * 2, sticky=tk.E, padx=(4, 2))
        spin = ttk.Spinbox(
            parent,
            from_=0,
            to=5000,
            increment=1 if key in ("cols", "rows") else 0.5,
            textvariable=self.grid_vars[key],
            width=7,
            command=self._redraw,
        )
        spin.grid(row=0, column=column * 2 + 1, sticky=tk.W)
        spin.bind("<Return>", lambda event: self._redraw())
        spin.bind("<FocusOut>", lambda event: self._redraw())

    # ------------------------------------------------------------- 데이터 --
    def _load_zip_list(self) -> None:
        if not SCAN_DIR.exists():
            messagebox.showerror("오류", f"data/scan 폴더를 찾을 수 없습니다: {SCAN_DIR}")
            return

        self.zip_paths = sorted(SCAN_DIR.glob("*.zip"))
        self.zip_combo["values"] = [path.name for path in self.zip_paths]
        if not self.zip_paths:
            self.status_var.set(f"{SCAN_DIR} 안에 zip 파일이 없습니다")
            return

        self.zip_combo.current(0)
        self._on_zip_selected()

    def _on_zip_selected(self, event: object = None) -> None:
        index = self.zip_combo.current()
        if index < 0:
            return

        path = self.zip_paths[index]
        if self.zip_file is not None:
            self.zip_file.close()

        self.zip_file = zipfile.ZipFile(path)
        self.zip_path = path
        self.jpg_entries = sorted(
            name for name in self.zip_file.namelist() if name.lower().endswith(".jpg")
        )

        self.image_listbox.delete(0, tk.END)
        for name in self.jpg_entries:
            self.image_listbox.insert(tk.END, Path(name).name)
        self.list_label_var.set(f"이미지 목록 ({len(self.jpg_entries)})")
        self._refresh_listbox_marks()

        self.current_image = None
        if self.jpg_entries:
            self.image_listbox.selection_set(0)
            self._on_image_selected()
        else:
            self.status_var.set(f"{path.name} 안에 jpg 파일이 없습니다")
            self._redraw()

    def _on_image_selected(self, event: object = None) -> None:
        selection = self.image_listbox.curselection()
        if not selection or self.zip_file is None:
            return

        entry = self.jpg_entries[selection[0]]
        try:
            data = self.zip_file.read(entry)
            image = Image.open(io.BytesIO(data))
            image.load()
        except (OSError, zipfile.BadZipFile) as exc:
            messagebox.showerror("오류", f"이미지를 열 수 없습니다: {entry}\n{exc}")
            return

        self.current_image = image
        self._rotated_cache = None
        assert self.zip_path is not None

        annotation = self._load_annotation(Path(entry).stem)
        self.current_annotated = annotation is not None
        self._apply_annotation_fields(annotation, selection[0])

        saved_mark = " [저장됨]" if annotation is not None else ""
        self.status_var.set(
            f"{self.zip_path.name} / {Path(entry).name} - {image.width}x{image.height}{saved_mark}"
        )
        self._redraw()

    def _apply_annotation_fields(self, data: dict | None, index: int) -> None:
        """이미지 선택 시 annotation 입력란을 채운다.

        저장된 annotation이 있으면 그 값을 그대로 불러온다. 없으면 첫 글자,
        회전각, 격자 값은 이미지마다 다르므로 기본값으로 초기화하지만,
        폰트 이름은 같은 폰트의 연속된 페이지에서 매번 다시 입력하지
        않도록 이전 값을 그대로 유지한다. 첫 글자는 이전 영상의 저장된
        annotation을 바탕으로 다음 글자를 추정해 기본값으로 채운다
        (사용자가 언제든 직접 override 할 수 있다). 격자 시작 좌표와 회전
        보정 각도는 "자동 보정"이 켜져 있으면 스캔 영상을 분석해 추정한
        값을 사용한다.
        """

        if data is not None:
            self.font_name_var.set(str(data.get("font_name", "")))
            self.first_char_var.set(str(data.get("first_char", "")))
            self.rotation_var.set(str(data.get("rotation_deg", 0.0)))
            grid = data.get("grid", {})
            for key, default in DEFAULT_GRID.items():
                self.grid_vars[key].set(str(grid.get(key, default)))
        else:
            self.first_char_var.set(self._guess_first_char(index))
            self.rotation_var.set("0.0")
            for key, default in DEFAULT_GRID.items():
                self.grid_vars[key].set(str(default))

            if self.auto_correct_var.get() and self.current_image is not None:
                origin_x, origin_y, rotation_deg = self._estimate_origin_and_rotation(
                    self.current_image
                )
                self.grid_vars["origin_x"].set(str(round(origin_x, 1)))
                self.grid_vars["origin_y"].set(str(round(origin_y, 1)))
                self.rotation_var.set(str(round(rotation_deg, 2)))

    def _on_auto_correct_toggled(self) -> None:
        if self.current_annotated:
            return
        selection = self.image_listbox.curselection()
        if not selection:
            return
        self._apply_annotation_fields(None, selection[0])
        self._redraw()

    def _estimate_origin_and_rotation(self, image: Image.Image) -> tuple[float, float, float]:
        """격자 시작 좌표와 회전 보정 각도를 함께 추정한다.

        실제 추정 로직은 GUI 상태에 의존하지 않는 `grid_autocorrect`
        모듈(원리는 docs/scan-font-browser.md 2.9절 참고)에 있다. 이
        메서드는 현재 격자 설정을 `GridParams`로 넘기고, 회전이
        감지되어 새로 계산된 회전 이미지가 있으면 `_rotated_cache`에
        저장해 화면을 다시 그릴 때 `_get_display_image`가 같은 회전을
        또 계산하지 않도록 한다.
        """

        params = self._get_grid_params()
        origin_x, origin_y, rotation_deg, rotated_image = estimate_origin_and_rotation(
            image, params)
        if rotated_image is not None:
            self._rotated_cache = (rotation_deg, image, rotated_image)
        return origin_x, origin_y, rotation_deg

    def _guess_first_char(self, index: int) -> str:
        """목록에서 index번째 영상의 첫 글자 기본값을 추정한다.

        폰트 한 개는 연속된 5장(가/동/붰/웝/탈로 시작하는 500자씩)으로
        구성된다. 목록의 첫 영상은 항상 새 폰트의 1페이지이므로 '가'를
        기본값으로 삼고, 그 외에는 바로 앞 영상에 저장된 annotation의
        마지막 글자 다음 글자를 기본값으로 사용한다. 앞 영상이 아직
        저장되지 않았거나 예외적인 페이지라면 추정할 수 없으므로 빈 값을
        두어 사용자가 직접 입력하게 한다.
        """

        if index <= 0:
            return self.hangul_table[0]

        prev_entry = self.jpg_entries[index - 1]
        prev_annotation = self._load_annotation(Path(prev_entry).stem)
        if prev_annotation is None:
            return ""

        last_index = prev_annotation.get("last_char_index")
        if not isinstance(last_index, int):
            return ""

        next_index = last_index + 1
        if next_index >= len(self.hangul_table):
            next_index = 0
        return self.hangul_table[next_index]

    # --------------------------------------------------------- annotation --
    def _annotation_path(self, stem: str) -> Path:
        return ANNOTATION_DIR / f"{stem}.json"

    def _load_annotation(self, stem: str) -> dict | None:
        path = self._annotation_path(stem)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _refresh_listbox_marks(self) -> None:
        for i, entry in enumerate(self.jpg_entries):
            color = "#1a7f37" if self._annotation_path(
                Path(entry).stem).exists() else "#D32F2F"
            self.image_listbox.itemconfig(i, foreground=color)

    def _compute_last_char(self, first_char: str, cols: int, rows: int):
        idx = self.char_index.get(first_char)
        if idx is None:
            return None
        count = min(cols * rows, len(self.hangul_table) - idx)
        last_idx = idx + count - 1
        return {
            "first_char_index": idx,
            "last_char": self.hangul_table[last_idx],
            "last_char_index": last_idx,
            "char_count": count,
        }

    def _update_last_char_label(self) -> None:
        first_char = self.first_char_var.get().strip()[:1]
        if not first_char:
            self.last_char_info_var.set("마지막 글자: -")
            return

        params = self._get_grid_params()
        result = self._compute_last_char(first_char, params.cols, params.rows)
        if result is None:
            self.last_char_info_var.set("마지막 글자: - (완성형 2,350자 표에 없는 글자)")
            return

        self.last_char_info_var.set(
            f"마지막 글자: {result['last_char']}  (총 {result['char_count']}자)"
        )

    def _on_save_clicked(self) -> bool:
        selection = self.image_listbox.curselection()
        if not selection or self.current_image is None or self.zip_path is None:
            messagebox.showwarning("Annotation 저장", "먼저 이미지를 선택하세요")
            return False

        font_name = self.font_name_var.get().strip()
        if not font_name:
            messagebox.showwarning("Annotation 저장", "폰트 이름을 입력하세요")
            return False

        first_char = self.first_char_var.get().strip()[:1]
        params = self._get_grid_params()
        result = self._compute_last_char(first_char, params.cols, params.rows)
        if not first_char or result is None:
            messagebox.showwarning(
                "Annotation 저장", "첫 글자를 완성형 2,350자 중 하나로 입력하세요"
            )
            return False

        entry = self.jpg_entries[selection[0]]
        data = {
            "zip": self.zip_path.name,
            "entry": entry,
            "image_name": Path(entry).name,
            "image_width": self.current_image.width,
            "image_height": self.current_image.height,
            "font_name": font_name,
            "first_char": first_char,
            "first_char_index": result["first_char_index"],
            "last_char": result["last_char"],
            "last_char_index": result["last_char_index"],
            "char_count": result["char_count"],
            "rotation_deg": round(self._get_rotation(), 2),
            "grid": {
                "cols": params.cols,
                "rows": params.rows,
                "origin_x": round(params.origin_x, 3),
                "origin_y": round(params.origin_y, 3),
                "cell_w": round(params.cell_w, 3),
                "cell_h": round(params.cell_h, 3),
            },
        }

        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        out_path = self._annotation_path(Path(entry).stem)
        out_path.write_text(json.dumps(
            data, ensure_ascii=False, indent=2), encoding="utf-8")

        self.image_listbox.itemconfig(selection[0], foreground="#1a7f37")
        self.current_annotated = True
        self.status_var.set(f"저장됨: {out_path.name}")
        self._redraw()
        return True

    def _on_save_and_next_clicked(self) -> None:
        if self._on_save_clicked():
            self._on_next_image_clicked()

    def _on_next_image_clicked(self) -> None:
        selection = self.image_listbox.curselection()
        current_index = selection[0] if selection else -1
        next_index = current_index + 1

        if next_index >= len(self.jpg_entries):
            self.status_var.set("목록의 마지막 영상입니다")
            return

        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(next_index)
        self.image_listbox.activate(next_index)
        self.image_listbox.see(next_index)
        self._on_image_selected()

    # --------------------------------------------------------------- 격자 --
    def _get_grid_params(self) -> GridParams:
        values: dict[str, float] = {}
        for key, default in DEFAULT_GRID.items():
            try:
                values[key] = float(self.grid_vars[key].get())
            except ValueError:
                values[key] = default

        return GridParams(
            cols=max(1, int(values["cols"])),
            rows=max(1, int(values["rows"])),
            origin_x=values["origin_x"],
            origin_y=values["origin_y"],
            cell_w=max(1.0, values["cell_w"]),
            cell_h=max(1.0, values["cell_h"]),
        )

    def _reset_grid(self) -> None:
        for key, value in DEFAULT_GRID.items():
            self.grid_vars[key].set(str(value))
        self._redraw()

    def _get_rotation(self) -> float:
        try:
            return float(self.rotation_var.get())
        except ValueError:
            return 0.0

    # --------------------------------------------------------------- 화면 --
    def _get_display_image(self) -> Image.Image | None:
        if self.current_image is None:
            return None

        angle = self._get_rotation()
        if angle == 0:
            return self.current_image

        if self._rotated_cache is not None:
            cached_angle, cached_source, cached_result = self._rotated_cache
            if cached_angle == angle and cached_source is self.current_image:
                return cached_result

        fill = (255, 255, 255) if self.current_image.mode == "RGB" else 255
        rotated = self.current_image.rotate(
            angle, resample=Image.BICUBIC, expand=False, fillcolor=fill
        )
        self._rotated_cache = (angle, self.current_image, rotated)
        return rotated

    def _redraw(self) -> None:
        self._update_last_char_label()
        self.canvas.delete("all")

        image = self._get_display_image()
        if image is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        scale = min(canvas_w / image.width, canvas_h / image.height)
        display_w = max(1, int(image.width * scale))
        display_h = max(1, int(image.height * scale))

        resized = image.resize((display_w, display_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2
        self.canvas.create_image(
            offset_x, offset_y, anchor=tk.NW, image=self.tk_image)
        self.display_info = (offset_x, offset_y, scale)

        if self.show_grid_var.get():
            self._draw_grid(offset_x, offset_y, scale)

        self._draw_font_name_overlay(offset_x, offset_y, display_w)

    def _draw_font_name_overlay(self, offset_x: int, offset_y: int, display_w: int) -> None:
        font_name = self.font_name_var.get().strip()
        if not font_name:
            return

        center_x = offset_x + display_w / 2
        top_y = offset_y + 8
        text_id = self.canvas.create_text(
            center_x, top_y, text=font_name, fill="#0000ff", font=("Malgun Gothic", 16, "bold"), anchor=tk.N
        )
        bbox = self.canvas.bbox(text_id)
        if bbox:
            pad_x, pad_y = 8, 4
            background = self.canvas.create_rectangle(
                bbox[0] - pad_x,
                bbox[1] - pad_y,
                bbox[2] + pad_x,
                bbox[3] + pad_y,
                fill="#FFFFFF",
                outline="#00cc00",
            )
            self.canvas.tag_lower(background, text_id)

    def _draw_grid(self, offset_x: int, offset_y: int, scale: float) -> None:
        params = self._get_grid_params()
        grid_w = params.cols * params.cell_w
        grid_h = params.rows * params.cell_h
        color = "#00CC44" if self.current_annotated else "#FF3030"

        top = offset_y + params.origin_y * scale
        bottom = offset_y + (params.origin_y + grid_h) * scale
        for col in range(params.cols + 1):
            x = offset_x + (params.origin_x + col * params.cell_w) * scale
            self.canvas.create_line(x, top, x, bottom, fill=color, width=1)

        left = offset_x + params.origin_x * scale
        right = offset_x + (params.origin_x + grid_w) * scale
        for row in range(params.rows + 1):
            y = offset_y + (params.origin_y + row * params.cell_h) * scale
            self.canvas.create_line(left, y, right, y, fill=color, width=1)

        self._draw_expected_chars(params, offset_x, offset_y, scale)

    def _draw_expected_chars(
        self, params: GridParams, offset_x: int, offset_y: int, scale: float
    ) -> None:
        first_char = self.first_char_var.get().strip()[:1]
        start_idx = self.char_index.get(first_char)
        if start_idx is None:
            return

        font_size = max(
            7, int(min(params.cell_w, params.cell_h) * scale * 0.32))
        text_font = ("Malgun Gothic", font_size)

        for row in range(params.rows):
            for col in range(params.cols):
                idx = start_idx + row * params.cols + col
                if idx >= len(self.hangul_table):
                    return
                expected = self.hangul_table[idx]
                x = offset_x + (params.origin_x + col *
                                params.cell_w + 2) * scale
                y = offset_y + (params.origin_y + row *
                                params.cell_h + 1) * scale
                self.canvas.create_text(
                    x, y, text=expected, fill="#0057FF", font=text_font, anchor=tk.NW
                )

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.current_image is None or self.display_info is None:
            return

        offset_x, offset_y, scale = self.display_info
        image_x = (event.x - offset_x) / scale
        image_y = (event.y - offset_y) / scale

        params = self._get_grid_params()
        col = (image_x - params.origin_x) / params.cell_w
        row = (image_y - params.origin_y) / params.cell_h

        base = f"좌표 ({image_x:.0f}, {image_y:.0f})"
        if 0 <= col < params.cols and 0 <= row < params.rows:
            self.status_var.set(
                f"{base} -> 행 {int(row) + 1}, 열 {int(col) + 1}")
        else:
            self.status_var.set(f"{base} - 격자 영역 밖")


def main() -> None:
    app = ScanFontBrowser()
    app.mainloop()


if __name__ == "__main__":
    main()
