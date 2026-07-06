"""`data/dataset`(construct-dataset.py가 만든 폰트별 디렉터리 +
index.json)을 모델 학습용 PyTorch `Dataset`으로 읽어 들이는 모듈.

핵심 설계는 두 가지다.

1. **메모리 안에서 바로 적용하는 augmentation**: 디스크에 저장하지 않고
   `__getitem__` 호출마다 요청된 글자 PNG 한 장을 읽어 즉석에서 변형을
   적용한다.
2. **기하 변환은 패딩 후 크롭**: `char_extract.py`가 글자의 긴 변을 정확히
   64px로 맞추므로, 원본 그대로 이동/회전/확대/기울이기를 적용하면 획이
   캔버스 경계에 걸려 잘려 나갈 수 있다. 여유 있게 패딩한 뒤 하나의 affine
   변환으로 합성 적용하고, 가운데 64x64를 다시 잘라내 잘림을 원천적으로
   막는다.

자세한 설계 근거는 docs/dataset-loader.md 참고.
"""

from __future__ import annotations

import io
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .char_extract import CHAR_SIZE
from .font_dataset import HANGUL_TABLE, SCAN_DIR, decompose_hangul_syllable

DATASET_DIR = SCAN_DIR.parent / "dataset"
NUM_CHARS = len(HANGUL_TABLE)

# 초기화 시 폰트별 유효 글자 마스크를 만들기 위해 전체 폰트를 한 번씩 읽는
# 사전 스캔(prescan)에서 사용할 스레드 수.
DEFAULT_PRESCAN_WORKERS = 8


@dataclass
class AugmentConfig:
    """augmentation 강도/확률 설정. 기본값은 docs/model-design.md 5.3절의
    "공통(약함)" 세트에 해당한다."""

    max_translate_px: float = 4.0
    max_rotate_deg: float = 5.0
    scale_range: tuple[float, float] = (0.92, 1.08)
    max_shear_deg: float = 5.0

    brightness_range: tuple[float, float] = (0.85, 1.15)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    gamma_range: tuple[float, float] = (0.85, 1.15)

    blur_prob: float = 0.15
    blur_sigma_range: tuple[float, float] = (0.3, 0.8)
    sharpen_prob: float = 0.15
    sharpen_factor_range: tuple[float, float] = (1.5, 2.5)

    noise_prob: float = 0.5
    noise_std_range: tuple[float, float] = (0.01, 0.05)

    jpeg_prob: float = 0.3
    jpeg_quality_range: tuple[int, int] = (30, 80)

    erasing_prob: float = 0.3
    erasing_count_range: tuple[int, int] = (1, 2)
    erasing_size_range: tuple[float, float] = (0.05, 0.2)


def _geometric_pad(cfg: AugmentConfig) -> int:
    half = CHAR_SIZE / 2
    angle_margin = half * math.sin(math.radians(
        cfg.max_rotate_deg + cfg.max_shear_deg))
    scale_margin = half * max(0.0, cfg.scale_range[1] - 1.0)
    return math.ceil(cfg.max_translate_px + angle_margin + scale_margin) + 2


def _load_index(dataset_dir: Path) -> list[dict]:
    index_path = dataset_dir / "index.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))

    ids = sorted(entry["id"] for entry in entries)
    if ids != list(range(len(entries))):
        raise ValueError(
            f"{index_path}: 'id' must form a contiguous 0..N-1 range "
            f"(N={len(entries)}), got e.g. {ids[:5]}"
        )

    for entry in entries:
        if "dir" not in entry:
            raise ValueError(f"{index_path}: each entry must contain 'dir'")

    entries.sort(key=lambda entry: entry["id"])
    return entries


class FontGlyphDataset(Dataset):
    """`data/dataset`의 폰트별 디렉터리에서 완성형 한글 낱글자 샘플을 읽어
    (augmentation이 적용된) 이미지와 폰트/초성/중성/종성 라벨을 반환하는
    PyTorch `Dataset`.

    빈 칸(annotation 없음/추출 실패/폰트가 그 글자를 지원하지 않음)은
    `construct-dataset.py`가 모두 흰 배경 PNG로 저장하므로, 생성자에서
    폰트별로 전체가 흰색인 칸을 걸러내 표본 공간에서 제외한다.
    """

    def __init__(
        self,
        dataset_dir: str | Path = DATASET_DIR,
        *,
        augment: bool = True,
        augment_config: AugmentConfig | None = None,
        prescan_workers: int = DEFAULT_PRESCAN_WORKERS,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.entries = _load_index(self.dataset_dir)
        self.augment = augment
        self.augment_config = augment_config or AugmentConfig()

        self._valid_index: list[tuple[int, int]] = []
        self._flat_indices_by_font: list[list[int]] = [[] for _ in self.entries]
        self._prescan(prescan_workers)

    @property
    def num_font_classes(self) -> int:
        return len(self.entries)

    def flat_indices_by_font(self) -> list[list[int]]:
        return self._flat_indices_by_font

    def _prescan(self, workers: int) -> None:
        n = len(self.entries)
        if workers <= 1:
            for font_pos in range(n):
                self._register_font(font_pos, self._decode_font(font_pos))
                self._log_prescan_progress(font_pos + 1, n)
            return

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for font_pos, arr in enumerate(pool.map(self._decode_font, range(n))):
                self._register_font(font_pos, arr)
                self._log_prescan_progress(font_pos + 1, n)

    @staticmethod
    def _log_prescan_progress(done: int, total: int) -> None:
        if done == total or done % 200 == 0:
            print(f"[FontGlyphDataset] scanned {done}/{total} font(s)...")

    def _decode_glyph(self, font_pos: int, char_idx: int) -> np.ndarray:
        """디스크에서 글자 PNG 한 장을 읽어 (64,64) uint8 배열로
        디코딩한다."""

        entry = self.entries[font_pos]
        path = self.dataset_dir / entry["dir"] / f"{char_idx:04d}.png"
        try:
            image = Image.open(path)
            image.load()
        except OSError as exc:
            raise ValueError(f"{path}: failed to open glyph PNG ({exc})") from exc

        glyph = np.array(image, dtype=np.uint8)
        if glyph.shape != (CHAR_SIZE, CHAR_SIZE):
            raise ValueError(
                f"{path}: unexpected image shape {glyph.shape}, "
                f"expected {(CHAR_SIZE, CHAR_SIZE)}"
            )
        return glyph

    def _decode_font(self, font_pos: int) -> np.ndarray:
        """디스크에서 폰트 디렉터리의 모든 글자 PNG를 읽어 (NUM_CHARS,64,64)
        uint8 배열로 디코딩한다. 사전 스캔에서 유효 글자 마스크를 만드는
        용도로만 쓴다."""

        arr = np.empty((NUM_CHARS, CHAR_SIZE, CHAR_SIZE), dtype=np.uint8)
        for char_idx in range(NUM_CHARS):
            arr[char_idx] = self._decode_glyph(font_pos, char_idx)
        return arr

    def _register_font(self, font_pos: int, arr: np.ndarray) -> None:
        mins = arr.reshape(NUM_CHARS, CHAR_SIZE * CHAR_SIZE).min(axis=1)
        flat_indices = self._flat_indices_by_font[font_pos]
        for char_idx in np.nonzero(mins < 255)[0].tolist():
            flat_indices.append(len(self._valid_index))
            self._valid_index.append((font_pos, char_idx))

    def __len__(self) -> int:
        return len(self._valid_index)

    def __getitem__(self, idx: int) -> dict:
        font_pos, char_idx = self._valid_index[idx]
        glyph = self._decode_glyph(font_pos, char_idx)

        if self.augment:
            image = _augment(glyph, self.augment_config)
        else:
            image = torch.from_numpy(glyph).to(torch.float32).div_(255.0).unsqueeze(0)

        entry = self.entries[font_pos]
        char = HANGUL_TABLE[char_idx]
        cho, jung, jong = decompose_hangul_syllable(char)

        return {
            "image": image,
            "font_label": torch.tensor(entry["id"], dtype=torch.long),
            "cho_label": torch.tensor(cho, dtype=torch.long),
            "jung_label": torch.tensor(jung, dtype=torch.long),
            "jong_label": torch.tensor(jong, dtype=torch.long),
            "font_id": entry["id"],
            "char_index": char_idx,
        }

    def __getstate__(self) -> dict:
        # 워커 프로세스로 pickle할 때, 메인 프로세스에서만 쓰는 폰트별 유효
        # 인덱스 목록(`FontGroupBatchSampler`용)은 굳이 함께 보내지 않는다.
        state = self.__dict__.copy()
        state.pop("_flat_indices_by_font", None)
        return state


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def _augment(glyph: np.ndarray, cfg: AugmentConfig) -> torch.Tensor:
    """64x64 grayscale uint8 글자 한 칸에 augmentation을 적용해 [0,1]
    범위의 (1,64,64) float32 텐서로 반환한다."""

    image = torch.from_numpy(glyph).to(torch.float32).div_(255.0).unsqueeze(0)

    image = _apply_geometric(image, cfg)

    image = TF.adjust_brightness(image, random.uniform(*cfg.brightness_range))
    image = TF.adjust_contrast(image.clamp(0.0, 1.0), random.uniform(*cfg.contrast_range))
    image = image.clamp_(0.0, 1.0).pow(random.uniform(*cfg.gamma_range))

    if random.random() < cfg.blur_prob:
        sigma = random.uniform(*cfg.blur_sigma_range)
        image = TF.gaussian_blur(image, kernel_size=3, sigma=sigma)
    elif random.random() < cfg.sharpen_prob:
        image = TF.adjust_sharpness(image, random.uniform(*cfg.sharpen_factor_range))

    if random.random() < cfg.noise_prob:
        std = random.uniform(*cfg.noise_std_range)
        image = (image + torch.randn_like(image) * std).clamp_(0.0, 1.0)

    if random.random() < cfg.jpeg_prob:
        image = _simulate_jpeg(image, random.randint(*cfg.jpeg_quality_range))

    if random.random() < cfg.erasing_prob:
        image = _random_erasing(image, cfg)

    return image.clamp_(0.0, 1.0)


def _apply_geometric(image: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """이동/회전/확대/기울이기를 하나의 affine 변환으로 합성해 한 번만
    리샘플링한다. 원본을 넉넉히 패딩한 뒤 변환하고 가운데 64x64를 다시
    잘라내므로, 설정된 범위 안의 변형은 절대 획을 캔버스 밖으로 밀어내지
    않는다(모듈 docstring 3절)."""

    pad = _geometric_pad(cfg)
    padded = torch.nn.functional.pad(image, [pad, pad, pad, pad], value=1.0)

    angle = random.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
    translate = [
        random.uniform(-cfg.max_translate_px, cfg.max_translate_px),
        random.uniform(-cfg.max_translate_px, cfg.max_translate_px),
    ]
    scale = random.uniform(*cfg.scale_range)
    shear = random.uniform(-cfg.max_shear_deg, cfg.max_shear_deg)

    warped = TF.affine(
        padded, angle=angle, translate=translate, scale=scale, shear=[shear, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=1.0,
    )
    return warped[:, pad:pad + CHAR_SIZE, pad:pad + CHAR_SIZE]


def _simulate_jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    """실제 JPEG 인코딩→디코딩을 한 번 왕복시켜 압축 아티팩트를 흉내낸다."""

    arr = (image.squeeze(0).clamp(0.0, 1.0) * 255).round().to(torch.uint8).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(decoded).unsqueeze(0)


def _random_erasing(image: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """흰색 또는 검은색 사각형 패치로 일부를 가린다(cutout). 패치 크기
    상한(`erasing_size_range`)이 작아 획 전체가 지워질 위험은 낮다."""

    image = image.clone()
    _, height, width = image.shape
    for _ in range(random.randint(*cfg.erasing_count_range)):
        eh = max(1, round(height * random.uniform(*cfg.erasing_size_range)))
        ew = max(1, round(width * random.uniform(*cfg.erasing_size_range)))
        top = random.randint(0, height - eh)
        left = random.randint(0, width - ew)
        fill = 1.0 if random.random() < 0.5 else 0.0
        image[:, top:top + eh, left:left + ew] = fill
    return image
