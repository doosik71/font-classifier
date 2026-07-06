"""`data/dataset-2`(글자 중심 chunk PNG + index.json)을 모델 학습용
PyTorch `Dataset`으로 읽어 들이는 모듈.

`dataset_loader.py`가 "폰트 하나 = PNG 한 장" 구조에 맞춰 폰트 단위 캐시를
두었다면, 이 모듈은 "여러 폰트 x 몇 개 글자" chunk PNG를 읽고 chunk 단위
LRU 캐시를 둔다. 샘플 하나는 여전히 `(font, char)` 낱글자 1개이며, 반환
형식은 기존 `FontGlyphDataset`과 동일하게 유지해 학습 루프를 그대로
재사용할 수 있게 한다.
"""

from __future__ import annotations

import io
import json
import math
import random
from collections import OrderedDict
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
from .font_dataset import HANGUL_TABLE, decompose_hangul_syllable

DATASET2_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset-2"
NUM_CHARS = len(HANGUL_TABLE)


@dataclass(frozen=True)
class ChunkEntry:
    chunk_id: int
    char_start: int
    char_end: int
    file: str

    @property
    def char_count(self) -> int:
        return self.char_end - self.char_start + 1


@dataclass
class AugmentConfig:
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


class _LRUChunkCache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._data: OrderedDict[int, np.ndarray] = OrderedDict()
        self._nbytes = 0

    def get(self, key: int) -> np.ndarray | None:
        arr = self._data.get(key)
        if arr is not None:
            self._data.move_to_end(key)
        return arr

    def put(self, key: int, arr: np.ndarray) -> None:
        old = self._data.get(key)
        if old is not None:
            self._nbytes -= old.nbytes
        self._data[key] = arr
        self._data.move_to_end(key)
        self._nbytes += arr.nbytes
        while self._nbytes > self.max_bytes and len(self._data) > 1:
            _, evicted = self._data.popitem(last=False)
            self._nbytes -= evicted.nbytes


def _load_index(dataset_dir: Path) -> tuple[list[dict], list[ChunkEntry], int]:
    index_path = dataset_dir / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("format") != "char_chunks_v1":
        raise ValueError(f"{index_path}: unsupported dataset-2 format")

    if int(payload.get("char_size", 0)) != CHAR_SIZE:
        raise ValueError(
            f"{index_path}: expected char_size={CHAR_SIZE}, "
            f"got {payload.get('char_size')}"
        )

    fonts = payload["fonts"]
    chunks = [ChunkEntry(**chunk) for chunk in payload["chunks"]]
    chunk_size = int(payload["chunk_size"])

    ids = sorted(entry["id"] for entry in fonts)
    if ids != list(range(1, len(fonts) + 1)):
        raise ValueError(
            f"{index_path}: 'fonts[].id' must form a contiguous 1..N range "
            f"(N={len(fonts)})"
        )

    expected_start = 0
    for chunk_id, chunk in enumerate(chunks):
        if chunk.chunk_id != chunk_id:
            raise ValueError(
                f"{index_path}: chunk_id mismatch at position {chunk_id}: "
                f"{chunk.chunk_id}"
            )
        if chunk.char_start != expected_start:
            raise ValueError(
                f"{index_path}: expected char_start={expected_start}, "
                f"got {chunk.char_start}"
            )
        if chunk.char_count <= 0 or chunk.char_count > chunk_size:
            raise ValueError(
                f"{index_path}: invalid chunk char_count={chunk.char_count}"
            )
        expected_start = chunk.char_end + 1

    if expected_start != NUM_CHARS:
        raise ValueError(
            f"{index_path}: chunks cover {expected_start} characters, "
            f"expected {NUM_CHARS}"
        )

    return fonts, chunks, chunk_size


class CharGroupedFontDataset(Dataset):
    """`data/dataset-2`의 char-major chunk PNG를 읽어 `(font, char)` 샘플을
    반환하는 Dataset.

    각 chunk PNG는 "세로축 = 폰트, 가로축 = chunk 안의 연속 글자들" 구조다.
    따라서 같은 글자를 여러 폰트로 묶어 읽는 배치가 chunk 캐시를 잘 활용할
    수 있다.
    """

    def __init__(
        self,
        dataset_dir: str | Path = DATASET2_DIR,
        *,
        augment: bool = True,
        augment_config: AugmentConfig | None = None,
        max_cache_bytes: int = 8 * 1024 ** 3,
        prescan_workers: int = 8,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.entries, self.chunks, self.chunk_size = _load_index(self.dataset_dir)
        self.augment = augment
        self.augment_config = augment_config or AugmentConfig()
        self._cache = _LRUChunkCache(max_cache_bytes)

        self._chunk_by_char = [0] * NUM_CHARS
        self._char_offset_in_chunk = [0] * NUM_CHARS
        for chunk_pos, chunk in enumerate(self.chunks):
            for char_idx in range(chunk.char_start, chunk.char_end + 1):
                self._chunk_by_char[char_idx] = chunk_pos
                self._char_offset_in_chunk[char_idx] = char_idx - chunk.char_start

        self._valid_index: list[tuple[int, int]] = []
        self._flat_indices_by_char: list[list[int]] = [[] for _ in range(NUM_CHARS)]
        self._flat_indices_by_font: list[list[int]] = [[] for _ in self.entries]
        self._prescan(prescan_workers)

    @property
    def num_font_classes(self) -> int:
        return len(self.entries)

    def flat_indices_by_char(self) -> list[list[int]]:
        return self._flat_indices_by_char

    def flat_indices_by_font(self) -> list[list[int]]:
        return self._flat_indices_by_font

    def _prescan(self, workers: int) -> None:
        n = len(self.chunks)
        if workers <= 1:
            for chunk_pos in range(n):
                self._register_chunk(chunk_pos, self._decode_chunk(chunk_pos))
                self._log_prescan_progress(chunk_pos + 1, n)
            return

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk_pos, arr in enumerate(pool.map(self._decode_chunk, range(n))):
                self._register_chunk(chunk_pos, arr)
                self._log_prescan_progress(chunk_pos + 1, n)

    @staticmethod
    def _log_prescan_progress(done: int, total: int) -> None:
        if done == total or done % 20 == 0:
            print(f"[CharGroupedFontDataset] scanned {done}/{total} chunk(s)...")

    def _decode_chunk(self, chunk_pos: int) -> np.ndarray:
        chunk = self.chunks[chunk_pos]
        image = Image.open(self.dataset_dir / chunk.file)
        image.load()
        arr = np.array(image, dtype=np.uint8)

        expected_shape = (
            len(self.entries) * CHAR_SIZE,
            chunk.char_count * CHAR_SIZE,
        )
        if arr.shape != expected_shape:
            raise ValueError(
                f"{chunk.file}: unexpected image shape {arr.shape}, "
                f"expected {expected_shape}"
            )

        return (
            arr.reshape(len(self.entries), CHAR_SIZE, chunk.char_count, CHAR_SIZE)
            .transpose(0, 2, 1, 3)
            .copy()
        )

    def _register_chunk(self, chunk_pos: int, arr: np.ndarray) -> None:
        mins = arr.reshape(len(self.entries), arr.shape[1], CHAR_SIZE * CHAR_SIZE).min(axis=2)
        chunk = self.chunks[chunk_pos]
        for font_pos, char_offset in np.argwhere(mins < 255):
            char_idx = chunk.char_start + int(char_offset)
            font_pos = int(font_pos)
            flat_idx = len(self._valid_index)
            self._valid_index.append((font_pos, char_idx))
            self._flat_indices_by_char[char_idx].append(flat_idx)
            self._flat_indices_by_font[font_pos].append(flat_idx)
        self._cache.put(chunk_pos, arr)

    def __len__(self) -> int:
        return len(self._valid_index)

    def _get_chunk_array(self, chunk_pos: int) -> np.ndarray:
        arr = self._cache.get(chunk_pos)
        if arr is None:
            arr = self._decode_chunk(chunk_pos)
            self._cache.put(chunk_pos, arr)
        return arr

    def __getitem__(self, idx: int) -> dict:
        font_pos, char_idx = self._valid_index[idx]
        chunk_pos = self._chunk_by_char[char_idx]
        char_offset = self._char_offset_in_chunk[char_idx]
        glyph = self._get_chunk_array(chunk_pos)[font_pos, char_offset]

        if self.augment:
            image = _augment(glyph, self.augment_config)
        else:
            image = torch.from_numpy(glyph).to(torch.float32).div_(255.0).unsqueeze(0)

        entry = self.entries[font_pos]
        char = HANGUL_TABLE[char_idx]
        cho, jung, jong = decompose_hangul_syllable(char)

        return {
            "image": image,
            "font_label": torch.tensor(entry["id"] - 1, dtype=torch.long),
            "cho_label": torch.tensor(cho, dtype=torch.long),
            "jung_label": torch.tensor(jung, dtype=torch.long),
            "jong_label": torch.tensor(jong, dtype=torch.long),
            "font_id": entry["id"],
            "char_index": char_idx,
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = _LRUChunkCache(self._cache.max_bytes)
        state.pop("_flat_indices_by_char", None)
        state.pop("_flat_indices_by_font", None)
        return state


def _augment(glyph: np.ndarray, cfg: AugmentConfig) -> torch.Tensor:
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
    arr = (image.squeeze(0).clamp(0.0, 1.0) * 255).round().to(torch.uint8).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(decoded).unsqueeze(0)


def _random_erasing(image: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
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
