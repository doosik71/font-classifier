"""`data/dataset`(construct-dataset.py가 만든 폰트별 PNG + index.json)을
모델 학습용 PyTorch `Dataset`으로 읽어 들이는 모듈.

핵심 설계는 세 가지다.

1. **폰트 단위 LRU 메모리 캐시**: 폰트 PNG(약 9.18MiB/폰트, 디코딩 상태)를
   한 번 읽으면 바이트 예산 안에서 메모리에 남겨 두고 재사용한다. 무작위로
   섞인 (폰트, 글자) 쌍을 매번 디스크에서 다시 읽고 PNG를 다시 디코딩하면
   글자 하나를 위해 9MB짜리 파일 전체를 매번 여는 셈이라 학습 속도의 병목이
   되기 때문이다.
2. **메모리 안에서 바로 적용하는 augmentation**: 디스크에 저장하지 않고
   `__getitem__` 호출마다 즉석에서 변형을 적용한다.
3. **기하 변환은 패딩 후 크롭**: `char_extract.py`가 글자의 긴 변을 정확히
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
from .font_dataset import HANGUL_TABLE, SCAN_DIR, decompose_hangul_syllable

DATASET_DIR = SCAN_DIR.parent / "dataset"
NUM_CHARS = len(HANGUL_TABLE)

# 폰트 하나(2,350 x 64 x 64 uint8)의 디코딩된 크기: 약 9.18MiB.
BYTES_PER_FONT = NUM_CHARS * CHAR_SIZE * CHAR_SIZE

# LRU 캐시 기본 예산: 폰트 약 400개 분량(~3.85GiB). `DataLoader(num_workers=N)`
# 는 워커마다 별도 프로세스와 별도 캐시를 가지므로(2.5절), 실제 총 메모리는
# 이 값의 최대 N배가 된다.
DEFAULT_MAX_CACHE_BYTES = 400 * BYTES_PER_FONT

# 초기화 시 폰트별 유효 글자 마스크를 만들기 위해 전체 PNG를 한 번씩 읽는
# 사전 스캔(prescan)에서 사용할 스레드 수. PIL의 PNG 압축 해제가 GIL을 일부
# 해제하므로 스레드로도 어느 정도 속도가 붙는다(2.2절).
DEFAULT_PRESCAN_WORKERS = 8


@dataclass
class AugmentConfig:
    """augmentation 강도/확률 설정. 기본값은 docs/model-design.md 5.3절의
    "공통(약함)" 세트에 해당한다 — font 분류 손실에도 안전하게 쓸 수 있는
    수준으로, 획 두께를 크게 바꾸는 변형(dilation/erosion)이나 사진 도메인
    전용 변형(그림자, motion blur 등)은 포함하지 않는다(자세한 근거는
    docs/dataset-loader.md 3절 참고).
    """

    # 1) 기하 변환 - translate/rotate/scale/shear를 하나의 affine으로 합성해
    #    한 번만 리샘플링한다(3.1절).
    max_translate_px: float = 4.0
    max_rotate_deg: float = 5.0
    scale_range: tuple[float, float] = (0.92, 1.08)
    max_shear_deg: float = 5.0

    # 2) 광도 - 매번 약하게 적용
    brightness_range: tuple[float, float] = (0.85, 1.15)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    gamma_range: tuple[float, float] = (0.85, 1.15)

    # 3) 흐림/샤픈 - 둘 중 하나만, 확률적으로 적용
    blur_prob: float = 0.15
    blur_sigma_range: tuple[float, float] = (0.3, 0.8)
    sharpen_prob: float = 0.15
    sharpen_factor_range: tuple[float, float] = (1.5, 2.5)

    # 4) 가우시안 노이즈
    noise_prob: float = 0.5
    noise_std_range: tuple[float, float] = (0.01, 0.05)  # [0,1] 스케일 기준

    # 5) JPEG 압축 흉내(실제 인코딩 왕복)
    jpeg_prob: float = 0.3
    jpeg_quality_range: tuple[int, int] = (30, 80)

    # 6) random erasing(흰/검 마스킹, cutout)
    erasing_prob: float = 0.3
    erasing_count_range: tuple[int, int] = (1, 2)
    erasing_size_range: tuple[float, float] = (0.05, 0.2)  # 한 변 길이 비율


def _geometric_pad(cfg: AugmentConfig) -> int:
    """기하 변환(이동/회전/확대/기울이기)을 적용해도 획이 잘리지 않도록
    필요한 패딩 폭(px)을 넉넉하게 계산한다. 정확한 최솟값이 아니라
    여유 있는 상한이면 충분하다 — 패딩을 몇 픽셀 더 쓴다고 품질/성능에
    의미 있는 차이가 없기 때문이다."""

    half = CHAR_SIZE / 2
    angle_margin = half * math.sin(math.radians(
        cfg.max_rotate_deg + cfg.max_shear_deg))
    scale_margin = half * max(0.0, cfg.scale_range[1] - 1.0)
    return math.ceil(cfg.max_translate_px + angle_margin + scale_margin) + 2


class _LRUFontCache:
    """폰트 하나(디코딩된 numpy 배열) 단위로 관리하는 바이트 예산 기반 LRU
    캐시. `OrderedDict`로 최근 사용 순서를 유지하며, 예산을 넘으면 가장
    오래전에 쓰인 폰트부터 내린다."""

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
        self._data[key] = arr
        self._data.move_to_end(key)
        self._nbytes += arr.nbytes
        while self._nbytes > self.max_bytes and len(self._data) > 1:
            _, evicted = self._data.popitem(last=False)
            self._nbytes -= evicted.nbytes


def _load_index(dataset_dir: Path) -> list[dict]:
    index_path = dataset_dir / "index.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))

    ids = sorted(entry["id"] for entry in entries)
    if ids != list(range(1, len(entries) + 1)):
        raise ValueError(
            f"{index_path}: 'id' must form a contiguous 1..N range "
            f"(N={len(entries)}), got e.g. {ids[:5]}"
        )

    entries.sort(key=lambda entry: entry["id"])
    return entries


class FontGlyphDataset(Dataset):
    """`data/dataset`의 폰트별 PNG에서 완성형 한글 낱글자 샘플을 읽어
    (augmentation이 적용된) 이미지와 폰트/초성/중성/종성 라벨을 반환하는
    PyTorch `Dataset`.

    빈 칸(annotation 없음/추출 실패/폰트가 그 글자를 지원하지 않음)은
    `construct-dataset.py`가 모두 흰 배경으로 남겨 두므로(2.1절),
    생성자에서 폰트별로 전체가 흰색인 칸을 걸러내 표본 공간에서 제외한다.
    """

    def __init__(
        self,
        dataset_dir: str | Path = DATASET_DIR,
        *,
        augment: bool = True,
        augment_config: AugmentConfig | None = None,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        prescan_workers: int = DEFAULT_PRESCAN_WORKERS,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.entries = _load_index(self.dataset_dir)
        self.augment = augment
        self.augment_config = augment_config or AugmentConfig()
        self._cache = _LRUFontCache(max_cache_bytes)

        self._valid_index: list[tuple[int, int]] = []
        self._flat_indices_by_font: list[list[int]] = [[] for _ in self.entries]
        self._prescan(prescan_workers)

    @property
    def num_font_classes(self) -> int:
        return len(self.entries)

    def flat_indices_by_font(self) -> list[list[int]]:
        """폰트 위치(`font_pos`)별로, 그 폰트에 속한 유효 글자의 `__getitem__`
        인덱스(=`_valid_index`에서의 위치) 목록을 반환한다.
        `FontGroupBatchSampler`(batch_sampler.py)가 폰트 단위로 배치를
        구성할 때 이 정보를 쓴다."""

        return self._flat_indices_by_font

    # ------------------------------------------------------------ 초기화 --
    def _prescan(self, workers: int) -> None:
        """폰트마다 PNG를 한 번 읽어 유효(비어 있지 않은) 글자 칸 목록을
        만든다. 이 과정에서 읽은 배열은 그대로 LRU 캐시에 넣어 두므로(예산
        범위 안에서) 첫 학습 스텝부터 일부는 이미 캐시가 데워져 있다."""

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

    def _decode_font(self, font_pos: int) -> np.ndarray:
        """디스크에서 폰트 PNG를 읽어 (NUM_CHARS, 64, 64) uint8 배열로
        디코딩한다. 캐시를 건드리지 않는 순수 함수라 스레드 풀에서 안전하게
        병렬 호출할 수 있다."""

        entry = self.entries[font_pos]
        image = Image.open(self.dataset_dir / entry["file"])
        image.load()
        # `np.array`(복사)를 쓴다 - `np.asarray`는 PIL의 내부 버퍼를 읽기
        # 전용 뷰로 그대로 감쌀 수 있어, 이후 `torch.from_numpy`가 매번
        # "non-writable" 경고를 낸다. 캐시에 넣고 오래 재사용할 배열이니
        # 한 번의 복사 비용을 들여 독립된 쓰기 가능 버퍼로 만든다.
        arr = np.array(image, dtype=np.uint8)

        expected_shape = (CHAR_SIZE * NUM_CHARS, CHAR_SIZE)
        if arr.shape != expected_shape:
            raise ValueError(
                f"{entry['file']}: unexpected image shape {arr.shape}, "
                f"expected {expected_shape}"
            )
        return arr.reshape(NUM_CHARS, CHAR_SIZE, CHAR_SIZE)

    def _register_font(self, font_pos: int, arr: np.ndarray) -> None:
        mins = arr.reshape(NUM_CHARS, CHAR_SIZE * CHAR_SIZE).min(axis=1)
        flat_indices = self._flat_indices_by_font[font_pos]
        for char_idx in np.nonzero(mins < 255)[0].tolist():
            flat_indices.append(len(self._valid_index))
            self._valid_index.append((font_pos, char_idx))
        self._cache.put(font_pos, arr)

    # -------------------------------------------------------- Dataset API --
    def __len__(self) -> int:
        return len(self._valid_index)

    def _get_font_array(self, font_pos: int) -> np.ndarray:
        arr = self._cache.get(font_pos)
        if arr is None:
            arr = self._decode_font(font_pos)
            self._cache.put(font_pos, arr)
        return arr

    def __getitem__(self, idx: int) -> dict:
        font_pos, char_idx = self._valid_index[idx]
        glyph = self._get_font_array(font_pos)[char_idx]  # (64,64) uint8 view

        if self.augment:
            image = _augment(glyph, self.augment_config)
        else:
            # `.to(torch.float32)`는 dtype이 바뀌므로 항상 새 텐서를
            # 만든다 - 캐시에 있는 원본 uint8 배열은 절대 수정하지 않는다.
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

    # `DataLoader(num_workers>0)`는 워커 프로세스를 만들 때 이 객체를
    # pickle한다. 캐시에 든 numpy 배열까지 그대로 직렬화해 보내면 워커마다
    # 불필요하게 큰 페이로드를 복사하게 되므로, 워커는 항상 빈 캐시로
    # 시작하게 한다(2.5절). `_flat_indices_by_font`는 `FontGroupBatchSampler`
    # (메인 프로세스에서만 실행됨)만 쓰므로 워커에는 보내지 않는다 -
    # `_valid_index`와 크기가 같은 구조라 그대로 보내면 워커마다 그만큼
    # 불필요한 pickle 비용이 두 배가 된다.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = _LRUFontCache(self._cache.max_bytes)
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
