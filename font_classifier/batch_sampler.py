"""폰트 단위로 배치를 구성하는 `FontGroupBatchSampler`.

`FontGlyphDataset`(dataset_loader.py)의 LRU 캐시는 짧은 시간 안에 같은
폰트가 반복 조회되어야 의미가 있다. 완전히 무작위인 `shuffle=True`는
폰트가 ~3,480개나 되는 이 데이터셋에서 배치 하나마다 사실상 전부 다른
폰트를 요구해 캐시 적중률을 0%에 가깝게 만든다 — 자세한 근거는
docs/batch-sampler.md 1절 참고.

이 샘플러는 한 번에 `fonts_per_batch`(K)개의 폰트만 활성화해 여러 배치에
걸쳐 재사용하고, 폰트마다 `chars_per_font`(M)개씩 글자를 뽑아 배치를
구성한다 — metric learning에서 흔히 쓰는 "PK 샘플링"(P개 클래스 x K개
샘플로 배치를 구성하는 방식)과 같은 원리다. 폰트 분류 손실의 클래스
다양성은 K(배치당 서로 다른 폰트 수)가 담당하고, 캐시 적중률은 폰트
그룹이 여러 배치에 걸쳐 재사용되는 동안 자연히 확보된다 - M 자체는 캐시
적중률과 무관하다(폰트가 캐시에 남아 있는 한, M과 상관없이 그 폰트의
디코딩 1회가 유효 글자 전체에 상각된다).
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from torch.utils.data import Sampler

from .dataset_loader import FontGlyphDataset

# 폰트 약 3,480개, 폰트당 최대 2,350자라는 실제 규모를 기준으로 고른 기본값
# (docs/batch-sampler.md 2절에 근거를 정리했다).
# - FONTS_PER_BATCH(K)=32: 배치마다 서로 다른 폰트 클래스 32개(전체
#   3,480종의 약 0.9%)가 섞여, softmax 폰트 분류 손실의 그래디언트가 매
#   스텝 소수 클래스에 치우치지 않게 한다.
# - CHARS_PER_FONT(M)=8: 배치 크기 K*M=256을 만들고, 폰트 하나당 같은
#   배치 안에 C(8,2)=28쌍의 동일 폰트 쌍을 확보해 향후 대조학습/재구성
#   손실(model-design.md 4.2~4.4절)에도 바로 쓸 수 있게 여유를 둔다.
DEFAULT_FONTS_PER_BATCH = 32
DEFAULT_CHARS_PER_FONT = 8


class FontGroupBatchSampler(Sampler[list[int]]):
    """`FontGlyphDataset`과 함께 `DataLoader(dataset, batch_sampler=...)`로
    쓰는 배치 샘플러.

    매 epoch(`__iter__` 호출)마다:

    1. 유효 글자가 하나 이상 있는 폰트를 무작위로 섞어 `fonts_per_batch`
       (K)개씩 묶는다(마지막 묶음은 `drop_last=False`면 K보다 작을 수 있다).
    2. 각 묶음 안에서, 폰트마다 자신의 유효 글자 인덱스를 무작위로 섞고
       `chars_per_font`(M)개씩 잘라 "라운드"를 만든다.
    3. 라운드 순서대로, 그 라운드에 아직 글자가 남은 모든 폰트의 몫을 모아
       배치 하나로 만든다 — 즉 같은 K개 폰트가 여러 라운드(배치)에 걸쳐
       연속으로 재사용된다.

    `drop_last=False`(기본값)면 마지막 폰트 묶음이나 폰트별 마지막
    라운드가 K*M보다 작을 수 있다(배치 크기가 가변적이지만 데이터를
    하나도 버리지 않는다). `drop_last=True`면 정확히 K*M 크기인 배치만
    내보낸다.
    """

    def __init__(
        self,
        dataset: FontGlyphDataset,
        fonts_per_batch: int = DEFAULT_FONTS_PER_BATCH,
        chars_per_font: int = DEFAULT_CHARS_PER_FONT,
        drop_last: bool = False,
    ) -> None:
        if fonts_per_batch <= 0 or chars_per_font <= 0:
            raise ValueError("fonts_per_batch and chars_per_font must be positive")

        self.fonts_per_batch = fonts_per_batch
        self.chars_per_font = chars_per_font
        self.drop_last = drop_last

        self._flat_indices_by_font = dataset.flat_indices_by_font()
        self._font_positions = [
            pos for pos, indices in enumerate(self._flat_indices_by_font) if indices
        ]

    def __iter__(self) -> Iterator[list[int]]:
        k, m = self.fonts_per_batch, self.chars_per_font

        font_order = self._font_positions[:]
        random.shuffle(font_order)

        for group_start in range(0, len(font_order), k):
            group = font_order[group_start:group_start + k]

            rounds: list[list[list[int]]] = []
            for font_pos in group:
                indices = self._flat_indices_by_font[font_pos][:]
                random.shuffle(indices)
                rounds.append([indices[i:i + m] for i in range(0, len(indices), m)])

            max_rounds = max(len(r) for r in rounds)
            for round_idx in range(max_rounds):
                batch = [
                    flat_idx
                    for chunks in rounds
                    if round_idx < len(chunks)
                    for flat_idx in chunks[round_idx]
                ]
                if not self.drop_last or len(batch) == k * m:
                    yield batch

    def __len__(self) -> int:
        """배치 개수의 근사치. 폰트 묶음 하나가 만드는 배치(라운드) 수는
        그 묶음의 폰트 수가 아니라 폰트당 유효 글자 수로 정해지므로(K보다
        작은 마지막 묶음도 똑같은 라운드 수를 만든다), 단순히
        "전체 유효 글자 수 / (K*M)"으로 나누면 크게 틀린다 - 묶음 수 x
        묶음당 평균 라운드 수로 근사한다. 실제 개수는 폰트별 유효 글자 수
        편차와 무작위 묶음 구성에 따라 조금씩 달라질 수 있어 정확한 값이
        아니라 추정치다(진행률 표시줄 등 용도로 충분하다)."""

        if not self._font_positions:
            return 0

        num_groups = -(-len(self._font_positions) // self.fonts_per_batch)
        avg_chars = sum(
            len(self._flat_indices_by_font[p]) for p in self._font_positions
        ) / len(self._font_positions)
        rounds_per_group = max(1, -(-round(avg_chars) // self.chars_per_font))
        return num_groups * rounds_per_group
