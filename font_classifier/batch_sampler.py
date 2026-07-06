"""폰트 단위로 배치를 구성하는 `FontGroupBatchSampler`.

`FontGlyphDataset`(dataset_loader.py)의 LRU 캐시는 짧은 시간 안에 같은
폰트가 반복 조회되어야 의미가 있다. 완전히 무작위인 `shuffle=True`는
폰트가 ~3,480개나 되는 이 데이터셋에서 배치 하나마다 사실상 전부 다른
폰트를 요구해 캐시 적중률을 0%에 가깝게 만든다 - 자세한 근거는
`docs/batch-sampler.md` 1절 참고.

이 샘플러는 한 번에 `fonts_per_batch`(K)개의 폰트만 활성화해 여러 배치에
걸쳐 재사용하고, 폰트마다 `chars_per_font`(M)개씩 글자를 뽑아 배치를
구성한다 - metric learning에서 흔히 쓰는 "PK 샘플링"(P개 클래스 x K개
샘플로 배치를 구성하는 방식)과 같은 원리다. 폰트 분류 손실의 클래스
다양성은 K(배치당 서로 다른 폰트 수)가 담당하고, 캐시 적중률은 폰트
그룹이 여러 배치에 걸쳐 재사용되는 동안 자연히 확보된다. 다만 같은 그룹을
끝까지 연속 소모하면 폰트 헤드가 그 소수 클래스에 과도하게 적응할 수 있어,
실제 배치는 "소수의 그룹을 동시에 활성화해 라운드 단위로 interleave"하는
방식으로 섞는다. M 자체는 캐시 적중률과 무관하다(폰트가 캐시에 남아 있는
한, M과 상관없이 그 폰트의 디코딩 1회가 유효 글자 전체에 상각된다).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

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
# - GROUPS_IN_FLIGHT(W)=4: 동시에 4개 폰트 묶음을 활성화해, 같은 묶음을
#   너무 오래 연속 소모하지 않으면서도 총 4*K=128개 폰트만 작업 집합으로
#   유지해 캐시 지역성을 지킨다.
DEFAULT_FONTS_PER_BATCH = 32
DEFAULT_CHARS_PER_FONT = 8
DEFAULT_GROUPS_IN_FLIGHT = 4


@dataclass
class _GroupState:
    rounds: list[list[list[int]]]
    round_idx: int = 0


class FontGroupBatchSampler(Sampler[list[int]]):
    """`FontGlyphDataset`과 함께 `DataLoader(dataset, batch_sampler=...)`로
    쓰는 배치 샘플러.

    매 epoch(`__iter__` 호출)마다:

    1. 유효 글자가 하나 이상 있는 폰트를 무작위로 섞어 `fonts_per_batch`
       (K)개씩 묶는다(마지막 묶음은 `drop_last=False`면 K보다 작을 수 있다).
    2. 각 묶음 안에서, 폰트마다 자신의 유효 글자 인덱스를 무작위로 섞고
       `chars_per_font`(M)개씩 잘라 "라운드"를 만든다.
    3. 묶음 몇 개(`groups_in_flight`)를 동시에 활성화한 뒤, 각 묶음의
       라운드를 round-robin으로 번갈아 내보낸다. 즉 같은 K개 폰트는 여러
       라운드에 걸쳐 재사용되지만, 다음 배치가 곧바로 같은 폰트 묶음일
       필요는 없다.

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
        groups_in_flight: int = DEFAULT_GROUPS_IN_FLIGHT,
        drop_last: bool = False,
    ) -> None:
        if fonts_per_batch <= 0 or chars_per_font <= 0 or groups_in_flight <= 0:
            raise ValueError(
                "fonts_per_batch, chars_per_font, and groups_in_flight must be positive"
            )

        self.fonts_per_batch = fonts_per_batch
        self.chars_per_font = chars_per_font
        self.groups_in_flight = groups_in_flight
        self.drop_last = drop_last

        self._flat_indices_by_font = dataset.flat_indices_by_font()
        self._font_positions = [
            pos for pos, indices in enumerate(self._flat_indices_by_font) if indices
        ]

    def _build_group_state(self, group: list[int]) -> _GroupState:
        rounds: list[list[list[int]]] = []
        for font_pos in group:
            indices = self._flat_indices_by_font[font_pos][:]
            random.shuffle(indices)
            rounds.append([
                indices[i:i + self.chars_per_font]
                for i in range(0, len(indices), self.chars_per_font)
            ])
        return _GroupState(rounds=rounds)

    def __iter__(self) -> Iterator[list[int]]:
        k = self.fonts_per_batch

        font_order = self._font_positions[:]
        random.shuffle(font_order)
        groups = [
            font_order[group_start:group_start + k]
            for group_start in range(0, len(font_order), k)
        ]

        active_groups: list[_GroupState] = []
        next_group_idx = 0
        while next_group_idx < len(groups) and len(active_groups) < self.groups_in_flight:
            active_groups.append(self._build_group_state(groups[next_group_idx]))
            next_group_idx += 1

        while active_groups:
            next_active_groups: list[_GroupState] = []
            for state in active_groups:
                round_idx = state.round_idx
                batch = [
                    flat_idx
                    for chunks in state.rounds
                    if round_idx < len(chunks)
                    for flat_idx in chunks[round_idx]
                ]
                if not self.drop_last or len(batch) == k * self.chars_per_font:
                    yield batch

                round_idx += 1
                if any(round_idx < len(chunks) for chunks in state.rounds):
                    state.round_idx = round_idx
                    next_active_groups.append(state)
                elif next_group_idx < len(groups):
                    next_active_groups.append(self._build_group_state(groups[next_group_idx]))
                    next_group_idx += 1
            active_groups = next_active_groups

    def __len__(self) -> int:
        """배치 개수의 근사치. interleave를 해도 총 배치 수 자체는
        "몇 개의 폰트 묶음이 있고, 묶음당 평균 몇 라운드가 생기는가"로
        근사할 수 있다. 실제 개수는 폰트별 유효 글자 수 편차와 무작위 묶음
        구성에 따라 조금씩 달라질 수 있어 정확한 값이 아니라 추정치다
        (진행률 표시줄 등 용도로 충분하다)."""

        if not self._font_positions:
            return 0

        num_groups = -(-len(self._font_positions) // self.fonts_per_batch)
        avg_chars = sum(
            len(self._flat_indices_by_font[p]) for p in self._font_positions
        ) / len(self._font_positions)
        rounds_per_group = max(1, -(-round(avg_chars) // self.chars_per_font))
        return num_groups * rounds_per_group
