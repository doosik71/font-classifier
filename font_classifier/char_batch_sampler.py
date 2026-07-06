"""글자 단위로 배치를 구성하는 `CharGroupBatchSampler`.

`dataset_loader_2.py`의 char-major dataset은 같은 글자를 여러 폰트로 묶어
읽을 때 캐시 지역성이 좋아지고, 배치 안에서 content(글자)를 상대적으로
고정한 채 style(폰트) 차이를 더 직접적으로 보게 할 수 있다.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from torch.utils.data import Sampler

from .dataset_loader_2 import CharGroupedFontDataset

DEFAULT_CHARS_PER_BATCH = 8
DEFAULT_FONTS_PER_CHAR = 32


class CharGroupBatchSampler(Sampler[list[int]]):
    """`CharGroupedFontDataset`과 함께 쓰는 배치 샘플러.

    매 배치는 `chars_per_batch`개의 서로 다른 글자를 고르고, 각 글자마다
    `fonts_per_char`개까지 서로 다른 폰트 샘플을 뽑아 이어붙인다. 즉 같은
    배치 안에 "같은 글자, 다른 폰트" 쌍이 다수 들어가도록 만든다.
    """

    def __init__(
        self,
        dataset: CharGroupedFontDataset,
        chars_per_batch: int = DEFAULT_CHARS_PER_BATCH,
        fonts_per_char: int = DEFAULT_FONTS_PER_CHAR,
        drop_last: bool = False,
    ) -> None:
        if chars_per_batch <= 0 or fonts_per_char <= 0:
            raise ValueError("chars_per_batch and fonts_per_char must be positive")

        self.chars_per_batch = chars_per_batch
        self.fonts_per_char = fonts_per_char
        self.drop_last = drop_last
        self._flat_indices_by_char = dataset.flat_indices_by_char()
        self._char_positions = [
            char_idx
            for char_idx, indices in enumerate(self._flat_indices_by_char)
            if indices
        ]

    def __iter__(self) -> Iterator[list[int]]:
        char_order = self._char_positions[:]
        random.shuffle(char_order)
        groups = [
            char_order[group_start:group_start + self.chars_per_batch]
            for group_start in range(0, len(char_order), self.chars_per_batch)
        ]

        for group in groups:
            rounds = []
            max_rounds = 0
            for char_idx in group:
                indices = self._flat_indices_by_char[char_idx][:]
                random.shuffle(indices)
                chunks = [
                    indices[i:i + self.fonts_per_char]
                    for i in range(0, len(indices), self.fonts_per_char)
                ]
                rounds.append(chunks)
                max_rounds = max(max_rounds, len(chunks))

            for round_idx in range(max_rounds):
                batch = [
                    flat_idx
                    for chunks in rounds
                    if round_idx < len(chunks)
                    for flat_idx in chunks[round_idx]
                ]
                if not self.drop_last or len(batch) == self.chars_per_batch * self.fonts_per_char:
                    yield batch

    def __len__(self) -> int:
        if not self._char_positions:
            return 0

        num_groups = -(-len(self._char_positions) // self.chars_per_batch)
        avg_fonts = sum(
            len(self._flat_indices_by_char[char_idx]) for char_idx in self._char_positions
        ) / len(self._char_positions)
        rounds_per_group = max(1, -(-round(avg_fonts) // self.fonts_per_char))
        return num_groups * rounds_per_group
