r"""한글 폰트 인식 모델. 설계 근거는 `docs/model-design.md` 3절("아키텍처")을
그대로 따르며, 이 파일은 그 절에서 설명한 구성 요소(인코더, content/style
투영, 초중종성 헤더, 폰트 헤더, 디코더, 초중종성 조합/디코딩)를 전부
담는다.

```text
입력 글자 영상 (1 x 64 x 64, grayscale)
        |
   공유 Encoder (ResNet 스타일, stage1~5)
        |
   Global Average Pooling
     /        \
content_proj   style_proj
    |              |
content code    style code
 (128-dim)      (512-dim, L2 정규화)
    |              |
 Hangul head     Font head
 cho/jung/jong    font logits + embedding
    |              |
    \____________ /
       concat(content, style)
             |
          Decoder (cross skip 없음)
             |
     복원 영상 (1 x 64 x 64)
```

`encode()`/`decode()`를 별도 공개 메서드로 나눈 이유는
model-design.md 4.2절의 모드 B(factorized cross-recombination)가 서로
다른 두 이미지에서 각각 content/style을 뽑아 조합해야 하기 때문이다 -
학습 스크립트가 `encode(구조_소스)`와 `encode(스타일_소스)`를 따로 호출한
뒤 `decode(content, style)`로 섞어 부를 수 있어야 한다. `forward()`는
같은 이미지를 encode한 뒤 그대로 decode하는(모드 A/평가에 쓰는) 편의
메서드일 뿐이다.

자세한 설계 근거(정규화 레이어 선택, MLP 은닉 차원 등 이 문서에 없는
세부사항에 대한 가정 포함)는 docs/model.md 참고.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .char_extract import CHAR_SIZE
from .font_dataset import (
    HANGUL_TABLE, NUM_CHO, NUM_JONG, NUM_JUNG, decompose_hangul_syllable,
)

CONTENT_DIM = 128
STYLE_DIM = 512
POOLED_DIM = 512  # stage5 채널 수 = GlobalAvgPool 직후 벡터 차원

# GroupNorm 그룹 수. 인코더/디코더에서 쓰는 모든 채널 수(32/64/128/256/512)를
# 나누어떨어지게 하는 값이다. model-design.md 3.2절이 "배치가 작아질
# 가능성이 있으면 GroupNorm을 우선 검토"하라고 명시했고, 이 프로젝트의
# FontGroupBatchSampler(batch_sampler.py)는 폰트 묶음 경계에서 배치 크기가
# 작아질 수 있다고 문서화되어 있어(docs/batch-sampler.md 1.3절), BatchNorm
# 대신 GroupNorm을 기본으로 택했다.
_GROUP_NORM_GROUPS = 32


def _norm(num_channels: int) -> nn.Module:
    return nn.GroupNorm(min(_GROUP_NORM_GROUPS, num_channels), num_channels)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class ResidualBlock(nn.Module):
    """표준 ResNet residual block(conv-norm-act-conv-norm + shortcut).
    `stride=2`면 첫 conv에서 다운샘플하고, 채널/해상도가 바뀌면 shortcut도
    1x1 conv + norm으로 맞춘다(model-design.md 3.2절)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                                padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        # 잔차 분기의 마지막 norm을 0으로 초기화하는 표준 ResNet 기법("zero-init
        # the last BN in each residual branch") - 학습 초반에는 각 블록이
        # 항등함수에 가깝게 시작해 아주 깊은 네트워크의 초기 안정성을 돕는다.
        nn.init.zeros_(self.norm2.weight)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class UpsampleResidualBlock(nn.Module):
    """디코더용 residual block. Transposed conv 대신 nearest-neighbor
    upsample + conv를 써서 체커보드 아티팩트를 피한다(model-design.md
    3.5절 - 표준적인 관행이며 문서에 별도로 명시되어 있지는 않다)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        nn.init.zeros_(self.norm2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class Encoder(nn.Module):
    """model-design.md 3.2절의 stage 표를 그대로 구현한다.

    | stage  | 해상도 | 채널 | 블록                     |
    | ------ | ------ | ---- | ------------------------ |
    | stem   | 64x64  | 32   | conv3x3                  |
    | stage1 | 64x64  | 32   | residual x2              |
    | stage2 | 32x32  | 64   | residual x2 (downsample) |
    | stage3 | 16x16  | 128  | residual x2 (downsample) |
    | stage4 | 8x8    | 256  | residual x3 (downsample) |
    | stage5 | 4x4    | 512  | residual x3 (downsample) |
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            _norm(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 32))
        self.stage2 = nn.Sequential(ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64))
        self.stage3 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256), ResidualBlock(256, 256),
        )
        self.stage5 = nn.Sequential(
            ResidualBlock(256, 512, stride=2), ResidualBlock(512, 512), ResidualBlock(512, 512),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x  # (B, 512, 4, 4)


class ProjectionHeads(nn.Module):
    """`pooled`(512-dim)에서 content(128-dim)와 style(512-dim, L2 정규화)
    code를 뽑는다(model-design.md 3.3절). 은닉 차원은 문서에 명시되지 않아
    이 파일에서 정한 가정이다 - content는 68개(19+21+28) 조합 공간으로
    압축해 가는 길목이라 절반씩 줄이고, style은 최종 차원(512)을 그대로
    유지한다."""

    def __init__(self, pooled_dim: int = POOLED_DIM,
                 content_dim: int = CONTENT_DIM, style_dim: int = STYLE_DIM) -> None:
        super().__init__()
        self.content_proj = _mlp(pooled_dim, (pooled_dim + content_dim) // 2, content_dim)
        self.style_proj = _mlp(pooled_dim, style_dim, style_dim)

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        content = self.content_proj(pooled)
        style = F.normalize(self.style_proj(pooled), dim=-1)
        return content, style


class HangulHead(nn.Module):
    """content(128) -> 독립 MLP(128->64->N) 3개로 cho/jung/jong logits
    (model-design.md 3.6절 - 은닉 차원 64는 문서에 명시된 값)."""

    def __init__(self, content_dim: int = CONTENT_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cho = _mlp(content_dim, hidden_dim, NUM_CHO)
        self.jung = _mlp(content_dim, hidden_dim, NUM_JUNG)
        self.jong = _mlp(content_dim, hidden_dim, NUM_JONG)

    def forward(self, content: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.cho(content), self.jung(content), self.jong(content)


class FontHead(nn.Module):
    """정규화된 style(512)를 한 번 더 비선형 MLP로 풀어 준 뒤 폰트 logits을
    예측한다. docs/model-design-enhancement-strategy.md 4.1절의 1순위안으로,
    3,000개가 넘는 미세한 폰트 클래스를 단일 선형 경계 대신 더 깊은 헤드로
    분리하게 하고, 마지막 분류기에 들어가는 특징의 크기도 다시 자유롭게
    만들어 logit 동적 범위 포화를 완화한다."""

    def __init__(
        self,
        num_font_classes: int,
        style_dim: int = STYLE_DIM,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, style_dim),
            nn.LayerNorm(style_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(style_dim, num_font_classes)

    def forward(self, style: Tensor) -> Tensor:
        return self.classifier(self.mlp(style))


class Decoder(nn.Module):
    """concat(content, style) -> 4x4 -> 4단계 업샘플 -> 64x64 (model-design.md
    3.5절). Cross skip이 없으므로 인코더 feature를 전혀 받지 않는다 -
    모드 A/B(4.2절)가 완전히 같은 디코더를 공유할 수 있는 이유가 이것이다.
    출력은 `FontGlyphDataset`이 만드는 입력과 같은 [0,1] 범위에 놓이도록
    sigmoid를 쓴다(문서의 "sigmoid 또는 clipped grayscale" 중 sigmoid 선택)."""

    def __init__(self, content_dim: int = CONTENT_DIM, style_dim: int = STYLE_DIM) -> None:
        super().__init__()
        in_dim = content_dim + style_dim
        self.fc = _mlp(in_dim, in_dim, 512 * 4 * 4)
        self.up1 = UpsampleResidualBlock(512, 256)  # 4 -> 8
        self.up2 = UpsampleResidualBlock(256, 128)  # 8 -> 16
        self.up3 = UpsampleResidualBlock(128, 64)   # 16 -> 32
        self.up4 = UpsampleResidualBlock(64, 32)    # 32 -> 64
        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        z = torch.cat([content, style], dim=-1)
        x = self.fc(z).view(-1, 512, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return torch.sigmoid(self.out_conv(x))


@dataclass
class FontModelOutput:
    content: Tensor
    style: Tensor
    cho_logits: Tensor
    jung_logits: Tensor
    jong_logits: Tensor
    font_logits: Tensor
    reconstruction: Tensor | None = None


class FontRecognitionModel(nn.Module):
    """model-design.md 3절 전체를 하나로 묶은 최상위 모델.

    `num_font_classes`는 이 파일에 하드코딩하지 않고 호출자가 명시적으로
    넘긴다 - annotation 작업이 계속 진행 중이라 실제 폰트 수(현재 약
    3,480종)가 시간이 지나며 바뀌므로, `FontGlyphDataset.num_font_classes`
    (font_classifier/dataset_loader.py)에서 그대로 가져와 쓰는 것을
    가정한다.
    """

    def __init__(self, num_font_classes: int) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = ProjectionHeads()
        self.hangul_head = HangulHead()
        self.font_head = FontHead(num_font_classes)
        self.decoder = Decoder()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # ResidualBlock/UpsampleResidualBlock의 __init__이 이미 각자의 마지막
        # norm을 0으로 되돌려 두었으므로(zero-init residual), 여기서 다시
        # 덮어쓰지 않는다.

    def encode(self, x: Tensor) -> FontModelOutput:
        """글자 영상 하나(또는 배치)에서 content/style code와 초중종성/폰트
        logits을 뽑는다. 모드 B(model-design.md 4.2절)처럼 서로 다른 소스
        이미지에서 content/style을 따로 뽑아야 할 때 이 메서드를 두 번
        호출한 뒤 `decode()`로 섞는다."""

        if x.shape[-2:] != (CHAR_SIZE, CHAR_SIZE):
            raise ValueError(
                f"expected {CHAR_SIZE}x{CHAR_SIZE} input, got {tuple(x.shape[-2:])}"
            )

        features = self.encoder(x)
        pooled = self.pool(features).flatten(1)
        content, style = self.projection(pooled)
        cho_logits, jung_logits, jong_logits = self.hangul_head(content)
        font_logits = self.font_head(style)
        return FontModelOutput(
            content=content, style=style,
            cho_logits=cho_logits, jung_logits=jung_logits, jong_logits=jong_logits,
            font_logits=font_logits,
        )

    def decode(self, content: Tensor, style: Tensor) -> Tensor:
        """content/style code로부터 64x64 영상을 복원한다. `content`와
        `style`은 같은 이미지에서 나왔을 수도(모드 A), 서로 다른 이미지에서
        나왔을 수도 있다(모드 B) - 디코더에 cross skip이 없으므로
        (model-design.md 3.4절) 두 경우를 구분할 필요가 없다."""

        return self.decoder(content, style)

    def forward(self, x: Tensor) -> FontModelOutput:
        """`encode(x)`로 얻은 content/style을 그대로 `decode`까지 이어붙이는
        편의 메서드(모드 A/평가에 해당) - 학습 스크립트가 재구성 손실을
        쓰지 않는 단계(model-design.md 4.5절 Phase 1)라면 `encode()`만
        호출해 디코더 연산을 아예 건너뛸 수 있다."""

        output = self.encode(x)
        output.reconstruction = self.decode(output.content, output.style)
        return output


# --------------------------------------------------------------------------
# 초중종성 조합/디코딩 (model-design.md 3.7절)
# --------------------------------------------------------------------------

def compose_hangul_syllable(cho: int, jung: int, jong: int) -> str:
    """(초성, 중성, 종성) 인덱스를 완성형 한글 음절 하나로 조합한다.
    `font_dataset.decompose_hangul_syllable`의 역함수이며 같은 산술 공식을
    쓴다 - 학습 라벨 생성과 추론 디코딩이 항상 일치하도록 공식을 두 곳에
    나눠 두지 않는다(font_dataset.py가 분해를, 이 함수가 조합을 맡는다)."""

    code = 0xAC00 + (cho * NUM_JUNG + jung) * NUM_JONG + jong
    return chr(code)


# HANGUL_TABLE(2,350자)의 각 글자를 (초성, 중성, 종성) 인덱스로 미리
# 분해해 둔다 - HANGUL_TABLE은 모듈 로드 시점에 고정되므로,
# `decode_restricted`를 호출할 때마다(예: 매 검증 배치) 다시 계산할
# 필요가 없다.
_TABLE_CHO, _TABLE_JUNG, _TABLE_JONG = zip(
    *(decompose_hangul_syllable(ch) for ch in HANGUL_TABLE)
)


def decode_restricted(cho_logits: Tensor, jung_logits: Tensor, jong_logits: Tensor) -> list[str]:
    """제한 디코딩(model-design.md 3.7절 1번): KS X 1001 2,350자
    (`HANGUL_TABLE`) 위에서 `log P(cho) + log P(jung) + log P(jong)`이
    최대인 글자를 고른다. 학습 라벨과 직접 비교할 수 있어 평가 지표로
    쓴다. 입력은 배치 단위 logits(`(B, N)`), 반환은 배치 크기만큼의 글자
    리스트다."""

    log_p_cho = F.log_softmax(cho_logits, dim=-1)
    log_p_jung = F.log_softmax(jung_logits, dim=-1)
    log_p_jong = F.log_softmax(jong_logits, dim=-1)

    device = cho_logits.device
    cho_idx = torch.as_tensor(_TABLE_CHO, device=device)
    jung_idx = torch.as_tensor(_TABLE_JUNG, device=device)
    jong_idx = torch.as_tensor(_TABLE_JONG, device=device)

    # (B, 2350) = 표의 각 글자에 대한 결합 log-확률
    scores = (
        log_p_cho[:, cho_idx] + log_p_jung[:, jung_idx] + log_p_jong[:, jong_idx]
    )
    best = scores.argmax(dim=-1).tolist()
    return [HANGUL_TABLE[i] for i in best]


def decode_open(cho_logits: Tensor, jung_logits: Tensor, jong_logits: Tensor) -> list[str]:
    """개방 디코딩(model-design.md 3.7절 2번): cho/jung/jong 각각의 argmax를
    조합 공식에 그대로 대입해 11,172 음절 전체 중 하나를 얻는다. 2,350자
    표 밖의 글자도 조합만 맞으면 복원할 수 있다."""

    cho = cho_logits.argmax(dim=-1).tolist()
    jung = jung_logits.argmax(dim=-1).tolist()
    jong = jong_logits.argmax(dim=-1).tolist()
    return [compose_hangul_syllable(c, j, g) for c, j, g in zip(cho, jung, jong)]
