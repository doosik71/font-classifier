# Font Recognition Model — 사용 설명서 및 상세설계서

`font_classifier/model.py`는 `docs/model-design.md` 3절("아키텍처")에
기술된 한글 폰트 인식 모델을 PyTorch `nn.Module`로 구현한 것이다. GUI
도구가 아니라 학습/추론 스크립트가 `import`해서 쓰는 라이브러리 모듈
이므로 `bin\*.bat` 런처는 없다.

이 파일은 아키텍처(인코더, content/style 투영, 초중종성 헤더, 폰트
헤더, 디코더)와 추론 시 초중종성 로짓을 실제 글자로 바꾸는 디코딩
유틸리티까지, `model-design.md` 3절 전체를 담는다. **학습 루프(손실
함수 가중합, 옵티마이저, 4.3절의 배치 구성과 4.2절의 두 재구성 모드
오케스트레이션 등, `model-design.md` 4절)는 이 파일의 범위가 아니다**
— 이번 작업의 요청 범위는 모델 자체였고, 학습 스크립트는 별도 작업으로
다룬다(2.6절).

이 문서는 두 부분으로 구성된다.

- **1부. 사용법** — 모델을 만들고 호출하는 방법
- **2부. 상세설계서** — 각 구성 요소의 설계 근거, `model-design.md`에
  명시되지 않아 이 구현이 임의로 정한 가정, 알려진 제한사항

## 1부. 사용법

### 1.1 모델 생성

```python
from font_classifier.model import HangulFontRecognitionModel
from font_classifier.dataset_loader import FontGlyphDataset

train_ds = FontGlyphDataset()
model = HangulFontRecognitionModel(num_font_classes=train_ds.num_font_classes)
```

`num_font_classes`는 모델 파일에 하드코딩하지 않는다 — annotation 작업이
계속 진행 중이라 실제 폰트 수(현재 약 3,480종)가 시간이 지나며 바뀌므로,
항상 그 시점의 `FontGlyphDataset.num_font_classes`를 그대로 넘긴다.

### 1.2 기본 순전파 (모드 A / 평가에 해당)

```python
output = model(images)              # images: (B, 1, 64, 64), FontGlyphDataset가 만드는 형식 그대로

output.content            # (B, 128)
output.style              # (B, 512), L2 정규화됨
output.cho_logits         # (B, 19)
output.jung_logits        # (B, 21)
output.jong_logits        # (B, 28)
output.font_logits        # (B, num_font_classes)
output.reconstruction     # (B, 1, 64, 64), [0, 1] 범위
```

`forward()`는 같은 이미지를 encode한 뒤 그대로 decode하는 편의
메서드다 — `model-design.md` 4.2절의 모드 A(self-reconstruction)나
재구성 손실 없이 분류 헤드만 쓰는 평가에 그대로 맞는다.

### 1.3 재구성 없이 분류만 (Phase 1 baseline)

`model-design.md` 4.5절 Phase 1은 재구성/대조학습을 꺼 둔 분류 헤드
baseline만 학습한다. `encode()`만 호출하면 디코더 연산 자체를 건너뛸 수
있다.

```python
encoded = model.encode(images)
encoded.reconstruction  # None - decode()를 호출하지 않았으므로
```

### 1.4 모드 B: 구조/스타일 소스를 분리해서 조합

`model-design.md` 4.2절의 모드 B(factorized cross-recombination)는 서로
다른 두 이미지에서 각각 content/style을 뽑아 섞은 뒤 복원한다.
`encode()`/`decode()`가 분리되어 있어 그대로 지원된다.

```python
struct_out = model.encode(structure_source_images)   # x(f_struct, c) - 목표와 같은 글자, 다른 폰트
style_out = model.encode(style_source_images)         # x(f, c_style) - 목표와 같은 폰트, 다른 글자
recon = model.decode(struct_out.content, style_out.style)  # x(f, c) 복원 시도
```

세 이미지(구조 소스/스타일 소스/목표) 모두 결측 셀이 아니어야 한다는
제약과, 이 쌍을 실제로 어떻게 뽑을지(배치 샘플링)는 학습 스크립트의
책임이다(2.6절 — 이 모듈은 그 조합을 계산하는 방법만 제공한다).

### 1.5 초중종성 로짓 → 실제 글자 (추론 시 디코딩)

```python
from font_classifier.model import decode_restricted, decode_open

chars = decode_restricted(output.cho_logits, output.jung_logits, output.jong_logits)
# ['가', '나', ...] - KS X 1001 2,350자 표 안에서만 고른다(평가용, 학습 라벨과 직접 비교 가능)

chars_open = decode_open(output.cho_logits, output.jung_logits, output.jong_logits)
# 2,350자 표 밖의 음절도 조합만 맞으면 나올 수 있다(실제 앱 추론 옵션)
```

두 함수 모두 배치 단위 로짓 `(B, N)`을 받아 길이 `B`의 문자열 리스트를
반환한다. 차이와 용도는 `model-design.md` 3.7절, 그리고 2.5절 참고.

## 2부. 상세설계서

### 2.1 구성 요소와 `model-design.md` 대응표

| 클래스/함수                                                   | `model-design.md` 절 | 비고                                                                               |
| ------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------- |
| `Encoder`                                                     | 3.2                  | stem + stage1~5, 표에 나온 해상도/채널/블록 수를 그대로 구현                       |
| `ResidualBlock`                                               | 3.2                  | "표준 ResNet residual block"                                                       |
| `ProjectionHeads`                                             | 3.3                  | pooled(512) → content(128) / style(512, L2 정규화)                                 |
| (skip 없음)                                                   | 3.4                  | `Decoder`가 인코더 feature를 전혀 입력받지 않는 구조 자체로 cross skip 부재를 강제 |
| `Decoder`, `UpsampleResidualBlock`                            | 3.5                  | concat(640) → 4x4 → 4단계 업샘플 → 64x64                                           |
| `HangulHead`, `FontHead`                                      | 3.6                  | 초중종성 3-way MLP, 폰트 baseline 분류기                                           |
| `compose_hangul_syllable`, `decode_restricted`, `decode_open` | 3.7                  | 조합 공식과 두 디코딩 모드                                                         |

### 2.2 정규화 레이어: GroupNorm (BatchNorm 대신)

`model-design.md` 3.2절은 "Batch가 작아질 가능성이 있으면 GroupNorm을
우선 검토한다"고 조건부로 남겨 두었다. 이 프로젝트의
`FontGroupBatchSampler`(font_classifier/batch_sampler.py)는 폰트 묶음
경계에서 배치 크기가 `K*M`보다 작아질 수 있다고 문서화되어
있으므로([batch-sampler.md](batch-sampler.md) 1.3절), 그 조건이 실제로
성립한다고 보고 처음부터 `GroupNorm(32, C)`를 기본으로 택했다(모든 채널
수 32/64/128/256/512가 32로 나누어떨어져 그룹 수를 늘 32로 고정할 수
있다). BatchNorm으로 바꾸고 싶다면 `_norm()` 헬퍼 함수 하나만 고치면
된다 - 인코더/디코더 전체가 이 함수를 통해서만 정규화 레이어를 만든다.

### 2.3 이 문서(model-design.md)에 없어 이 구현이 정한 가정

`model-design.md`가 입출력 차원만 못박고 은닉 구조를 명시하지 않은
부분들이다. 모두 나중에 바꾸기 쉬운(파라미터 수가 작거나 국소적인)
결정이다.

- **`ProjectionHeads`의 은닉 차원**: content는 `pooled(512) → (512+128)/2=320 → 128`
  로 절반씩 줄이고, style은 `512 → 512 → 512`로 폭을 유지하는 2-layer
  MLP로 구현했다. "MLP"라는 표현이 최소 은닉층 하나를 함의한다고 보고
  Linear 하나가 아니라 `Linear-ReLU-Linear`로 만들었다(`_mlp` 헬퍼).
- **`Decoder`의 첫 투영(`concat → 512x4x4`)도 같은 이유로 `Linear-ReLU-Linear`
  MLP로 구현했다(은닉 차원은 입력과 같은 640).**
- **가중치 초기화**: 문서에 명시되지 않아, ResNet 계열에서 흔히 쓰는
  두 관행을 적용했다 — (1) conv 레이어는 Kaiming normal(He init),
  (2) 각 residual/upsample block의 **마지막** norm 레이어의 scale을
  0으로 초기화("zero-init the last norm in each residual branch") —
  학습 초반 모든 블록이 항등함수에 가깝게 시작해 깊은 네트워크의 초기
  안정성을 돕는 표준 기법이다.
- **디코더 업샘플 방식**: transposed convolution 대신 nearest-neighbor
  upsample + conv를 썼다(체커보드 아티팩트를 피하는 일반적인 관행).
- **디코더 출력 활성화**: 문서의 "sigmoid 또는 clipped grayscale" 중
  sigmoid를 택했다 — `FontGlyphDataset`이 만드는 입력이 이미 `[0,1]`
  범위 float이므로(dataset-loader.md 1.3절), 재구성 손실(L1 등)을 같은
  스케일에서 바로 계산할 수 있다.

### 2.4 파라미터 수

| 구성 요소                      | 파라미터 수  |
| ------------------------------ | ------------ |
| `Encoder`                      | 약 17.1M     |
| `ProjectionHeads`              | 약 0.73M     |
| `HangulHead`                   | 약 0.03M     |
| `FontHead` (폰트 3,480종 기준) | 약 1.79M     |
| `Decoder`                      | 약 8.19M     |
| **합계**                       | **약 27.8M** |

`model-design.md` 3.2절의 "ResNet-18~34 수준(10~25M)"은 인코더 단독
기준이며(그 문장이 3.2절 인코더 설명 바로 뒤에 나온다), 실측 인코더
파라미터(17.1M)가 그 범위 안에 정확히 든다. 디코더/헤더는 그 위에 얹히는
별도 예산이라 전체 합계(27.8M)가 그 범위를 넘는 것은 설계에서 벗어난
것이 아니다. `FontHead`는 `num_font_classes`에 선형 비례하므로, annotation
작업으로 폰트가 늘어나면 이 항목도 같이 커진다.

### 2.5 `decode_restricted`가 매 호출 표를 다시 만들지 않는 이유

`HANGUL_TABLE`(2,350자)을 (초성, 중성, 종성) 인덱스로 미리 분해해
모듈 로드 시점에 `_TABLE_CHO`/`_TABLE_JUNG`/`_TABLE_JONG`으로 캐시해
둔다. `HANGUL_TABLE` 자체가 모듈 로드 이후 바뀌지 않는 상수이므로,
검증 루프에서 배치마다 이 함수를 불러도 매번 2,350자를 다시 분해하지
않는다.

### 2.6 이번 범위에 포함하지 않은 것 / 알려진 제한사항

- **학습 루프 없음**: 손실 함수 가중합(4.1절), 두 재구성 모드의 실제
  오케스트레이션(4.2절 - 어떤 이미지를 구조/스타일 소스로 쓸지 배치에서
  고르는 로직), 대조학습 항(4.4절), 학습 커리큘럼(4.5절), 옵티마이저
  설정(4.6절)은 전부 다루지 않았다. 이 모델은 그 학습 스크립트가 가져다
  쓸 `nn.Module`일 뿐이다.
- **폰트 분류기는 아직 baseline뿐**: 3.6절이 예정한 ArcFace/CosFace 등
  margin 기반 분류기나 supervised contrastive loss로의 전환은 폰트 간
  혼동이 실제로 커야 판단할 수 있는 문제라 지금 만들지 않았다(문서 자체가
  "조건부 전환"으로 서술한다). `FontHead.classifier`만 교체하면 되도록
  독립된 서브모듈로 분리해 뒀다.
- **7절(여러 glyph 폰트 집계)은 이 파일에 없다**: `style` embedding을
  문서 단위로 평균/정규화하는 로직(`document_font_embedding` 등)은
  추론 스크립트의 몫으로 남겨 뒀다 - 이 모델은 글자 하나 단위의 forward만
  제공한다.
- **성능 검증 없음**: 이 구현은 아키텍처가 문서의 치수/데이터 흐름과
  일치하는지, forward/backward가 정상 동작하는지(합성 데이터 + 실제
  `FontGlyphDataset` 배치로 확인)만 검증했다. 실제 학습 곡선, 수렴
  여부, 6절의 평가 지표는 학습을 시작해야 알 수 있다.

### 2.7 모듈 구조 요약

| 구성 요소                                                       | 역할                                            |
| --------------------------------------------------------------- | ----------------------------------------------- |
| `CONTENT_DIM`(128) / `STYLE_DIM`(512) / `POOLED_DIM`(512)       | 핵심 차원 상수 (3.3절)                          |
| `_norm`                                                         | GroupNorm 생성 헬퍼 (2.2절)                     |
| `_mlp`                                                          | `Linear-ReLU-Linear` MLP 생성 헬퍼              |
| `ResidualBlock` / `UpsampleResidualBlock`                       | 인코더/디코더 잔차 블록 (2.1, 2.3절)            |
| `Encoder`                                                       | stem + stage1~5 (3.2절)                         |
| `ProjectionHeads`                                               | content/style 투영 (3.3절)                      |
| `HangulHead` / `FontHead`                                       | 초중종성/폰트 분류 헤더 (3.6절)                 |
| `Decoder`                                                       | concat → 4x4 → 업샘플 → 64x64 (3.5절)           |
| `FontModelOutput`                                               | `encode`/`forward`가 반환하는 결과 dataclass    |
| `HangulFontRecognitionModel`                                    | 최상위 모델 - `encode`/`decode`/`forward` (1부) |
| `compose_hangul_syllable` / `decode_restricted` / `decode_open` | 초중종성 조합/디코딩 (3.7절, 1.5절)             |
