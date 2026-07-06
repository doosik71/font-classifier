# 한글 폰트 인식 모델 설계

## 0. 전제

- 폰트 원본 파일(ttf/otf)은 보유하지 않는다. `data/scan`의 스캔 영상이
  유일한 데이터 소스이며, 합성 렌더링으로 데이터를 보충할 수 없다.
- 최종 추론 환경은 서버/클라우드다. 모바일 온디바이스 제약이 없으므로
  모델 크기보다 정확도를 우선한다.

## 1. 데이터와 문제 정의

### 1.1 데이터 구조

- 폰트 약 3,000개 x 완성형 한글 2,350자(KS X 1001) = 약 705만 셀의 (거의)
  완전한 카르테시안 곱.
- 각 `(폰트, 글자)` 조합의 실제 촬영 샘플은 정확히 1장이다.
- 일부 폰트는 특정 글자가 비어 있어(결측) 인쇄되지 않는다
  (`font_dataset.py`의 `missing_count` 처리 참고).
- 현재 정규화 기준은 64x64 그레이스케일 영상이다(`char_extract.py`).

### 1.2 데이터 희소성의 실제 위치

폰트 분류만 보면 폰트당 최대 2,350장, 자소(초/중/종성) 분류만 보면 자소당
최소 수만 장의 표본이 있어 두 태스크 자체는 데이터가 궁핍하지 않다. 진짜
희소성은 **(폰트, 글자) 조합 단위**에만 있고, 이것이 실제로 문제가 되는
지점은 **스타일과 구조를 분리(disentangle)하는 학습**이다 — 모든 셀에서
폰트와 글자가 동시에 달라지므로, 같은 조합을 두 번 보지 못한 채로 두 축을
갈라내야 한다. 이 설계의 self-supervised/multi-task 구성 요소는 표본 수를
늘리는 것이 아니라 이 disentanglement 신호를 만드는 것을 목표로 한다.

### 1.3 도메인 갭

최종 입력은 스캔 영상이 아니라 사용자 사진에서 잘라낸 글자다. 조명,
원근, 카메라 노이즈, 배경, 부분 가림 등 스캔 데이터에 없는 변형이
발생하며, 이는 실제 사진 crop 데이터가 확보된 뒤에야 최종 검증할 수 있다
(README "진행 상황" 참고). 이 설계는 스캔 데이터만으로 도메인 변화에
버틸 수 있는 augmentation 기준선을 만드는 것을 1차 목표로 한다.

## 2. 태스크 정의

모델은 낱글자 영상 `x(font=f, char=c)` 하나에서 다음을 동시에 예측/복원한다.

| 태스크         | 출력                                        | 목적                                               |
| -------------- | ------------------------------------------- | -------------------------------------------------- |
| 한글 구조 인식 | 초성 19 / 중성 21 / 종성 28(받침 없음 포함) | 글자 구조 특징 학습                                |
| 폰트 인식      | 폰트 ~3,000 클래스 + embedding              | 스타일 특징 학습                                   |
| 영상 복원      | 입력과 같은 64x64 글자 영상                 | self-supervised 보조 학습, feature disentanglement |

### 2.1 왜 초성/중성/종성 분리 인식인가

2,350자를 직접 분류하지 않고 초/중/종성을 분리 인식한 뒤 조합하는 방식을
쓴다.

1. **표현 공간 축소**: 2,350개 클래스 대신 19+21+28=68개 클래스를
   학습하면 각 자소가 여러 글자에 걸쳐 반복 등장해 학습 신호가 조밀해진다.
2. **오픈 보캐뷸러리 일반화**: 학습 데이터는 KS X 1001 2,350자로 한정되지만
   초/중/종성 조합 공간(19x21x28=11,172)은 유니코드 현대 한글 음절 전체와
   정확히 일치한다. 각 자소의 모양만 제대로 배우면 학습 중 보지 못한 글자
   (2,350자 밖의 나머지 8,822자)도 조합으로 인식할 수 있다.

## 3. 아키텍처

### 3.1 개요

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

### 3.2 인코더

| stage  | 해상도 | 채널 | 블록                     |
| ------ | ------ | ---- | ------------------------ |
| stem   | 64x64  | 32   | conv3x3                  |
| stage1 | 64x64  | 32   | residual x2              |
| stage2 | 32x32  | 64   | residual x2 (downsample) |
| stage3 | 16x16  | 128  | residual x2 (downsample) |
| stage4 | 8x8    | 256  | residual x3 (downsample) |
| stage5 | 4x4    | 512  | residual x3 (downsample) |

표준 ResNet residual block(conv-norm-act-conv-norm + shortcut)을 쓴다.
Batch가 작아질 가능성이 있으면 GroupNorm을 우선 검토한다. 파라미터 규모는
ResNet-18~34 수준(10~25M)으로, 서버 추론 전제에서 문제되지 않는다. 성능이
부족하면 stage4/5의 블록 수나 채널을 늘린다.

### 3.3 Content code / Style code

- `pooled = GlobalAvgPool(stage5_output)` → 512-dim
- `content = MLP(pooled) → 128-dim` — 초/중/종성 조합 공간(68개)에 맞춰
  압축.
- `style = MLP(pooled) → 512-dim, L2 정규화` — 폰트 ~3,000개와 다중 glyph
  집계(3.6, 7절)를 고려해 얼굴 인식 embedding과 비슷한 차원을 준다.
  이 정규화된 style embedding은 추론 시 폰트 근거 집계에 직접 사용하고,
  실제 폰트 분류는 3.6절의 별도 FontHead MLP가 담당한다.

두 code는 같은 `pooled` 벡터에서 각각 별도 프로젝션으로 나오므로, 실제
분리는 (a) 각 code에 연결된 분류 손실의 직접 지도학습, (b) 4.2~4.4절의
재구성/대조학습으로 강제한다.

### 3.4 Skip connection 정책

**Encoder-decoder 사이의 U-Net식 cross skip connection은 사용하지 않는다.**
같은 stage 내부의 ResNet식 residual bypass는 허용하지만, encoder feature
map을 decoder로 직접 넘기는 cross skip은 어떤 해상도에서도 두지 않는다.
64x64 한글 glyph에서는 고해상도 feature가 이미 획 위치·두께·자소 배치
같은 정답에 가까운 정보를 담기 때문에, cross skip을 허용하면 decoder가
content/style code를 거치지 않고 픽셀 정보를 우회 복사해 latent가
disentangle되지 않을 위험이 있다. Content/jamo/font 각각에 직접적인 정답
라벨이 있어 disentanglement의 주된 힘은 분류 손실이 담당하므로, cross
skip을 없애도 분류 정확도는 위협받지 않는다. 대신 재구성 품질(특히 얇은
획·세리프 같은 고주파 디테일)이 다소 흐려질 수 있다는 트레이드오프를
받아들이며, 이는 디코더의 재구성이 목적이 아니라 보조 신호이기 때문에
감당 가능한 비용으로 판단한다. 재구성이 참고 지표로도 쓸모없을 만큼
나빠지면 skip을 재도입하지 않고 디코더 용량(채널/블록 수)을 늘려
대응한다.

### 3.5 디코더

```text
concat(content 128, style 512) = 640-dim
  -> MLP -> 512 x 4 x 4
  -> upsample residual blocks: 4 -> 8 -> 16 -> 32 -> 64
     (encoder stage와의 cross skip 없음)
  -> 1 x 64 x 64 (sigmoid 또는 clipped grayscale)
```

Cross skip이 없으므로 디코더 구현에 조건 분기가 필요 없다 — 4.2절의 두
재구성 모드가 완전히 같은 디코더를 공유한다.

### 3.6 헤더

- **Hangul head**: `content(128)` → 독립 MLP(128→64→N) 3개로 cho(19)/
  jung(21)/jong(28) logits.
- **Font head**: `style(512, L2 정규화)`를 바로 선형 분류하지 않고,
  별도 MLP 분기를 한 번 더 거쳐 폰트 logits을 예측한다.

```text
style embedding (512, L2 정규화)
  -> Linear(512 -> 1024)
  -> LayerNorm(1024)
  -> GELU
  -> Dropout(0.1)
  -> Linear(1024 -> 512)
  -> LayerNorm(512)
  -> GELU
  -> Linear(512 -> num_font_classes)
```

이 설계의 목적은 두 가지다.

1. 정규화된 style embedding을 바로 선형 분류기에 넣을 때 생길 수 있는
   logit 동적 범위 제약을 완화한다.
2. 3,000개가 넘는 세밀한 폰트 클래스를 더 표현력 있는 헤드로 분리한다.

LayerNorm을 쓰는 이유는 배치 크기 변화에 덜 민감하기 때문이다. 추론 시
여러 glyph의 근거를 집계하는 용도로는 **정규화된 style embedding 자체를
그대로 보존**하고, FontHead의 MLP는 분류 전용 분기로만 사용한다(7절).

### 3.7 초중종 조합 및 디코딩

유니코드 한글 음절은 `code = 0xAC00 + (cho*21 + jung)*28 + jong`로
결정된다(cho 0~18, jung 0~20, jong 0~27, jong=0은 받침 없음). 학습 라벨
생성과 추론 디코딩에 같은 산술 공식을 쓴다.

```text
index = ord(char) - 0xAC00
cho_label  = index // 588
jung_label = (index % 588) // 28
jong_label = index % 28
```

추론 시 두 가지 디코딩 모드를 둔다.

1. **제한 디코딩 (기본, 평가용)**: KS X 1001 2,350자 표 위에서
   `log P(cho) + log P(jung) + log P(jong)`이 최대인 글자를 고른다. 학습
   라벨과 직접 비교 가능해 평가 지표로 쓴다.
2. **개방 디코딩 (앱 추론 옵션)**: cho/jung/jong 각각의 argmax를 산술
   공식에 그대로 대입해 11,172 음절 전체 중 하나를 얻는다. 2,350자 표
   밖의 글자도 조합만 맞으면 복원된다. Confidence가 애매하면 제한 디코딩
   결과를 fallback으로 함께 제시하는 하이브리드 방식을 쓴다.

## 4. 학습 전략

### 4.1 손실 함수

```text
L = lambda_jamo   * (CE_cho + CE_jung + CE_jong)
  + lambda_font   * CE_font
  + lambda_reconA * L_recon_modeA      (self-reconstruction, denoising)
  + lambda_reconB * L_recon_modeB      (factorized cross-recombination)
  + lambda_cont   * L_contrastive
```

| 항목            | 초기값 | 설명                                         |
| --------------- | -----: | -------------------------------------------- |
| `lambda_jamo`   |    1.0 | 자소 인식                                    |
| `lambda_font`   |    1.0 | FontHead logits에 대한 softmax cross entropy |
| `lambda_reconA` |    0.2 | self-reconstruction (L1, 필요시 SSIM 추가)   |
| `lambda_reconB` |    0.3 | factorized cross-recombination               |
| `lambda_cont`   |    0.1 | 대조학습/augmentation consistency            |

수동 튜닝이 부담스러워지면 uncertainty weighting(Kendall et al.) 같은
자동 균형 기법으로 전환한다.

### 4.2 재구성 2-모드

재구성 태스크를 두 모드로 나눠 학습하며, 둘 다 cross skip 없이 병목
벡터(content+style)만으로 복원한다. 차이는 code의 출처뿐이다.

- **모드 A. Self-reconstruction**: 같은 이미지(또는 그 augmentation
  view)에서 content/style code를 모두 추출해 그 이미지를 복원한다. 입력에
  약한 noise/masking을 걸어 denoising 변형으로 쓴다. 목적은
  disentanglement가 아니라 인코더/디코더 pretrain 안정화와 재구성 품질
  참고 지표 확보다.
- **모드 B. Factorized cross-recombination reconstruction**: 구조 소스
  `x(f_struct, c)`(목표와 같은 글자, 다른 폰트)에서 content code를, 스타일
  소스 `x(f, c_style)`(목표와 같은 폰트, 다른 글자)에서 style code를 각각
  추출해 concat한 뒤, 실제로 존재하는 목표 이미지 `x(f, c)`를 복원한다.
  카르테시안 곱 구조 덕분에 대부분의 `(f, c)`에서 목표 이미지가 실존해
  supervised reconstruction loss를 바로 계산할 수 있다. 이 모드가
  disentanglement를 검증/강제하는 핵심 시험대다.

### 4.3 배치 샘플링 전략

배치를 "K개 폰트 x M개 글자" 그리드로 구성하는 전용 sampler를 쓴다(예:
K=8, M=8 → 배치 크기 64). 이렇게 하면 별도 pair-mining 로직 없이 배치
안에서:

- 같은 폰트(행)를 공유하는 다른 글자 쌍 → style-positive 쌍, 모드 B의
  스타일 소스
- 같은 글자(열)를 공유하는 다른 폰트 쌍 → content-positive 쌍, 모드 B의
  구조 소스
- 그리드의 대각 밖 셀 `(f, c)` → 모드 B의 목표 이미지

가 자연스럽게 확보된다. 구조/스타일 소스는 목표와 각각 폰트·글자가 달라야
하고(비자명한 조합만 사용), 세 이미지 모두 결측 셀이 아니어야 한다.

### 4.4 대조학습/self-supervised 보강

- **Supervised contrastive on style**: 같은 폰트의 여러 글자 style code가
  서로 가까워지도록.
- **Supervised contrastive on content**: 같은 (cho,jung,jong) 조합을 가진
  여러 폰트의 content code가 서로 가까워지도록.
- **Instance-level augmentation consistency**: 같은 셀에 서로 다른
  augmentation을 두 번 적용한 view가 같은 content/style code를 내도록
  (SimCLR/BYOL류). 라벨을 쓰지 않는 유일한 순수 self-supervised 항이며,
  스캔-사진 도메인 격차에 대한 강건성을 기르는 핵심 메커니즘이다 — 사진
  도메인 augmentation을 이 항에 강하게 반영한다.

대안으로 gradient reversal 기반 adversarial disentanglement도 가능하지만,
대조학습이 디버깅하기 쉬워 1차 구현에서는 대조학습을 우선한다.

### 4.5 학습 커리큘럼

1. **Phase 0. 데이터셋 export**: `data/annotation` + `data/scan`에서
   64x64 정규화 이미지와 manifest를 생성한다. manifest는 최소한
   `font_id`, `font_name`, `char`, `char_index`, `cho/jung/jong_label`,
   `source_zip`, `source_image_name`, `row`, `col`, `is_empty_cell`,
   `normalized_image_path`를 포함한다.
2. **Phase 1. 분류 헤드 baseline**: encoder + content/style head + jamo
   head + font head만 학습(softmax CE), 재구성/대조학습은 끈다. 자소·폰트
   분류가 각각 안정적으로 수렴하는지 먼저 확인한다.
3. **Phase 2. 모드 A(self-recon) 추가**: denoising 변형을 포함해 재구성을
   켜고, 분류 지표가 나빠지지 않는지 확인한다.
4. **Phase 3. 모드 B(factorized recombination) + 대조학습 추가**:
   disentanglement probe 지표(6.2절)가 실제로 개선되는지 확인한다. 개선이
   없으면 `lambda_reconB`/`lambda_cont` 가중치나 latent 차원을 조정한다
   (cross skip 재도입은 옵션에서 제외).
5. **Phase 4. Augmentation 강화 + 사진 도메인 적응**: 5.4~5.5절.


### 4.6 최적화 실무

AdamW + cosine decay, mixed precision(fp16/bf16), 그래디언트 클리핑을
기본으로 한다.

## 5. 데이터 파이프라인

### 5.1 분할 전략

최종 앱은 "학습된 3,000개 폰트 중 어느 것인가"를 맞히는 closed-set
문제이므로 폰트 자체를 test에서 제외하지 않는다. `(font, char)` 셀 단위로
hold-out한다.

| split                | 목적                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| cell-holdout (기본)  | 모든 폰트·글자가 train에 등장하되, 폰트마다 일부 글자(10~15%)를 test로 남겨 단일 glyph memorization 방지 |
| held-out-font (보조) | 소수 폰트를 통째로 제외해 style embedding이 새 폰트에도 의미 있게 군집하는지 retrieval 성능으로 확인     |

### 5.2 결측 셀 처리

빈 칸(해당 폰트에 그 글자가 없어 인쇄되지 않은 셀)은 학습/평가에서
제외하되, manifest에는 결측 표시로 남긴다.

### 5.3 Augmentation

목적별로 분리한다(종횡비·획 두께 비율은 폰트 형태 분류의 핵심 특징이라는
원칙을 항상 우선한다).

- **공통(약함)**: ±2px 이동, ±2도 회전, 0.92~1.08 스케일, 약한
  perspective/shear, 밝기/대비, 약한 Gaussian noise/blur, JPEG 압축
  흉내, 획을 완전히 지우지 않는 범위의 random erasing.
- **구조(content) 전용 강한 변형**: 약한 dilation/erosion, threshold
  jitter, blur/sharpen 변화 — 이 view는 font 손실에는 쓰지 않는다.
- **스타일(style) 보호 규칙**: font 손실 학습 view에서는 획 두께,
  serif/디테일을 크게 바꾸는 dilation/erosion·과도한 blur를 낮은
  확률/강도로만 적용한다.
- **사진 도메인 모사** (4.4절 consistency 항에 집중 투입): 원근 왜곡,
  부분 가림(cutout), 불균일 조명/그림자, 컬러 배경, 모션 블러.

### 5.4 사진 도메인 적응 (실사진 확보 이후)

실제 사진 crop이 확보되면 다음을 추가한다.

- 같은 사진 crop의 augmentation view 간 embedding consistency(4.4절 메커니즘을 실사진에도 적용).
- 스캔 데이터에서 학습한 pseudo-label을 낮은 가중치로 쓰는 fine-tuning.

사진 데이터가 없는 지금은 시작할 수 없으므로, 현재 설계의 목표는 스캔
데이터만으로 도메인 변화에 최대한 버티는 기준선을 만드는 것으로 한정한다.

## 6. 평가

### 6.1 지표

- **한글**: cho/jung/jong 개별 accuracy, 셋이 모두 맞은 syllable accuracy,
  2,350자 제한 디코딩 기준 top-1/top-5, 개방 디코딩 결과가 2,350자 표
  안에 있는 비율(참고).
- **폰트**: 단일 glyph top-1/top-5, 여러 glyph embedding 집계 후
  top-1/top-5, 폰트별/글자별 macro accuracy, 혼동 행렬.
- **복원**: L1/SSIM (참고 지표, 목표가 아님).

### 6.2 Disentanglement probe

- `style` code만으로 cho/jung/jong을 예측하는 얕은 probe accuracy → 낮을수록 좋다.
- `content` code만으로 font를 예측하는 얕은 probe accuracy → 낮을수록 좋다.

Phase 2(모드 A만)와 Phase 3(모드 B 추가) 사이에서 이 지표가 실제로
개선되는지가 재구성 2-모드 설계가 유효했는지의 판단 근거다. 좋은 모델은
reconstruction metric만 좋은 모델이 아니라, 이 probe 지표가 함께 낮은
모델이다.

## 7. 추론 방식

### 7.1 단일 glyph

1. 글자 crop을 학습과 같은 방식으로 64x64 그레이스케일로 정규화한다.
2. 모델이 초/중/종성 확률과 폰트 logit/embedding을 출력한다.
3. 자소 확률을 3.7절 방식으로 조합해 글자를 결정한다.
4. 폰트는 top-k 후보와 confidence를 유지한다.

### 7.2 여러 glyph 폰트 집계

같은 사진/텍스트 영역의 글자들이 같은 폰트라고 가정할 수 있으면, 폰트
evidence를 문서 단위로 집계한다.

```text
document_font_logit = mean_or_sum(font_logits for glyph in document)
document_font_embedding = normalize(mean(e_font for glyph in document))
```

글자별 confidence가 낮거나 crop 품질이 나쁜 샘플은 낮은 가중치를 준다.
최종 결과는 top-1 하나가 아니라 top-5 후보와 confidence를 함께 제공한다.

## 8. 구현 로드맵

1. Phase 0 데이터셋 export 스크립트 (manifest + 정규화 이미지 파일)
2. 초중종 라벨 생성 함수 + 제한/개방 디코딩 함수, 단위 테스트
3. Encoder + 분류 헤드 baseline 학습 (재구성 없음)
4. `K x M` 그리드 배치 sampler 구현
5. 모드 A(self-recon, denoising) 디코더 추가
6. 모드 B(factorized recombination) 학습 루프 + disentanglement probe 추가
7. Augmentation 강화 + held-out-font split 평가
8. Font embedding 기반 multi-glyph aggregation 평가
9. 실사진 crop 확보 후 도메인 적응 fine-tuning

## 9. 1차 성공 기준

- cell-holdout split에서 자소 기반 음절 top-1 accuracy가 안정적으로 수렴한다.
- 단일 glyph 폰트 top-5 accuracy가 의미 있는 기준선을 만든다.
- 같은 폰트의 여러 glyph를 집계했을 때 폰트 top-1/top-5가 단일 glyph보다 뚜렷하게 오른다.
- 재구성(모드 A/B)을 추가했을 때 자소/폰트 metric이 떨어지지 않거나,
  떨어지더라도 원인을 loss 가중치와 latent 용량으로 설명할 수 있다.
- 결측 글자와 annotation 오류가 학습 실패로 섞이지 않도록 manifest와 로그에서 분리된다.

이 기준을 통과한 뒤에야 더 큰 backbone이나 다른 복잡한 기법을 추가한다.

## 10. 가정과 열린 질문

- 학습 프레임워크는 명시되지 않았으나 PyTorch를 전제로 서술했다.
- 개방 디코딩(3.7절)에서 2,350자 표 밖 결과를 신뢰할 confidence 임계값은
  실사진 검증 데이터 없이는 확정할 수 없다.
- 3,000개 밖의 새 폰트가 추후 추가될 가능성을 염두에 두고 style을
  embedding 기반으로 설계했지만, 재학습 없이 embedding만으로 충분한
  확장이 가능한지는 별도 실험이 필요하다.
- Cross skip 금지로 인한 재구성 품질 저하가 참고 지표로도 쓸모없는
  수준인지는 Phase 2~3에서 확인이 필요하다.
- 폰트 이름은 다르지만 실제 디자인이 동일/유사한 경우(클래스 앨리어싱)의
  탐지·처리 방안은 이 문서의 범위 밖이며, 학습 전략 수립 단계에서 별도로
  다룬다.
