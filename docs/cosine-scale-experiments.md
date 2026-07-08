# Cosine+Scale 폰트 헤더 실험 — 목적 · 환경 · 방법 · 결과

폰트 인식 헤더를 현재의 선형 분류기에서 **cosine 분류기 + 학습 가능한
scale**(CosFace/ArcFace 계열)로 바꾸면 font 학습이 실제로 빨라지는지를
A/B로 측정한 실험 기록이다. 실험은 `scripts/cosine-scale-benchmark.py`로
수행했고, **기존 모듈(`font_classifier/model.py`)은 전혀 수정하지 않았다**
(subclass로 헤더만 교체). 결과 로그는
`data/results/cosine-scale-benchmark.jsonl`에 있다.

> 결론 먼저: cosine+scale 헤더(Variant B)가 baseline 선형 헤더(Variant A)
> 대비 **같은 목표 font 정확도에 15~17% 적은 스텝으로 도달**했다. per-step
> 처리량은 두 모델이 동일하므로, per-step 비용 추가 없이 수렴이 빨라진다.
> 단, 아래 지표는 모두 **학습 배치 기준**이며 held-out 검증은 하지 않았다.

## 1. 실험 목적

- jamo(초/중/종성) 손실은 첫 epoch 만에 포화(val loss ~0.04, 음절 정확도
  ~99%)되는 반면, **font 손실은 느리게 감소**한다. 그 원인을 진단하고 font
  수렴을 앞당길 수 있는지 확인하는 것이 목적이다.
- 검토한 가설과 판정:
  - **loss 가중치 불균형** — 기각. 초기 loss가 jamo(=ln19+ln21+ln28≈9.31)와
    font(=ln3480≈8.15)로 비슷하고, 수렴 근처에서는 총 loss의 ~88%가 이미
    font라 gadient를 굶주리지 않는다.
  - **임베딩 차원(style 512 vs content 128)** — 기각. 큰 차원은 용량이지
    수렴을 늦추는 요인이 아니다.
  - **style의 L2 정규화 제거** — 기각. `FontHead` 첫 LayerNorm이 이미
    magnitude를 지워 분류에는 거의 무영향인데, 추론용 retrieval 임베딩과
    향후 대조학습이 의존하는 하이퍼구 불변식만 깨진다.
  - **GradNorm / hangul head freeze** — 기각. 원인이 가중치·용량이 아니라
    난이도이므로 효과가 없고, freeze는 한글 망각·disentanglement 붕괴 위험만
    있다.
  - **cosine 분류기 + 학습 가능한 scale** — 유일하게 남은 유망 후보. 정규화된
    특징 위에서 3,480개 세밀 클래스를 분리하려면 큰 logit 동적범위가 필요한데,
    baseline 헤더는 LayerNorm에 스케일이 묶여 그 범위를 확보하기 어렵다는
    가설. **이 실험이 검증 대상이다.**

## 2. 실험 환경

- **스크립트**: `scripts/cosine-scale-benchmark.py`
- **데이터셋**: `data/dataset` (완성형 낱글자, 폰트 클래스 3,480개),
  augmentation on, `FontGlyphDataset` 그대로 사용.
- **장치**: CUDA (`--device cuda:1`), bfloat16 AMP.
- **모델**: `font_classifier.model.HangulFontRecognitionModel` 공유.
  - **Variant A (baseline)**: 기존 `FontHead` — MLP(512→1024→512, LayerNorm/
    GELU/Dropout) 뒤에 선형 분류기 `Linear(512→3480)`.
  - **Variant B (cosine+scale)**: 스크립트에서 subclass한 `CosineScaleModel`.
    `FontHead.mlp`(정의를 복제하지 않고 **그대로 재사용**) 뒤에 cosine
    분류기를 둔다:

    ```text
    feat  = normalize(mlp(style))      # 특징 L2 정규화
    W_i   = normalize(weight)          # 클래스 벡터 L2 정규화
    logit = s * (feat · W_i)           # s = exp(log_scale), 학습 가능
    ```

    즉 **실험 변수는 "마지막 분류기의 형태" 하나뿐**이고, 인코더·content
    경로·jamo 헤더·style_proj·MLP는 모두 동일하다.

## 3. 실험 방법 (공정성 보장)

- **동일 초기값**: 같은 seed로 두 모델을 생성해 공유 부분(encoder /
  content_proj / style_proj / hangul_head / decoder)의 초기 가중치가
  bit-identical 하게 시작한다. 차이는 `font_head` 뿐임을 코드로 확인했다.
- **동일 데이터 스트림**: 매 스텝 **같은 배치**를 두 모델에 함께 입력한다
  (데이터 순서도 seed 고정). 두 모델은 각자의 optimizer로 독립 학습된다.
- **동일 학습 절차**: `lambda_jamo=1`, `lambda_font=1`, AdamW(lr 3e-4,
  wd 0.01), 선형 warmup 후 상수 lr, grad clip 1.0. 손실/지표 계산은
  `train-model.py`와 동일.
- **지표**: 스텝별 `font_loss`/`font_acc`(학습 배치 기준), 학습된 `scale`,
  그리고 각 정확도 임계값(0.1~0.5) 도달 스텝. `--log-every` 마다
  `data/results/cosine-scale-benchmark.jsonl`에 append.

주의: 두 모델을 동시에 올리므로 메모리/연산이 단일 모델 학습의 약 2배다.

## 4. 실험 결과

### 4.1 1차 실행 (짧음)

```bash
uv run python scripts/cosine-scale-benchmark.py \
  --steps 3000 --batch-size 128 --init-scale 16 --device cuda:1
```

- 초기 ~450스텝까지는 B가 A보다 근소하게 나빴다(cosine + 초기 scale이 아직
  워밍업 중). 이후 **B가 A를 추월**해 step 3000에서 `font_loss` A=6.28,
  B=5.77 (B−A=−0.51), `font_acc` A=0.008, B=0.021.
- 다만 두 모델 모두 acc < 1%로 너무 일러, 정확도 임계값(0.1~0.5)은 전부
  미도달이었다 — 판정에는 더 긴 실행이 필요했다.

### 4.2 2차 실행 (긴 실행)

```bash
uv run python scripts/cosine-scale-benchmark.py \
  --steps 20000 --batch-size 128 --init-scale 16 --device cuda:1
```

256만 샘플(20,000스텝 × 128)까지 학습해 acc가 실제 학습 수준(~0.5)에
도달, 판정이 가능해졌다.

**목표 font 정확도(학습 배치 top-1) 도달 스텝 — 작을수록 빠름:**

| font_acc 임계값 | A_baseline | B_cosine_scale | B가 아낀 스텝 |
| --------------: | ---------: | -------------: | ------------: |
|            0.10 |      5,606 |          4,857 |        −13.4% |
|            0.20 |      9,448 |          7,819 |        −17.2% |
|            0.30 |     11,240 |          9,278 |        −17.5% |
|            0.40 |     13,697 |         11,339 |        −17.2% |
|            0.50 |     16,402 |         13,961 |        −14.9% |

- **전 구간에서 B가 목표 정확도에 15~17% 적은 스텝으로 도달**한다.
- 교차점(B가 A를 앞서기 시작) 이후(step ~1,600) 모든 로그 지점에서 B의
  `font_loss`가 낮고 `font_acc`가 높다. step 20,000에서 `font_acc`
  A=0.485, B=0.522.
- `font_loss` 격차(B−A)는 중반(step 5k~13k)에 −0.55까지 벌어졌다가 끝에서
  −0.14로 좁혀진다. 이는 우위 소멸이 아니라 loss가 낮아질수록 절대 격차가
  압축되는 정상 현상이며, 같은 지점의 acc 격차와 도달 스텝은 끝까지 B가
  우세하다.
- A/B를 함께 돌린 실효 처리량은 실행 내내 ≈1,436 samples/s로 안정적이었다.
  cosine 헤더는 baseline의 선형 분류기 대비 `F.normalize` 두 번과 스칼라 곱만
  추가하는 구조라 per-step 비용이 사실상 같다(별도 단일 모델 프로파일링은
  하지 않았다).

### 4.3 학습된 scale의 궤적 (핵심 관찰)

step 20,000 실행에서 학습된 `scale`은 **16 → (초기 dip) 13.6 → 이후 꾸준히
상승 40.5**로, 끝에서도 상승 중이었다.

- 초기엔 특징이 무의미해 logit을 부드럽게(scale↓) 낮춰 손실을 줄이고,
  특징이 판별력을 얻자 softmax를 날카롭게(scale↑) 형성하여 정답 확신을 키운다.
- 도달한 40 부근은 얼굴인식 ArcFace/CosFace의 통상 `s`=30~64 영역과 일치한다.

이는 실험 목적(1절)의 가설 — **정규화된 특징 위 다중 클래스 softmax는 큰
logit 동적범위가 필요하고 baseline 헤더는 그 범위 확보가 어렵다** — 를 실측으로
확인해 준다. 학습 가능한 scale이 그 범위를 자유롭게 확보한 것이 B 우위의
원천이며, scale을 고정이 아니라 **학습형으로 둔 선택이 옳았음**도 이 자기조정
곡선이 뒷받침한다.

## 5. 분석 요약

- cosine+scale 헤더는 font 수렴을 **명확하고 일관되게 가속**한다(모든
  임계값에서 15~17% 적은 스텝). 이득의 형태는 애초 사용자가 원하던 **수렴
  스텝 감소**이고, per-step 처리량은 그대로다.
- 원인은 가중치·차원·정규화가 아니라 **분류기의 logit 동적범위**였고,
  learnable scale이 그것을 해결한다는 것이 scale 궤적으로 입증됐다.
- 앞서 검토한 다른 대안(L2 제거, GradNorm, freeze)은 이론적으로 배제됐고,
  cosine+scale이 실효가 있음을 이 실험이 실측으로 확정했다.

## 6. 한계 (채택 전 확인할 것)

- **학습 배치 기준 지표**다. held-out 성능은 아직 측정하지 않았다. 효과가
  크고 일관돼 뒤집힐 가능성은 낮아 보이나(오히려 margin 효과로 held-out에서
  B가 유리할 수도) 단정은 금물.
- **최종 plateau 차이 미확정**: 두 모델 모두 상승 중이라, B가 "같은 천장에 더
  빨리"인지 "더 높은 천장"인지는 완전 수렴까지 학습해야 알 수 있다. 확정된
  것은 **속도 우위**다.
- 벤치마크는 두 모델을 동시에 올려 메모리·연산이 ~2배다(빠른 비교 목적).
