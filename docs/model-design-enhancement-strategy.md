# FontHead 개선 전략 (model-design 보강안)

## 0. 이 문서의 위치

이 문서는 폰트 인식률이 낮은 문제를 **FontHead(폰트 분류 경로) 구조 개선**
관점에서 분석하고 개선안을 세운다. `docs/model-design.md`의 3.3절(content/
style code)과 3.6절(헤더)을 보강하는 실험 계획서이며, 여기 제안한 변경의
성능 개선 효과가 실제로 확인되면 그 내용을 `model-design.md`에 반영하고 이
문서는 흡수/삭제할 예정이다.

학습 스크립트/지표 관련 배경은 [train-model-v1.md](train-model-v1.md),
[train-model-v2.md](train-model-v2.md)를, 시각적 유사 폰트 문제는
[research-paper.md](research-paper.md)를 참고한다.

## 1. 증상과 문제의 성격

- 학습이 진행되어 **자소(jamo) 손실은 0.01 수준까지 떨어지는데, 폰트
  (font) 손실은 여전히 2~5 사이를 오간다.** 폰트 정확도(top-1)도 매우
  낮다.
- 이 지표들은 전부 **학습 배치 기준**이다([train-model-v1.md]
  (train-model-v1.md) 2.2절). 즉 held-out 일반화 문제가 아니라 **학습
  데이터조차 맞히지 못하는 underfitting(혹은 최적화/구조적 포화)** 이다.
  일반화 이전에 "훈련셋에 대한 적합" 단계에서 막혀 있다는 뜻이다.
- 참고로 3,480 클래스의 무작위 추정 손실은 $\ln(3480)\approx 8.15$이다.
  현재 2~5는 이보다 낮으므로 모델이 무언가 배우고는 있으나, 0 근처로
  내려가지 못하고 **바닥(floor)에 걸려 있다.**

핵심 질문: **왜 같은 인코더를 쓰는데 자소는 되고 폰트는 안 되는가?**
아래 2절에서 세 가지 원인을 근거와 함께 분리한다. 사용자의 가설("FontHead
레이어가 너무 작다")은 그중 하나이며, 그보다 더 값싸게 큰 효과를 낼 수
있는 원인이 하나 더 있다고 본다.

## 2. 원인 분석

### 2.1 (핵심 가설) 정규화된 임베딩 + scale 없는 분류기 → logit 동적 범위 포화

같은 `pooled`(512) 벡터에서 두 code가 나오지만, 분류기에 들어가는 형태가
**비대칭**이다(`model.py` `ProjectionHeads`/`HangulHead`/`FontHead`).

| 경로 | code         | 정규화        | 분류기 입력                  | logit                                     |
| ---- | ------------ | ------------- | ---------------------------- | ----------------------------------------- |
| 자소 | content(128) | **없음**      | 비정규화 벡터                | $\mathbf{w}_c\cdot\mathbf{h}$ (크기 자유) |
| 폰트 | style(512)   | **L2 정규화** | 단위 벡터 $\|\mathbf{z}\|=1$ | $\|\mathbf{w}_c\|\cos\theta_c + b_c$      |

- 자소 경로: content가 비정규화라 HangulHead의 logit이 자유롭게 커질 수
  있다 → softmax가 얼마든지 날카로워짐 → CE가 0.01까지 내려간다.
- 폰트 경로: style이 단위 벡터라 FontHead(단일 Linear)의 logit은 본질적
  으로 $\|\mathbf{w}_c\|\cos\theta_c$ 꼴이라 **범위가 구조적으로 제한**된다.
  게다가 AdamW weight decay(0.01)가 $\|\mathbf{w}_c\|$를 억제하므로 유효
  "온도(temperature)"가 작게 유지된다. 그 결과 softmax가 충분히 날카로워
  지지 못하고 **CE가 바닥에 걸린다.**

이 비대칭이 "자소는 0.01, 폰트는 2~5"라는 증상을 **정확히** 설명한다.
정규화 자체는 얼굴 인식류 임베딩 관례를 따른 선택이지만(model-design.md
3.3절), 정규화된 임베딩에 **scale 없는 평범한 softmax**를 붙이는 조합은
잘 알려진 실패 패턴이다(그래서 ArcFace/CosFace/NormFace가 scale $s$를
반드시 둔다). 이 원인은 레이어를 키우지 않고도 고칠 수 있어 우선순위가
높다.

> v2(sigmoid) 손실에서도 같은 포화가 생긴다. 입력 범위가 좁아 $z$가 크게
> 못 자라면 $\sigma(z)$도 0/1로 못 붙어 손실이 바닥에 걸린다. 즉 이 문제는
> **손실(v1/v2)과 무관한 구조 문제**다.

### 2.2 (사용자 가설) FontHead 용량 부족

헤드 용량 자체도 문제다. 더 어려운 문제에 더 얕은 헤드가 붙어 있다.

| 헤드       | 구조                    | 클래스 수        | 은닉층   |
| ---------- | ----------------------- | ---------------- | -------- |
| HangulHead | `_mlp(128 → 64 → N)` ×3 | 19/21/28 (합 68) | 있음     |
| FontHead   | `Linear(512 → C)`       | ~3,480           | **없음** |

- FontHead는 **은닉층이 없는 순수 선형 분류기**다. 3,480개의 세밀
  (fine-grained) 폰트 클래스를 512차원에서 **선형 경계만으로** 분리해야
  하므로, 스타일 표현을 뽑는 부담이 전부 인코더/`style_proj`로 쏠린다.
- 자소 헤드가 은닉층을 가진 것과 대조적이다. 훨씬 쉬운 68-클래스 문제에
  더 표현력 있는 헤드가 붙어 있는 셈이다.

### 2.3 (환원 불가) 진짜 폰트 aliasing

이름은 다르지만 렌더링이 사실상 동일/유사한 폰트가 섞여 있으면
([research-paper.md](research-paper.md) 1~2절), 어떤 헤드로도 그 쌍의 train
손실은 내려가지 않는다. 이 부분은 **용량/scale로 해결 불가**하며, v2의
Top-$k$ Relaxed Negative Learning이 겨냥하는 영역이다. 2.1/2.2를 고친
뒤에도 남는 잔여 손실이 여기에 해당할 수 있다.

## 3. 구조를 바꾸기 전에: 병목 진단

원인 2.1/2.2/2.3의 비중을 먼저 확인하면 헛수고를 줄일 수 있다. 값싼 진단
순서:

1. **소규모 과적합 테스트.** 폰트 50~100종만 골라 현재 구조로 학습한다.
   - train font 손실이 0 근처로 내려가면 → 구조 자체는 소수 클래스는
     맞힐 수 있고, 3,480으로 갈 때 **scale/용량이 클래스 수에 눌린다**
     (2.1/2.2). 이 경우 4절 개선의 효과가 크다.
   - 소수 폰트조차 0으로 못 내려가면 → **scale 포화(2.1)** 가 지배적일
     가능성이 높다. 4.1/4.2를 최우선으로.
2. **혼동 행렬 / 최근접 폰트 분석.** 오분류가 특정 near-duplicate 폰트
   쌍에 몰리면 aliasing(2.3), 넓게 흩어지면 용량/scale(2.1/2.2).
3. **정규화 제거·scale 추가 A/B.** 가장 싼 실험(4.2). 이것만으로 손실
   바닥이 확 내려가면 2.1이 주범이었음이 확정된다.

## 4. FontHead 구조 개선안 (핵심)

### 4.1 (1순위) MLP 헤드로 심화 — 용량 부족과 정규화 포화를 동시에 해소

FontHead를 은닉층 있는 MLP로 바꾼다. 이 한 변경이 **2.2(용량)와 2.1(포화)를
동시에** 완화한다. 핵심 통찰: MLP의 은닉/출력은 비정규화이므로, **마지막
Linear에 들어가는 특징의 크기가 더 이상 1로 묶이지 않는다** → logit 동적
범위 제약이 자연히 풀린다.

```python
class FontHead(nn.Module):
    def __init__(self, num_font_classes, style_dim=512,
                 hidden_dim=1024, dropout=0.1):
        super().__init__()
        # style(정규화된 512)을 받아 비선형 변환 후 분류.
        # LayerNorm은 배치 크기에 무관 - FontGroupBatchSampler가 폰트 묶음
        # 경계에서 배치를 작게 만들 수 있어(model-design.md 3.2절이 GroupNorm을
        # 택한 것과 같은 이유) BatchNorm1d 대신 LayerNorm을 쓴다.
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

    def forward(self, style):
        return self.classifier(self.mlp(style))
```

설계 포인트:

- **은닉 차원**: 512 → 1024(≈2배)에서 시작. 필요하면 2,048까지 키우거나
  블록을 2단으로 늘린다. 3,480 클래스이므로 여유 있게 잡아도 서버 추론
  전제(model-design.md 0절)에서 문제 없다.
- **정규화**: LayerNorm 권장(위 주석의 이유). BatchNorm1d는 작은/편향된
  배치에서 불안정하다.
- **비선형/드롭아웃**: GELU(또는 ReLU) + Dropout(0.1~0.3). 클래스가 많아
  헤드가 과적합하기 쉬우므로 드롭아웃으로 정규화한다.
- **임베딩 보존**: 추론 시 여러 glyph 근거를 집계하는 데 쓰는 `style`
  임베딩(model-design.md 3.6/7절)은 지금처럼 정규화된 벡터를 그대로
  노출한다. 헤드의 MLP는 그와 **별개 분기**이므로 임베딩 의미를 훼손하지
  않는다.

### 4.2 (병행/대안) 학습형 scale — scaled cosine 분류기

선형 헤드를 유지하면서 2.1만 직접 겨냥하려면, 정규화된 특징에 **scale**을
도입한다(NormFace 계열). weight도 정규화한 뒤 온도 $s$로 키운다.

```python
# logit_c = s * cos(theta_c) = s * (w_c/||w_c||) . (z/||z||)
self.scale = nn.Parameter(torch.tensor(16.0))   # 학습형(또는 고정 16~30)
w = F.normalize(self.classifier.weight, dim=1)
logits = self.scale * F.linear(F.normalize(style, dim=1), w)
```

- $s$를 학습형으로 두거나 16~30 범위로 고정한다. 이것만으로 softmax가
  날카로워질 여지가 생겨 손실 바닥이 내려갈 수 있다.
- **4.1과 병행 가능**: MLP로 심화한 뒤 최종 분류를 scaled cosine으로 두면
  용량↑ + 포화 해소 + 코사인 기하(임베딩 친화)를 모두 얻는다.
- 4.1을 채택하면 포화는 이미 풀리므로 4.2는 선택 사항이다. 둘 다
  ablation으로 비교한다(7절).

## 5. 헤드 밖 보조 레버 (필요 시)

헤드를 고친 뒤에도 부족하면 검토한다. 우선순위는 낮다(자소가 잘 되는 것으로
보아 공유 인코더/pooled 표현 자체는 풍부하다).

- **5.1 `style_proj` 심화**: 현재 `_mlp(512 → 512 → 512)`. 은닉/깊이를
  키워 스타일 특징 추출력을 보강.
- **5.2 분류기 weight decay 분리**: 분류기 파라미터의 weight decay를
  작게/0으로(파라미터 그룹 분리) 두어 logit scale 억제를 완화(2.1 보조).
- **5.3 pooling 개선**: GAP는 공간 정보를 뭉갠다. 폰트는 획 두께/세리프
  같은 고주파 디테일이 중요하므로 **GeM pooling**이나 attention pooling으로
  바꾸면 style 표현이 좋아질 수 있다.
- **5.4 인코더 style 경로 용량**: 공유 인코더가 병목이면 stage4/5 용량을
  키운다(model-design.md 3.2절). 다만 자소가 잘 되므로 후순위.
- **5.5 헤드 전용 학습률**: 헤드가 크게 새로 학습해야 하므로 헤드
  파라미터에 더 큰 LR(또는 더 긴 warmup)을 주는 것도 검토.

## 6. 손실(v1/v2)과의 상호작용

- **구조 개선은 loss-agnostic하게 적용한다.** 4.1/4.2/5의 변경은 v1
  (softmax CE)과 v2(Top-$k$ Relaxed Negative)에 **똑같이** 넣어야 두 방법의
  공정 비교가 유지된다([train-model-v2.md](train-model-v2.md) 2.1절의
  통제 원칙).
- 2.1(포화) 해소는 v2의 sigmoid 포화도 함께 완화하므로 v2에도 이득이다.

## 7. 실험 / 롤아웃 계획 (싸고 효과 큰 것부터)

각 실험은 **독립 ablation**으로, 데이터셋/샘플러/옵티마이저/에폭을 동일하게
두고 폰트 경로만 바꾼다. 결과는 `data/checkpoints/<이름>`으로 분리 저장해
[train-monitor.md](train-monitor.md)로 나란히 비교한다.

| 순서 | 변경                                    | 겨냥 원인 | 기대                                |
| ---- | --------------------------------------- | --------- | ----------------------------------- |
| 0    | 소규모 과적합 진단(3.1) + 혼동행렬(3.2) | 병목 규명 | 2.1/2.2 vs 2.3 비중 파악            |
| 1    | scaled cosine + $s$ (4.2)만             | 2.1       | 최소 변경으로 손실 바닥 하강 여부   |
| 2    | MLP 헤드 (4.1)                          | 2.1+2.2   | 주력안. train font 손실 급하강 기대 |
| 3    | MLP + scaled cosine (4.1+4.2)           | 2.1+2.2   | 둘의 결합 효과                      |
| 4    | (v1 전용) margin (4.3)                  | 분리도↑   | v2와 배타 비교                      |
| 5    | 보조 레버 (5: pooling/style_proj 등)    | 잔여      | 추가 개선 여지                      |

**판정 지표**:

- **train font 손실 바닥**(1차 신호): 현재 2~5 floor가 뚜렷이 내려가면
  underfitting(2.1/2.2)이 해소된 것이다.
- **`font_acc` / `font_top5_acc`**: v1/v2 공통 비교 지표
  ([train-model-v2.md](train-model-v2.md) 2.5절). top-1보다 top-5가 유사
  폰트 상황에서 더 신뢰할 신호다.
- **(후속) held-out 정확도, 혼동 행렬**: 일반화와 aliasing 잔여 확인.

## 8. 성공 기준과 model-design.md 병합 방침

- **성공 기준**: (1) train font 손실이 현재 2~5 floor에서 유의하게 하강,
  (2) `font_top5_acc`가 baseline 대비 뚜렷이 상승, (3) 자소 지표가 나빠지지
  않음(공유 인코더 악영향 없음), (4) 이후 held-out에서도 개선 확인.
- 위 기준을 통과한 구성(예: "MLP 헤드 + scaled cosine")이 확정되면
  `model-design.md` 3.3절(content/style code — 정규화/ scale 정책)과 3.6절
  (헤더 — FontHead 구조)을 그 내용으로 갱신하고, 본 문서는 흡수/삭제한다.
- model-design.md 9절의 "이 기준을 통과한 뒤에야 margin-based 같은 복잡한
  기법을 추가한다"는 순서도 그대로 지킨다.