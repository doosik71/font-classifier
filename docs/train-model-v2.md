# Train Model v2 — 사용 설명서 및 상세설계서

`scripts/train-model-v2.py`는 `font_classifier/model.py`의
`FontRecognitionModel`을 학습하는 진입점 스크립트로,
**폰트 인식 헤더에 `docs/research-paper.md`가 제안한 동적 후보 선택 기반
Top-$k$ Relaxed Negative Learning을 적용한 버전**이다.

baseline인 [train-model-v1.md](train-model-v1.md)(scripts/train-model-v1.py)은
초성/중성/종성(글자 구조) 헤더는 잘 수렴하지만 폰트 헤더는 잘 학습되지
않는다. 이름은 다르지만 시각적으로 매우 유사한 폰트가 데이터셋에 섞여
있어, 정답 폰트를 제외한 모든 폰트를 똑같이 억제하는 softmax cross
entropy가 지나치게 경직된 지도 신호를 주기 때문으로 본다
(research-paper.md 1~2절). v2는 이 문제를 완화하려는 실험용 학습 방법이며,
나중에 **v1 방법과 v2 방법의 모델 성능을 비교**하기 위해 결과를 서로
다른 폴더(`data/checkpoints/v1` vs `data/checkpoints/v2`)에 저장한다.

**v1과의 차이는 폰트 손실 하나뿐이다.** 인코더/헤더 구조, K x M 배치
샘플러, 초중종성 손실(cross entropy), AdamW + warmup·cosine 스케줄,
증강, 체크포인트 포맷은 v1과 완전히 동일하다 — 두 방법을 공정하게
비교하기 위해서다. 따라서 이 문서는 v1과 겹치는 부분은 짧게 요약하고
[train-model-v1.md](train-model-v1.md)로 넘기며, **바뀐 부분(폰트 손실과
그 하이퍼파라미터)을 집중적으로** 설명한다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용법** — 실행 방법, 옵션, 콘솔/체크포인트 읽는 법
- **2부. 상세설계서** — 제안 손실의 수식과 근거, v1과의 차이, 알려진
  제한사항

## 1부. 사용법

### 1.1 이 스크립트가 하는 일 (그리고 하지 않는 일)

`FontGlyphDataset` + `FontGroupBatchSampler`로 배치를 구성하고,
`FontRecognitionModel.encode()`(디코더는 호출하지 않는다)로 얻은
초성/중성/종성/폰트 logits로 손실을 계산해 AdamW + warmup·cosine
스케줄로 학습한다. 여기까지는 v1과 같다.

**v1과 다른 점은 폰트 손실뿐이다.** 폰트 헤더 logit에 softmax cross
entropy 대신, sigmoid 기반의 **Top-$k$ Relaxed Negative curriculum 손실**
(research-paper.md 2.3~2.4절)을 적용한다. 자소(초/중/종성) 손실은 v1과
똑같이 cross entropy 그대로다. 자세한 수식은 2.3절 참고.

**하지 않는 것**: v1과 동일하게 재구성(모드 A/B, `model-design.md`
4.2절), 대조학습(4.4절), 학습/검증 분할(5.1절의 cell-holdout), margin
기반 폰트 분류기(3.6절)는 이번 스크립트의 범위 밖이다. 근거는
[train-model-v1.md](train-model-v1.md) 2.1~2.2절 참고(v2도 같은 이유로
같은 범위를 유지한다).

### 1.2 사전 조건

`data/dataset/index.json`과 그 안에 나열된 PNG가 이미 만들어져 있어야
한다(먼저 [construct-dataset.md](construct-dataset.md)의 도구를 실행).
v1과 같은 데이터셋을 그대로 쓴다.

### 1.3 실행 방법

```bash
bin\train-model-v2.bat
```

또는 직접:

```bash
uv run python scripts/train-model-v2.py
```

`bin\train-model-v2.bat`는 다른 `bin\*.bat`와 동일하게 `VIRTUAL_ENV`를
비운 뒤 프로젝트의 `uv` `.venv`로 실행한다. `--num-workers`를 1 이상으로
쓸 계획이면(기본값 4) 반드시 이 방식(스크립트를 `if __name__ ==
"__main__":` 아래에서 실행)으로 실행해야 한다 — Windows의 spawn 기반
multiprocessing 요구사항이며, `dataset-loader.md`/`batch-sampler.md`의
같은 요구사항과 동일한 이유다.

### 1.4 주요 옵션

v1과 공통인 옵션은 [train-model-v1.md](train-model-v1.md) 1.4절과 같다.
**기본 체크포인트 위치와 Top-$k$ Relaxed Negative Learning 전용 옵션만
다르다.**

공통 옵션(기본값은 v1과 동일):

| 옵션                      | 기본값                    | 설명                                                                      |
| ------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| `--dataset-dir`           | `data/dataset`            | 학습에 쓸 데이터셋 위치                                                   |
| `--checkpoint-dir`        | **`data/checkpoints/v2`** | 체크포인트/`metrics.jsonl` 저장 위치 (v1은 `data/checkpoints/v1`)         |
| `--resume`                | (없음)                    | 이 경로의 체크포인트에서 이어서 학습                                      |
| `--epochs`                | 30                        | 총 epoch 수                                                               |
| `--fonts-per-batch`, `-K` | 32                        | `FontGroupBatchSampler`의 K                                               |
| `--chars-per-font`, `-M`  | 8                         | `FontGroupBatchSampler`의 M                                               |
| `--lr`                    | 3e-4                      | AdamW 학습률(피크, warmup 이후)                                           |
| `--weight-decay`          | 0.01                      | AdamW weight decay                                                        |
| `--warmup-steps`          | 500                       | **학습률(LR)** 선형 warmup 스텝 수 (curriculum warm-up과 무관, 아래 주의) |
| `--grad-clip`             | 1.0                       | gradient norm clipping 임계값                                             |
| `--lambda-jamo`           | 1.0                       | 자소 손실 가중치 (`model-design.md` 4.1절)                                |
| `--lambda-font`           | 1.0                       | 전체 손실에서 폰트 손실(`L_curr`) 항에 곱하는 가중치                      |
| `--max-cache-bytes`       | ~3.85GiB                  | `FontGlyphDataset` LRU 캐시 예산                                          |
| `--prescan-workers`       | 8                         | 데이터셋 초기화 시 병렬 스캔 스레드 수                                    |
| `--num-workers`           | 4                         | `DataLoader` 워커 프로세스 수 (메모리 주의: v1 1.6절)                     |
| `--device`                | 자동(cuda 우선)           | `cuda`, `cuda:0`, `cpu` 등                                                |
| `--no-amp`                | (꺼짐)                    | bfloat16 자동 혼합 정밀도를 끈다                                          |
| `--checkpoint-every`      | 1                         | 이 epoch 수마다 번호 붙은 체크포인트 저장                                 |
| `--log-every`             | 50                        | 이 스텝마다 진행 상황 출력 + `metrics.jsonl` 기록                         |
| `--seed`                  | (없음)                    | 지정하면 모델 초기화/셔플을 시드 - 완전한 재현은 안 됨(2.6절)             |

Top-$k$ Relaxed Negative Learning 전용 옵션(research-paper.md 2.3~2.4절):

| 옵션              | 기본값 | 논문 기호           | 설명                                                                                                         |
| ----------------- | ------ | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--topk-k`        | 5      | $k$                 | 정답을 제외하고 negative 손실에서 빼줄 ambiguous candidate 개수                                              |
| `--lambda-neg`    | 1.0    | $\lambda$           | relaxed negative loss 강도 (`L_warm`/`L_TRN` 공통)                                                           |
| `--beta-sparse`   | 1e-4   | $\beta$             | sparsity 정규화 강도 — 비정답 약 C개 activation의 **합**에 곱하므로 $\lambda$보다 훨씬 작게 둔다(2.3절 주의) |
| `--warmup-epochs` | 5      | $T_{\mathrm{warm}}$ | curriculum warm-up epoch 수. 이 전까지는 $\alpha=0$이라 `L_warm`(단일 라벨)만 쓴다                           |
| `--ramp-epochs`   | 10     | $T_{\mathrm{ramp}}$ | warm-up 이후 $\alpha$를 0→1로 선형 증가시키는 epoch 수                                                       |

> **주의: `--warmup-steps`와 `--warmup-epochs`는 다른 개념이다.**
> `--warmup-steps`는 v1과 똑같은 **학습률(LR) 스케줄**의 선형 warmup
> 스텝 수다. `--warmup-epochs`는 이 스크립트에서 새로 추가된 **손실
> curriculum**의 warm-up 구간(단일 라벨만 쓰는 초기 epoch 수)이다. 서로
> 독립적으로 동작한다.

기본값으로는 epoch 0~4는 warm-up(단일 라벨), epoch 5~14는 $\alpha$가
0→1로 증가, epoch 15~29는 완전한 Top-$k$ Relaxed Negative Loss로 학습한다.

### 1.5 콘솔 출력 읽는 법

```text
Device: cuda
Loading dataset from D:\dev\font-classifier\data\dataset ...
[FontGlyphDataset] scanned 3480/3480 font(s)...
8035120 valid sample(s) across 3480 font(s)
Mixed precision (bf16): on
Top-k Relaxed Negative Learning: k=5 lambda=1.0 beta=0.0001 warmup_epochs=5 ramp_epochs=10
epoch 0: curriculum alpha=0.000 (warm-up only)
epoch 0 step 50 loss=12.3456 (jamo=9.8765 font=2.4691 [warm=2.4691 trn=2.1030]) acc[cho=0.045 jung=0.052 jong=0.041 syllable=0.002 font=0.031 font_top5=0.104] alpha=0.00 lr=3.00e-05 (812 samples/s)
...
epoch 10: curriculum alpha=0.500 (ramping/relaxed)
...
Saved checkpoint: data\checkpoints\v2\checkpoint-epoch-0000.pt
```

- 각 epoch 시작 시 그 epoch의 curriculum 계수 `alpha`를 출력한다
  (`alpha=0`이면 warm-up만, 그 외에는 relaxed 손실이 섞였다는 뜻).
- `loss`는 `lambda_jamo*(CE_cho+CE_jung+CE_jong) + lambda_font*L_curr`.
  괄호 안 `jamo`는 가중치 곱하기 전 자소 손실, `font`는 실제로 학습에
  쓰인 폰트 손실 `L_curr`이다.
- `font` 뒤 대괄호 `[warm=... trn=...]`는 참고용으로, 같은 배치에 대해
  `L_warm`(warm-up 손실)과 `L_TRN`(Top-$k$ Relaxed Negative Loss)을 따로
  계산해 보여준 값이다. `L_curr = (1-alpha)*warm + alpha*trn`이므로
  `alpha=0`이면 `font`와 `warm`이 같고, `alpha=1`이면 `font`와 `trn`이
  같다(2.3절). curriculum이 진행되며 폰트 손실이 warm → trn으로 옮겨
  가는 과정을 이 두 값으로 관찰할 수 있다.
- `acc[...]`의 여섯 값은 각각 초성/중성/종성 개별 정확도, 셋이 모두 맞은
  음절(syllable) 정확도, 폰트 top-1 정확도, 폰트 **top-5** 정확도(`font_top5`
  — 정답 폰트가 상위 5개 logit 안에 들면 정답)다. sigmoid를 써도 폰트
  top-1/top-5는 `argmax`/`topk(logit)`으로 v1과 동일하게 계산하므로 v1의
  `font_acc`/`font_top5_acc`와 직접 비교할 수 있다(2.5절). **특히 이
  방법에서는 유사 폰트를 허용하므로 top-1이 낮아도 `font_top5`가 오르는지가
  학습 진행의 더 좋은 신호다**(2.5절). **전부 그 순간의 학습 배치 기준이며
  검증(held-out) 지표가 아니다** — v1과 같은 한계다([train-model-v1.md]
  (train-model-v1.md) 2.2절).
- 모든 값은 직전 `--log-every` 스텝 구간의 평균이며, 출력 후 누적치를
  초기화한다.
- 같은 정보가 `data/checkpoints/v2/metrics.jsonl`에 한 줄짜리 JSON으로도
  쌓인다. v2의 JSON에는 v1에 없는 `alpha`, `loss_font_warm`,
  `loss_font_trn` 키가 추가로 들어 있다.

### 1.6 메모리 주의사항: `--num-workers` x `--max-cache-bytes`

v1과 완전히 동일하다 — 실제 캐시 메모리 사용량은 대략
`--max-cache-bytes × --num-workers`다. 자세한 내용은
[train-model-v1.md](train-model-v1.md) 1.6절 참고.

### 1.7 체크포인트와 재개

v1과 동일한 방식/포맷이며 저장 위치만 `data/checkpoints/v2`다.

- 매 `--checkpoint-every` epoch(및 마지막 epoch)마다
  `checkpoint-epoch-<번호>.pt`와, 항상 최신 상태를 가리키는 `latest.pt`를
  `--checkpoint-dir`에 저장한다.
- `Ctrl+C`로 중단하면 `interrupted.pt`를 저장한 뒤 종료한다.
- 체크포인트에는 모델/옵티마이저/스케줄러 state와 `epoch`, `global_step`,
  그리고 그때 쓰인 CLI 인자 전체(Top-$k$ 하이퍼파라미터 포함)가 들어 있다.
- 이어서 학습하려면 `--resume <경로>`를 준다(보통 `latest.pt`). 저장된
  epoch 다음 epoch부터 다시 시작하며, curriculum `alpha`도 그 epoch
  번호에 맞춰 이어진다(2.4절).

```bash
uv run python scripts/train-model-v2.py --resume data\checkpoints\v2\latest.pt
```

## 2부. 상세설계서

### 2.1 왜 폰트 손실만 바꾸는가

문제의 위치가 폰트 헤더로 한정되어 있기 때문이다. baseline(v1)에서
자소 헤더는 잘 수렴하는 반면 폰트 헤더는 수렴이 나쁘고, research-paper.md는
그 원인을 **시각적으로 유사한 폰트 클래스 간의 모호성을 무시하는 단일
라벨(softmax CE) 학습**으로 진단한다(1~2절). 따라서 제안 방법도 폰트
분류 목표에만 적용된다.

그래서 v2는 인코더·자소 헤더·배치 샘플러·옵티마이저·스케줄러·증강·
체크포인트 포맷을 전부 v1과 동일하게 두고, **폰트 손실 항 하나만**
Top-$k$ Relaxed Negative curriculum 손실로 교체했다. 이렇게 해야 나중에
성능을 비교했을 때 차이를 "폰트 손실 방법의 차이"로 귀속할 수 있다
(다른 변수를 통제한다). 자소 손실은 `F.cross_entropy` 그대로다.

**모델 구조는 바꾸지 않았다.** 폰트 헤더(`FontHead`)는 여전히 `style`
(512, L2 정규화) → `Linear`로 클래스별 logit을 낸다. 제안 방법은 이
logit을 softmax가 아니라 **클래스별 독립 sigmoid**로 해석할 뿐이라,
가중치를 하나도 추가하지 않고 손실 함수 안에서만 처리한다
(research-paper.md 2.1절).

### 2.2 학습/검증 분할을 넣지 않은 이유

v1과 동일하게 이번에도 cell-holdout 분할을 넣지 않았다. **따라서 1.5절의
정확도 지표는 전부 학습 데이터 기준이며 일반화 성능이 아니다.** 근거와
향후 구현 방향은 [train-model-v1.md](train-model-v1.md) 2.2절과 같다.

> 참고: 시각적 유사 폰트에 대한 relaxation이 실제로 도움이 되는지는
> 궁극적으로 held-out 정확도로 판단해야 한다. 현재 지표(학습 배치 기준)
> 로는 v1/v2의 학습 손실·학습 정확도 추이를 비교할 수 있을 뿐이며,
> 최종 성능 비교에는 cell-holdout 평가가 별도로 필요하다(향후 작업).

### 2.3 제안 손실: Top-$k$ Relaxed Negative curriculum (research-paper.md 2.3~2.4절)

폰트 손실은 배치 평균 기준으로 다음과 같다. 클래스 수를 $C$, 정답 폰트를
$y$, logit을 $z_c$, sigmoid activation을 $p_c=\sigma(z_c)$라 하자.

**정답(공통):**

$$
\mathcal{L}_{\mathrm{pos}} = -\log \sigma(z_{y})
$$

**warm-up 손실**(모든 비정답을 음성으로 취급하는 단일 라벨 binary loss):

$$
\mathcal{L}_{\mathrm{warm}} = \mathcal{L}_{\mathrm{pos}}
  - \lambda \frac{1}{C-1} \sum_{c \neq y} \log\left(1-\sigma(z_c)\right)
$$

**Top-$k$ Relaxed Negative Loss**(상위 $k$개 ambiguous candidate를 negative
에서 제외):

$$
\mathcal{A} = \operatorname{TopK}_{c \neq y}(p_c), \qquad
\mathcal{N}^{\mathrm{relaxed}} = \{1,\dots,C\} \setminus (\{y\} \cup \mathcal{A})
$$

$$
\mathcal{L}_{\mathrm{TRN}} = \mathcal{L}_{\mathrm{pos}}
  - \lambda \frac{1}{|\mathcal{N}^{\mathrm{relaxed}}|} \sum_{c \in \mathcal{N}^{\mathrm{relaxed}}} \log\left(1-\sigma(z_c)\right)
  + \beta \sum_{c \neq y} \sigma(z_c)
$$

**curriculum 결합**:

$$
\mathcal{L}_{\mathrm{curr}}(t) = (1-\alpha_t)\,\mathcal{L}_{\mathrm{warm}} + \alpha_t\,\mathcal{L}_{\mathrm{TRN}}
$$

핵심 설계 결정과 그 근거:

- **후보 선택은 그래디언트를 흘리지 않는다.** $\mathcal{A}$는 현재
  activation `torch.sigmoid(z).detach()`에서 고른다(집합 선택이므로
  미분 대상이 아니다). 정답 $y$는 선택 전 `-inf`로 마스킹해 후보에서
  제외한다. 선택된 후보는 pseudo-label로 정답 취급되지 않고, 단지
  negative 손실에서 빠질 뿐이다(research-paper.md 2.5절 — noisy
  pseudo-labeling과의 차이).
- **수치 안정성**: $\log\sigma(z)=$ `F.logsigmoid(z)`,
  $\log(1-\sigma(z))=$ `F.logsigmoid(-z)`로 계산해 큰 $|z|$에서도 안정적
  이다. 손실은 autocast(bf16) 안에서도 `out.font_logits.float()`로 fp32
  캐스팅해 계산한다(logsigmoid/topk 정밀도 확보).
- **sparsity 항($\beta$)의 스케일 주의**: $\sum_{c\neq y}\sigma(z_c)$는
  약 $C-1$개(현재 ~3,479개) activation의 **합**이라 값 자체가 크다. 그래서
  $\beta$ 기본값을 1e-4로 아주 작게 두었다($\lambda$보다 훨씬 작게 —
  research-paper.md 2.3절이 요구하는 관계). warm-up 이후 대부분의
  비정답 activation이 이미 눌려 있으므로 이 항은 top-$k$ 후보의 무분별한
  증가만 약하게 억제한다. collapse(모든 클래스 동시 활성화)가 보이면
  $\beta$를 키우고, 유사 폰트에 대한 multi-hot이 과하게 눌리면 줄인다.
- **$k$의 의미**: $k$가 크면 더 많은 유사 폰트를 negative에서 풀어 주지만,
  그만큼 억제되지 않는 클래스가 늘어 식별력이 약해질 수 있다. 3,000여
  클래스에서 "정답과 헷갈릴 만한 소수"를 상정해 기본값 5로 두었다(논문이
  구체적 값을 규정하지는 않는다 — 2.7절).

구현은 `font_loss_topk_relaxed()` 한 함수에 모여 있으며, `L_curr`(학습에
쓰는 값)과 함께 `L_warm`, `L_TRN`을 로그용으로 반환한다.

### 2.4 Curriculum 스케줄 (research-paper.md 2.4절)

`curriculum_alpha()`가 epoch $t$에서 계수 $\alpha_t$를 정한다.

$$
\alpha_t = \begin{cases}
0 & t < T_{\mathrm{warm}} \\
\min\!\left(1,\; \dfrac{t - T_{\mathrm{warm}}}{T_{\mathrm{ramp}}}\right) & t \geq T_{\mathrm{warm}}
\end{cases}
$$

- **$t$는 epoch 단위로 쓴다.** 논문은 epoch 또는 iteration 둘 다
  허용하는데, epoch 단위가 로그로 추적·재현하기 쉽고 warm-up "이후"라는
  개념과 잘 맞아 epoch을 택했다(가정).
- `alpha`는 매 epoch 시작 시 한 번 계산해 그 epoch 내내 고정한다. 즉
  같은 epoch의 모든 배치는 같은 $\alpha$를 쓴다.
- **`--resume` 시에도 일관된다**: `alpha`는 저장된 epoch 번호로부터 다시
  계산되므로, 이어서 학습해도 curriculum 위치가 어긋나지 않는다.
- warm-up 초기에 후보를 신뢰하기 어렵다는 논문의 우려(2.4절)를 그대로
  반영한다 — $T_{\mathrm{warm}}$까지는 단일 라벨로 기본 식별력을 먼저
  쌓고, 그 뒤에야 relaxation을 서서히 켠다.

### 2.5 v1과의 지표 비교 시 주의

- **폰트 손실 값(`loss`/`font`)의 절대 크기는 v1과 직접 비교하면 안
  된다.** v1의 `font`는 softmax cross entropy, v2의 `font`는 sigmoid 기반
  multi-hot 손실이라 스케일 자체가 다르다.
- **비교는 `font_acc`/`font_top5_acc`(그리고 향후 held-out 정확도)로
  한다.** 폰트 top-1/top-5는 두 버전 모두 `argmax`/`topk(logit)`으로 동일
  하게 계산하므로 같은 잣대로 볼 수 있다.
- **top-1보다 `font_top5_acc`를 더 중요하게 본다.** 이 방법의 전제는
  시각적으로 유사한 폰트가 많아 top-1 배타성이 애초에 부적절하다는 것
  이므로(research-paper.md 1~2절), 정답 폰트가 유일한 1등이 되는지보다
  상위 후보군에 드는지가 설계 의도에 맞는 지표다. top-1이 낮게 머물러도
  top-5가 오르면 방법이 의도대로 동작하는 신호이고, top-1·top-5가 함께
  낮으면 실제로 학습이 안 되는 것이다.
- 자소 지표(`cho/jung/jong/syllable_acc`, `jamo` 손실)는 두 버전이
  완전히 같은 손실·구조를 쓰므로, v2에서 자소 성능이 v1과 크게 달라진다면
  그것은 공유 인코더가 폰트 손실 변화의 영향을 받았다는 신호로 해석할 수
  있다(둘은 인코더를 공유한다).

### 2.6 v1에서 그대로 물려받은 설계

아래 항목은 v1과 동일하므로 근거만 옮겨 적는다. 자세한 설명은
[train-model-v1.md](train-model-v1.md)의 같은 절 참고.

- **옵티마이저/스케줄**: AdamW + 선형 warmup 후 cosine decay, gradient
  clipping(기본 1.0). `total_steps`는 근사치(v1 2.4절).
- **Mixed precision**: bf16, `GradScaler` 없음(v1 2.5절). CPU/`--no-amp`
  에서 자동으로 꺼진다.
- **재현성의 한계**: `--num-workers > 0`이면 augmentation RNG가 워커별로
  통제되지 않아 완전 재현은 안 된다(v1 2.6절).
- **체크포인트 포맷**: `model`/`optimizer`/`scheduler`/`epoch`/
  `global_step`/`args` 딕셔너리(v1 2.7절). `args`에 v2의 Top-$k$
  하이퍼파라미터가 추가로 들어간다.

### 2.7 알려진 제한사항

- **held-out 지표 없음**(2.2절) — v1과 동일. 최종 성능 비교에는
  cell-holdout 평가가 별도로 필요하다.
- **하이퍼파라미터 기본값은 논문이 규정한 값이 아니다**(2.3절): $k$,
  $\lambda$, $\beta$, $T_{\mathrm{warm}}$, $T_{\mathrm{ramp}}$ 기본값은
  합리적 출발점일 뿐 튜닝이 필요할 수 있다. 특히 $\beta$는 sparsity 항이
  합 형태라 스케일에 민감하다.
- **curriculum이 epoch 단위**(2.4절): iteration 단위의 더 매끄러운 ramp가
  필요하면 `curriculum_alpha`를 `global_step` 기반으로 바꿔야 한다.
- **재구성/대조학습/margin 분류기 없음**: v1과 같은 범위 한계
  ([train-model-v1.md](train-model-v1.md) 2.8절). 특히 3.6절의
  ArcFace/CosFace 전환은 이 방법과는 다른 접근이며 여기서 다루지 않는다.
- **완전한 재현성 없음 / `--num-workers`의 메모리 곱 증가**: v1과 동일.

### 2.8 검증 방법

이 문서 작성 시점에는 전체 데이터셋 학습(약 3,480종, 800만 표본) 대신
손실 함수의 정확성을 단위 수준에서 확인했다.

- `font_loss_topk_relaxed`가 `L_curr = (1-alpha)*L_warm + alpha*L_TRN`을
  정확히 만족하는지(여러 `alpha`에서).
- 그래디언트가 유한하게 흐르는지.
- **relaxation이 실제로 동작하는지**: activation이 아주 높은 비정답
  클래스를 인위로 만들었을 때, 그 클래스가 ambiguous candidate로 뽑혀
  `L_TRN`의 negative penalty에서 빠지는지(그 결과 `L_TRN < L_warm`).
- `k`가 비정답 클래스 수보다 클 때의 방어 코드(작은 합성 데이터셋 대비).
- `curriculum_alpha` 스케줄이 $T_{\mathrm{warm}}$/$T_{\mathrm{ramp}}$
  경계에서 0 → 선형 증가 → 1로 맞는지.
- 스크립트 임포트/`--help`가 정상 동작하는지.

**아직 확인하지 못한 것**: 실제 전체 데이터셋에서의 수렴 여부, v1 대비
폰트 정확도 개선 여부, 처리량(samples/s), 메모리 사용량. 처음 실행할 때
콘솔의 `font_acc`와 `[warm=... trn=...]` 추이, `samples/s`, 시스템 메모리를
직접 관찰해야 한다. 특히 warm-up이 끝나고 `alpha`가 커지는 구간(기본
epoch 5 이후)에서 `font_acc`가 무너지지 않는지 확인한다.

### 2.9 모듈 구조 요약

v1과 공통인 요소(`parse_args`, `build_lr_lambda`, `RunningAverage`,
`save_checkpoint`/`load_checkpoint`, `main`)에 더해, v2에는 다음 두 함수가
추가되었다.

| 구성 요소                | 역할                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| `curriculum_alpha`       | epoch → curriculum 계수 $\alpha_t$ (2.4절)                                                       |
| `font_loss_topk_relaxed` | Top-$k$ Relaxed Negative curriculum 폰트 손실 `L_curr`(및 로그용 `L_warm`/`L_TRN`) 계산 (2.3절)  |
| `main`                   | v1과 동일한 학습 루프에서 폰트 손실만 `font_loss_topk_relaxed`로 교체, epoch별 `alpha` 계산·로깅 |
