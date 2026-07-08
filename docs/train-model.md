# Train Model — 사용 설명서 및 상세설계서

`scripts/train-model.py`는 `font_classifier/model.py`의
`HangulFontRecognitionModel`을 학습하는 진입점 스크립트다. `docs/model-design.md`
4.5절 로드맵의 **Phase 1(분류 헤드 baseline, 재구성 없음)만** 다룬다 —
왜 이번 범위를 Phase 1로 한정했는지, 그리고 학습/검증 분할을 왜 이번에는
넣지 않았는지는 1.1절과 2.1절에서 근거를 설명한다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용법** — 실행 방법, 옵션, 콘솔/체크포인트 읽는 법
- **2부. 상세설계서** — 범위를 좁힌 이유, 손실/옵티마이저/증강 설계 근거,
  알려진 제한사항

## 1부. 사용법

### 1.1 이 스크립트가 하는 일 (그리고 하지 않는 일)

`FontGlyphDataset` + `FontGroupBatchSampler`로 배치를 구성하고,
`HangulFontRecognitionModel.encode()`(디코더는 호출하지 않는다)로 얻은
초성/중성/종성/폰트 logits에 대해 cross entropy 손실을 계산해
AdamW + warmup·cosine 스케줄로 학습한다.

**하지 않는 것**: 재구성(모드 A/B, `model-design.md` 4.2절), 대조학습
(4.4절), 학습/검증 분할(5.1절의 cell-holdout), margin 기반 폰트 분류기
(3.6절). 전부 이번 스크립트의 범위 밖이며, 근거는 2.1절 참고.

### 1.2 사전 조건

`data/dataset/index.json`과 그 안에 나열된 PNG가 이미 만들어져 있어야
한다(먼저 [construct-dataset.md](construct-dataset.md)의 도구를 실행).

### 1.3 실행 방법

```bash
bin\train-model.bat
```

또는 직접:

```bash
uv run python scripts/train-model.py
```

`bin\train-model.bat`는 다른 `bin\*.bat`와 동일하게 `VIRTUAL_ENV`를 비운
뒤 프로젝트의 `uv` `.venv`로 실행한다. `--num-workers`를 1 이상으로 쓸
계획이면(기본값 4) 반드시 이 방식(스크립트를 `if __name__ ==
"__main__":` 아래에서 실행)으로 실행해야 한다 — Windows의 spawn 기반
multiprocessing 요구사항이며, `dataset-loader.md`/`batch-sampler.md`의
같은 요구사항과 동일한 이유다.

### 1.4 주요 옵션

| 옵션                      | 기본값                                             | 설명                                                           |
| ------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| `--dataset-dir`           | `data/dataset`                                     | 학습에 쓸 데이터셋 위치                                        |
| `--checkpoint-dir`        | `data/checkpoints`                                 | 체크포인트/`metrics.jsonl` 저장 위치                           |
| `--resume`                | (없음)                                             | 이 경로의 체크포인트에서 이어서 학습                           |
| `--epochs`                | 30                                                 | 총 epoch 수                                                    |
| `--fonts-per-batch`, `-K` | 32                                                 | `FontGroupBatchSampler`의 K (batch-sampler.md 참고)            |
| `--chars-per-font`, `-M`  | 8                                                  | `FontGroupBatchSampler`의 M                                    |
| `--lr`                    | 3e-4                                               | AdamW 학습률(피크, warmup 이후)                                |
| `--weight-decay`          | 0.01                                               | AdamW weight decay                                             |
| `--warmup-steps`          | 500                                                | 선형 warmup 스텝 수                                            |
| `--grad-clip`             | 1.0                                                | gradient norm clipping 임계값                                  |
| `--lambda-jamo`           | 1.0                                                | `model-design.md` 4.1절 표의 초기값                            |
| `--lambda-font`           | 1.0                                                | 〃                                                             |
| `--max-cache-bytes`       | `dataset_loader.DEFAULT_MAX_CACHE_BYTES`(~3.85GiB) | `FontGlyphDataset` LRU 캐시 예산                               |
| `--prescan-workers`       | 8                                                  | 데이터셋 초기화 시 병렬 스캔 스레드 수                         |
| `--num-workers`           | 4                                                  | `DataLoader` 워커 프로세스 수 - **1.6절 메모리 주의사항 참고** |
| `--device`                | 자동(cuda 우선)                                    | `cuda`, `cuda:0`, `cpu` 등                                     |
| `--no-amp`                | (꺼짐)                                             | bfloat16 자동 혼합 정밀도를 끈다                               |
| `--checkpoint-every`      | 1                                                  | 이 epoch 수마다 번호 붙은 체크포인트 저장                      |
| `--log-every`             | 50                                                 | 이 스텝마다 진행 상황 출력 + `metrics.jsonl` 기록              |
| `--seed`                  | (없음)                                             | 지정하면 모델 초기화/셔플을 시드 - 완전한 재현은 안 됨(2.6절)  |

### 1.5 콘솔 출력 읽는 법

```text
Device: cuda
Loading dataset from D:\dev\font-classifier\data\dataset ...
[FontGlyphDataset] scanned 3480/3480 font(s)...
8035120 valid sample(s) across 3480 font(s)
Mixed precision (bf16): on
epoch 0 step 50 loss=12.3456 (jamo=9.8765 font=2.4691) acc[cho=0.045 jung=0.052 jong=0.041 syllable=0.002 font=0.031 font_top5=0.104] lr=3.00e-05 (812 samples/s)
...
Saved checkpoint: data\checkpoints\checkpoint-epoch-0000.pt
```

- `loss`는 `lambda_jamo*(CE_cho+CE_jung+CE_jong) + lambda_font*CE_font`
  (기본 가중치 둘 다 1.0, `model-design.md` 4.1절). 괄호 안 `jamo`/`font`는
  가중치 곱하기 전의 원래 손실값이다.
- `acc[...]`의 여섯 값은 각각 초성/중성/종성 개별 정확도, 셋이 모두 맞은
  음절(syllable) 정확도, 폰트 top-1 정확도, 폰트 top-5 정확도(`font_top5`
  — 정답 폰트가 상위 5개 logit 안에 들면 정답)다(`model-design.md` 6.1절이
  요구하는 지표와 이름을 맞췄다). **전부 그 순간의 학습 배치 기준이다 —
  검증(held-out) 지표가 아니다(2.1절).**
- 모든 값은 직전 `--log-every` 스텝 구간의 평균이며, 출력 후 누적치를
  초기화한다.
- 같은 정보가 `data/checkpoints/metrics.jsonl`에 한 줄짜리 JSON으로도
  쌓인다(그래프를 그리거나 나중에 분석할 때 사용).

### 1.6 메모리 주의사항: `--num-workers` x `--max-cache-bytes`

[dataset-loader.md](dataset-loader.md) 1.5절과 동일한 이유로, 실제 캐시
메모리 사용량은 대략 `--max-cache-bytes × --num-workers`다(워커마다
별도 프로세스와 별도 캐시). 기본값(3.85GiB x 4)이면 최대 약 15.4GiB를
쓸 수 있다는 뜻이다. RAM이 부족하면 `--num-workers`를 줄이거나
`--max-cache-bytes`를 그만큼 나눠 줄인다.

### 1.7 체크포인트와 재개

- 매 `--checkpoint-every` epoch(및 마지막 epoch)마다
  `checkpoint-epoch-<번호>.pt`와, 항상 최신 상태를 가리키는 `latest.pt`를
  `--checkpoint-dir`에 저장한다.
- `Ctrl+C`로 중단하면 `interrupted.pt`를 저장한 뒤 종료한다.
- 체크포인트에는 모델/옵티마이저/스케줄러 state와 `epoch`, `global_step`,
  그리고 그때 쓰인 CLI 인자 전체가 들어 있다.
- 이어서 학습하려면 `--resume <경로>`를 준다(보통 `latest.pt`). 저장된
  epoch 다음 epoch부터 다시 시작한다.

```bash
uv run python scripts/train-model.py --resume data\checkpoints\latest.pt
```

## 2부. 상세설계서

### 2.1 왜 이번 범위를 Phase 1로 한정했는가

`model-design.md` 4.5절은 Phase 1(분류 헤드 baseline) → Phase 2(모드
A 재구성 추가) → Phase 3(모드 B + 대조학습 추가)로 로드맵을 나눈다. 이번
스크립트는 Phase 1만 구현했다.

- **모드 B(4.2절)는 이번 범위에 넣을 수 없었다**: 목표 이미지와 다른
  폰트의 같은 글자(구조 소스), 같은 폰트의 다른 글자(스타일 소스)를
  배치 안에서 짝지어야 하는데, 이는
  [batch-sampler.md](batch-sampler.md) 2.5절이 명시한 대로 아직 구현되지
  않은 "같은 M개 글자를 K개 폰트에 공통 적용하는 그리드" 샘플러가
  있어야 가능하다. 그 샘플러 자체가 별도 작업이다.
- **모드 A(self-reconstruction)는 `model.forward()`만 부르면 될 만큼
  "공짜"처럼 보였지만 그렇지 않다**: 4.2절이 말하는 진짜
  denoising(노이즈 낀 입력 → **깨끗한** 목표를 복원)을 하려면
  `FontGlyphDataset`이 한 샘플에 대해 증강된 뷰와 원본 뷰를 함께
  반환해야 하는데, 이는 [dataset-loader.md](dataset-loader.md) 2.9절이
  이미 "대조학습/재구성용 다중 view 미지원 - Phase 3 도입 시 별도
  wrapper/Dataset이나 이 클래스의 확장이 필요"라고 명시적으로 미뤄 둔
  항목이다. (증강된 이미지를 그대로 다시 복원하는 단순 autoencoding으로
  근사할 수도 있었지만, 이번에는 Phase 1만 하기로 결정해 아예 포함하지
  않았다.)
- 따라서 이번 스크립트는 `model.encode()`만 호출하고 `decode()`는 전혀
  쓰지 않는다 - 디코더 파라미터는 존재하지만 이 스크립트로는 학습되지
  않는다(옵티마이저 대상에는 포함되므로 gradient가 없을 뿐 에러가 나지는
  않는다 - `model.parameters()` 전체를 최적화 대상으로 넘기지만
  `encode()` 경로에 없는 디코더 파라미터는 단순히 grad가 `None`으로
  남아 업데이트되지 않는다).

### 2.2 학습/검증 분할을 넣지 않은 이유

`model-design.md` 5.1절은 폰트마다 10~15% 글자를 held-out으로 남기는
cell-holdout 분할을 규정하고, 9절의 성공 기준도 이 분할 기준으로
서술되어 있다. 이번 스크립트는 이 분할을 구현하지 않기로 결정했다 -
**따라서 1.5절의 정확도 지표는 전부 학습 데이터 기준이며, 실제 일반화
성능(held-out 정확도)을 보여주지 않는다.** 학습이 정상적으로 도는지,
손실이 줄어드는지 정도의 진행 상황 확인용으로만 써야 한다. cell-holdout
분할은 다음 작업에서 다룬다 - 구현하려면 `FontGlyphDataset`을
`augment=True`/`augment=False`로 두 벌 만들고(둘 다 같은 파일에서
결정론적으로 같은 유효 인덱스를 만들어 내므로 인덱스를 그대로
재사용할 수 있다), `FontGroupBatchSampler`에 인덱스 필터링 옵션을
추가해야 한다 - 이번 스크립트를 만들면서 검토했지만 범위에서 뺐다.

### 2.3 손실 함수 (`model-design.md` 4.1절)

```text
loss = lambda_jamo * (CE_cho + CE_jung + CE_jong) + lambda_font * CE_font
```

재구성/대조학습 항(`lambda_reconA`/`lambda_reconB`/`lambda_cont`)은
2.1절의 이유로 이번 스크립트에 없다. `lambda_jamo`/`lambda_font` 기본값
(둘 다 1.0)은 4.1절 표를 그대로 따른다.

### 2.4 옵티마이저와 학습률 스케줄 (4.6절)

- **AdamW** + **선형 warmup 후 코사인 감쇠**(`build_lr_lambda`). 4.6절이
  "AdamW + cosine decay"만 명시하고 warmup 여부는 (margin 기반 분류기
  맥락에서만) 언급하므로, 이번 구현은 일반적인 관행에 따라 warmup을
  기본으로 넣었다(가정).
- 코사인 곡선의 분모로 쓰는 `total_steps = len(sampler) * epochs`는
  [batch-sampler.md](batch-sampler.md) 2.4절이 설명하듯 근사치다. 실제
  스텝 수가 조금 다르더라도 코사인 곡선의 전체적인 모양(초반 낮음 →
  피크 → 후반 감쇠)에는 큰 영향이 없다.
- **Gradient clipping**(기본 max-norm 1.0)도 4.6절이 명시한 항목이다.

### 2.5 Mixed precision: bf16, GradScaler 없음

4.6절은 "mixed precision(fp16/bf16)"이라고만 적어 두 방식 중 하나를
선택해야 했다. bf16을 택한 이유:

- bf16은 fp32와 지수부(exponent) 범위가 같아 fp16처럼 언더/오버플로를
  막기 위한 `GradScaler`(동적 loss scaling)가 필요 없다 - 코드가
  간단해지고 스케일링 관련 버그 여지가 없다.
- 이 프로젝트가 개발 중인 GPU(RTX 3090, Ampere 세대)는 bf16 텐서 코어를
  네이티브로 지원한다.
- CPU나 bf16을 지원하지 않는 환경에서는 `--no-amp`로 끄거나, `--device
  cpu`를 쓰면 자동으로 꺼진다(`use_amp = (not args.no_amp) and
  device.type == "cuda"`).

### 2.6 재현성의 한계

`--seed`를 주면 `random`/`numpy`/`torch`의 전역 시드를 고정한다. 이
시드는 **모델 가중치 초기화와, `FontGroupBatchSampler`가 메인
프로세스에서 수행하는 폰트/글자 셔플(배치 구성)까지는 재현한다.** 하지만
`--num-workers > 0`이면 augmentation은 각 `DataLoader` 워커 프로세스
안에서 일어나고, [dataset-loader.md](dataset-loader.md) 2.8절이 설명한
대로 각 워커는 자신만의(시드로 통제되지 않는) 전역 RNG로 다시
시작한다 - 따라서 같은 `--seed`로 두 번 실행해도 augmentation 결과나
정확한 손실값까지 완전히 똑같지는 않다. 완전한 재현이 필요하면
`--num-workers 0`으로 실행해야 한다(속도는 느려진다).

### 2.7 체크포인트 포맷

`torch.save`로 아래 딕셔너리를 저장한다.

| 키            | 내용                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `HangulFontRecognitionModel.state_dict()` (인코더/헤더/디코더 전부 - 디코더는 2.1절 이유로 학습되지 않은 초기값 그대로)                         |
| `optimizer`   | `AdamW.state_dict()`                                                                                                                            |
| `scheduler`   | `LambdaLR.state_dict()`                                                                                                                         |
| `epoch`       | 이 체크포인트를 저장한 시점의 epoch 번호(0-based)                                                                                               |
| `global_step` | 누적 스텝 수                                                                                                                                    |
| `args`        | 그 실행에 쓰인 전체 CLI 인자(`Path`는 문자열로 변환) - 참고용, 재개 시 자동으로 다시 적용되지는 않는다(재개 시에도 CLI 인자를 다시 넘겨야 한다) |

### 2.8 알려진 제한사항

- **held-out 지표 없음**(2.2절) - 지금은 학습 손실/정확도만 볼 수 있다.
- **재구성/대조학습 없음**(2.1절) - 디코더는 만들어지지만 학습되지 않는다.
- **margin 기반 폰트 분류기로의 전환 없음**: `model-design.md` 3.6절이
  예정한 ArcFace/CosFace 전환은 이 스크립트에 없다 -
  `FontHead.classifier`를 손으로 교체해야 한다.
- **`total_steps`가 근사치**(2.4절) - 코사인 스케줄의 정확한 종료 지점이
  약간 어긋날 수 있다.
- **완전한 재현성 없음**(2.6절).
- **`--num-workers`가 메모리를 곱으로 늘림**(1.6절) - `dataset-loader.md`
  에서 이미 문서화된 한계를 그대로 물려받는다.
- **uncertainty weighting 없음**: 4.1절이 언급한 "수동 튜닝이 부담스러워
  지면" 전환할 자동 손실 가중 기법은 구현하지 않았다 - 지금은 손실
  항이 둘(`lambda_jamo`/`lambda_font`)뿐이라 수동 튜닝 부담이 크지
  않다고 보았다.

### 2.9 검증 방법

`data/dataset` 전체(약 3,480종, 800만 표본)로 실제 학습을 도는 대신,
합성 데이터셋(폰트 몇 종, CPU)으로 다음을 확인했다.

- 1 epoch를 끝까지 돌려 손실이 전반적으로 감소하고, 학습률이 warmup →
  피크 → 코사인 감쇠 → 0으로 정확히 움직이는지.
- 체크포인트가 저장되고, `--resume`으로 이어서 시작한 epoch 번호가
  올바른지.
- `Ctrl+C` 중단 시 `interrupted.pt`가 저장되는지.

실제 전체 데이터셋으로 며칠 단위 학습을 돌렸을 때의 수렴 여부, 실제
처리량(samples/s), 메모리 사용량은 이 문서 작성 시점에는 확인하지
못했다 - 처음 실행할 때 콘솔의 `samples/s`와 시스템 메모리 사용량을
직접 관찰해야 한다.

### 2.10 모듈 구조 요약

| 구성 요소                             | 역할                                                              |
| ------------------------------------- | ----------------------------------------------------------------- |
| `parse_args`                          | CLI 인자 정의 (1.4절)                                             |
| `build_lr_lambda`                     | warmup + cosine decay 스케줄 (2.4절)                              |
| `RunningAverage`                      | 로그 구간 평균 계산기                                             |
| `save_checkpoint` / `load_checkpoint` | 체크포인트 저장/복원 (2.7절)                                      |
| `main`                                | 데이터셋/샘플러/모델/옵티마이저 구성, 학습 루프, 로깅, 체크포인트 |
