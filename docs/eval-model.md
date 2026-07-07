# Eval Model

학습된 폰트 인식 체크포인트를 데이터셋 전체에 대해 평가하고, 한글/폰트 인식
성능과 인식 속도를 측정해 `data/results/eval.json`에 기록하는 스크립트
(`scripts/eval-model.py`).

```bash
bin/eval-model.sh        # 또는  uv run python scripts/eval-model.py
```

## 1. 무엇을 어떻게 평가하나

`scripts/train-model.py`와 **같은** `FontGlyphDataset`(폰트 중심
`data/dataset`)을 읽되, 학습과 달리 augmentation을 끄고(깨끗한 입력) 유효한
(폰트, 글자) 표본 전체를 순회한다. 기본값은 전체(100%) 평가이며, 필요하면
데이터셋에서 임의 추출한 일부 비율만 평가할 수 있다. 지표 정의는
train-model.py의 학습 로그와 맞춰, 같은 축으로 비교할 수 있게 했다.

- **한글**
  - `cho_acc` / `jung_acc` / `jong_acc`: 초/중/종성 각 argmax 정확도.
  - `syllable_acc`: 셋 다 맞은 비율(학습 로그의 음절 정확도와 같은 정의).
  - `restricted_char_acc`: 제한 디코딩(2,350자 표, `model.decode_restricted`)
    으로 얻은 글자가 정답 글자와 같은 비율.
  - `open_char_acc`: 개방 디코딩(11,172자, `model.decode_open`)의 글자 정확도.
- **폰트**: `top1_acc` / `top5_acc` / `top10_acc`.
- **속도**: 한 글자를 인식(encode → 제한/개방 디코딩 → 폰트 top-k)하는 순수
  연산 시간. 데이터 로딩과 정답 대조 시간은 제외하며, 첫 배치는 워밍업으로
  빼고 CUDA에서는 `torch.cuda.synchronize()`로 커널 완료를 기다린 뒤 잰다.
  `samples_per_second`와 `ms_per_sample`로 보고한다.

## 2. 명령행 옵션

| 옵션                | 기본값                       | 설명                                                                                     |
| ------------------- | ---------------------------- | ---------------------------------------------------------------------------------------- |
| `--checkpoint`      | `data/checkpoints/latest.pt` | 평가할 `.pt` 파일                                                                        |
| `--dataset-dir`     | `data/dataset`               | 평가 대상 데이터셋 폴더                                                                  |
| `--output`          | `data/results/eval.json`     | 결과 JSON 경로                                                                           |
| `--batch-size`      | 256                          | 평가 배치 크기                                                                           |
| `--num-workers`     | 4                            | DataLoader 워커 수                                                                       |
| `--prescan-workers` | 8                            | 데이터셋 초기 스캔 스레드 수                                                             |
| `--sample-percent`  | 100                          | 평가에 사용할 데이터셋 비율. 10이면 전체에서 임의로 10%를 뽑아 평가한다.                |
| `--device`          | cuda 있으면 cuda             | 실행 장치                                                                                |
| `--no-amp`          | (cuda에서 켜짐)              | bfloat16 자동 혼합 정밀도 끄기                                                           |
| `--log-every`       | 1000                         | 진행 상황 출력 주기(배치)                                                                |

폰트 클래스 수는 체크포인트의 폰트 분류기 가중치 크기에서 직접 읽어 모델을
구성한다(train-model.py의 저장 형식과 같은 `model` 키).

## 3. eval.json 형식

```json
{
  "checkpoint": "…/data/checkpoints/latest.pt",
  "dataset_dir": "…/data/dataset",
  "device": "cuda",
  "amp_bf16": true,
  "num_font_classes_checkpoint": 3480,
  "num_font_classes_dataset": 3480,
  "num_samples_evaluated": 8177436,
  "sample_percent": 100.0,
  "hangul": { "cho_acc": …, "jung_acc": …, "jong_acc": …,
              "syllable_acc": …, "restricted_char_acc": …, "open_char_acc": … },
  "font": { "top1_acc": …, "top5_acc": …, "top10_acc": … },
  "speed": { "batch_size": …, "timed_samples": …, "inference_seconds": …,
             "samples_per_second": …, "ms_per_sample": …, "note": "…" },
  "wall_seconds": …,
  "timestamp": "2026-07-06T20:59:13"
}
```

## 4. 전제: 체크포인트와 데이터셋의 정합성

폰트 라벨은 `data/dataset/index.json`의 `id - 1`이다. 체크포인트의 폰트 클래스
순서가 이 데이터셋과 어긋나면(예: 서로 다른 시점의 산출물) **폰트 지표만 크게
낮게** 나온다 — 한글 지표는 폰트 순서와 무관하므로 정상이다. 두 폰트 클래스
수가 다르면 실행 시 경고를 출력하고 `num_font_classes_checkpoint` /
`num_font_classes_dataset`를 결과에 함께 남기므로, 폰트 정확도가 유독 낮으면
가장 먼저 이 정합성을 의심한다(같은 주의사항이
[docs/font-classifier.md](font-classifier.md) 4절에도 있다).
