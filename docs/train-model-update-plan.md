# Train Model Update Plan

이 문서는 `scripts/generate_split.py`가 만든 cell-holdout manifest를 `scripts/train-model.py`에 통합하는 작업 계획이다.

## 1. 목표

- train은 cell-holdout train split만 사용한다.
- validation은 같은 manifest의 val split을 augmentation 없이 평가한다.
- `metrics.jsonl`에는 학습 지표와 validation 지표를 구분해 기록한다.
- 조기 정지는 validation 지표를 기준으로 수행한다.
- 기존 전체 데이터 학습 경로는 필요하면 유지하되, 새 학습의 기본 권장은 split manifest 사용이다.

## 2. 현재 상태

현재 `train-model.py`는 `FontGlyphDataset(augment=True)` 전체를 하나의 DataLoader로 학습한다. 로그에 찍히는 accuracy는 모두 학습 배치 기준이며, held-out validation 지표가 아니다.
`model-design.md`는 기본 평가 전략으로 per-font cell-holdout을 전제한다. 따라서 학습 루프가 validation split을 직접 읽고 평가하도록 바꾸는 것이 다음 단계다.

## 3. 새 CLI

권장 옵션:

| 옵션                      | 기본값                               | 설명                                            |
| ------------------------- | ------------------------------------ | ----------------------------------------------- |
| `--split-manifest`        | `data/splits/cell-holdout-seed.json` | `generate_split.py`가 만든 JSON manifest        |
| `--train-split`           | `train`                              | 학습에 사용할 split 이름                        |
| `--val-split`             | `val`                                | validation에 사용할 split 이름                  |
| `--validate-every`        | `1`                                  | 몇 epoch마다 validation을 실행할지              |
| `--early-stop`            | 꺼짐                                 | validation 기반 조기 정지 사용 여부             |
| `--early-stop-min-epochs` | `3`                                  | 이 epoch 이전에는 멈추지 않음                   |
| `--early-stop-patience`   | `1`                                  | 개선 없이 기다릴 validation 횟수                |
| `--early-stop-min-delta`  | `0.0015`                             | `val_font_acc` 개선으로 인정할 최소 변화        |
| `--early-stop-loss-delta` | `0.003`                              | accuracy 동률일 때 `val_loss_font` 개선 인정 폭 |
| `--best-checkpoint-name`  | `best.pt`                            | 최고 validation 지점 저장 파일명                |

`--split-manifest`가 없으면 기존처럼 전체 데이터셋으로 학습하되, 콘솔에 validation이 비활성화되었다는 경고를 출력한다.

## 4. Dataset 구성

같은 `dataset_dir`에서 dataset 인스턴스를 두 개 만든다.

```python
train_base = FontGlyphDataset(dataset_dir, augment=True, prescan_workers=...)
val_base = FontGlyphDataset(dataset_dir, augment=False, prescan_workers=...)
```

manifest에서 split별 `(font_id, char_index)`를 읽어 각 base dataset의 flat index로 변환한다.

권장 helper:

```python
def build_flat_index_lookup(dataset: FontGlyphDataset) -> dict[tuple[int, int], int]:
    ...

def indices_from_manifest(dataset: FontGlyphDataset, manifest: dict, split: str) -> list[int]:
    ...
```

`FontGlyphDataset`의 내부 `_valid_index`를 직접 읽는 방식은 빠르지만 private 필드에 의존한다. 구현 시에는 작은 공개 메서드를 추가하는 편이 낫다.

권장 공개 API:

```python
def valid_cells(self) -> list[tuple[int, int]]:
    return list(self._valid_index)
```

이 변경은 `dataset_loader.py`에 작고 명확한 읽기 전용 API만 추가하므로 영향 범위가 좁다.

## 5. DataLoader

학습:

```python
train_dataset = Subset(train_base, train_indices)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda"),
)
```

validation:

```python
val_dataset = Subset(val_base, val_indices)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda"),
)
```

validation에는 augmentation을 적용하지 않는다. 같은 checkpoint를 반복 평가해도 같은 결과가 나와야 early stopping 기준으로 쓸 수 있다.

## 6. Validation 지표

학습 로그와 같은 지표를 validation에서도 계산한다.

- `val_loss`
- `val_loss_jamo`
- `val_loss_font`
- `val_cho_acc`
- `val_jung_acc`
- `val_jong_acc`
- `val_syllable_acc`
- `val_font_acc`
- `val_font_top5_acc`

validation은 `model.eval()`과 `torch.no_grad()`로 실행한다. AMP 설정은 학습과 동일하게 `use_amp`를 따르되 gradient는 계산하지 않는다. validation이 끝나면 다시 `model.train()`으로 복귀한다.

## 7. metrics.jsonl 형식

학습 중 기존 record는 유지한다.

```json
{
  "type": "train",
  "epoch": 2,
  "step": 95800,
  "loss": 0.34,
  "font_acc": 0.877
}
```

epoch validation 후 별도 record를 append한다.

```json
{
  "type": "val",
  "epoch": 2,
  "step": 95800,
  "split_manifest": "data/dataset/splits/cell-holdout-seed-20260708.json",
  "num_samples": 817744,
  "val_loss": 0.36,
  "val_loss_jamo": 0.04,
  "val_loss_font": 0.32,
  "val_syllable_acc": 0.990,
  "val_font_acc": 0.874,
  "val_font_top5_acc": 0.994
}
```

기존 `train-monitor.py`가 `type` 필드를 모르는 상태에서도 깨지지 않도록, train record에도 `type: "train"`을 추가하되 기존 metric key는 유지한다. 모니터 업데이트는 후속 작업으로 분리할 수 있다.

## 8. Early Stopping 기준

주 지표는 `val_font_acc`다. `val_font_top5_acc`와 `val_syllable_acc`는 비교적 빠르게 포화되므로 단독 정지 기준으로 쓰지 않는다.

개선 판정:

```text
val_font_acc >= best_val_font_acc + early_stop_min_delta
```

또는 accuracy가 거의 같고 font loss가 의미 있게 줄었을 때:

```text
abs(val_font_acc - best_val_font_acc) < early_stop_min_delta
and val_loss_font <= best_val_loss_font - early_stop_loss_delta
```

정지 조건:

```text
epoch + 1 >= early_stop_min_epochs
and validation이 early_stop_patience회 연속 개선 없음
```

초기 권장값:

```text
early_stop_min_epochs = 3
early_stop_patience = 1
early_stop_min_delta = 0.0015
early_stop_loss_delta = 0.003
```

이 값은 기존 `metrics.jsonl`의 5,000-step rolling 분석에서 나온 보수적 기준이다.
서버에서 validation 로그가 쌓이면 다시 조정한다.

## 9. Checkpoint 정책

기존 정책:

- `checkpoint-epoch-XXXX.pt`
- `latest.pt`
- `interrupted.pt`

추가 정책:

- validation 기준 최고 모델을 `best.pt`로 저장한다.
- `best.pt`에는 기존 checkpoint 필드에 더해 `best_metric` 정보를 넣는다.

예시:

```json
"best_metric": {
  "name": "val_font_acc",
  "value": 0.874,
  "epoch": 2,
  "global_step": 95800,
  "val_loss_font": 0.32,
  "split_manifest": "data/dataset/splits/cell-holdout-seed.json"
}
```

`latest.pt`는 항상 마지막 학습 상태를 가리키고, `best.pt`는 validation 기준 최고 상태를 가리킨다. 조기 정지로 종료하더라도 두 파일의 의미를 섞지 않는다.

## 10. Resume 동작

`--resume`은 optimizer/scheduler/global_step을 복원하는 현재 동작을 유지한다.

추가로 확인할 것:

- resume 시 CLI의 `--split-manifest`가 checkpoint args의 split manifest와 같은지 경고한다.
- 다르면 학습을 막을지 경고만 할지 결정해야 한다. 권장은 오류로 중단하는 것이다.
- `best_metric`이 checkpoint에 있으면 복원해 early stopping 상태를 이어간다.
- 없으면 resume 이후 첫 validation부터 새 best를 잡는다.

## 11. 구현 순서

1. `scripts/generate_split.py`를 구현하고 manifest를 생성한다(완료).
2. `FontGlyphDataset.valid_cells()` 읽기 전용 API를 추가한다.
3. manifest 로더와 split index 변환 helper를 `train-model.py`에 추가한다.
4. `--split-manifest`가 있을 때 train/val `Subset`을 구성한다.
5. validation loop를 함수로 분리한다.
6. validation record를 `metrics.jsonl`에 append한다.
7. `best.pt` 저장 정책을 추가한다.
8. `--early-stop` 옵션과 정지 판정을 추가한다.
9. resume 시 split manifest 일치 검사를 추가한다.
10. 문서와 실행 예시를 업데이트한다.

## 12. 남겨둘 결정 사항

- manifest 파일 크기가 너무 크면 JSON 명시 목록을 유지할지, gzip 또는 compact JSON로
  바꿀지 결정한다. 첫 구현은 plain JSON을 우선한다.
- `train-monitor.py`가 validation record를 별도 곡선으로 그리도록 바꾸는 작업은 별도 문서/티켓으로 분리한다.
