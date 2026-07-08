# Generate Split

`scripts/generate_split.py`는 `data/dataset`의 실제 유효 glyph 셀을 기준으로
재현 가능한 per-font cell-holdout split manifest를 생성하는 도구다.

## 1. 목표

- 모든 폰트 클래스는 train/val/test에 모두 등장하게 유지한다.
- 각 폰트 안에서 실제 존재하는 `char_index`만 85/10/5로 나눈다.
- split은 seed와 데이터셋 내용이 같으면 항상 같은 결과를 만든다.
- 결과는 이후 `train-model.py`, `eval-model.py`가 직접 읽을 수 있는 JSON
  manifest로 저장한다.
- split 생성 시 데이터 분포 리포트를 같이 저장해 치우침과 빈 split을 확인한다.

## 2. 기본 사용법

```bash
uv run python scripts/generate_split.py
```

권장 CLI:

```bash
uv run python scripts/generate_split.py ^
  --dataset-dir data/dataset ^
  --output data/split/cell-holdout-seed.json ^
  --seed 123 ^
  --train-ratio 0.85 ^
  --val-ratio 0.10 ^
  --test-ratio 0.05
```

필수 기본값:

| 옵션                  | 기본값                               | 설명                                               |
| --------------------- | ------------------------------------ | -------------------------------------------------- |
| `--dataset-dir`       | `data/dataset`                       | `index.json`과 폰트별 PNG 디렉터리가 있는 폴더     |
| `--output`            | `data/splits/cell-holdout-seed.json` | split manifest 경로                                |
| `--seed`              | `123`                                | deterministic hash에 섞는 seed                     |
| `--train-ratio`       | `0.85`                               | 폰트별 train 비율                                  |
| `--val-ratio`         | `0.10`                               | 폰트별 validation 비율                             |
| `--test-ratio`        | `0.05`                               | 폰트별 test 비율                                   |
| `--min-val-per-font`  | `1`                                  | 유효 글자가 충분한 폰트에서 보장할 최소 val 셀 수  |
| `--min-test-per-font` | `1`                                  | 유효 글자가 충분한 폰트에서 보장할 최소 test 셀 수 |
| `--overwrite`         | 꺼짐                                 | 기존 manifest가 있으면 덮어쓸지 여부               |

## 3. Split 단위

split 단위는 `(font_id, char_index)` cell이다.

폰트 자체를 validation/test에서 제외하지 않는다. 현재 모델의 font head는 known
font class softmax이므로, 학습하지 않은 폰트를 정답 class로 맞히는 문제는 기본
validation 목적과 맞지 않는다. held-out-font split은 나중에 style embedding
retrieval이나 clustering 품질을 보는 보조 평가로 별도 설계한다.

## 4. 데이터 스캔

입력은 `data/dataset/index.json`과 폰트별 PNG 파일이다.

처리 순서:

1. `index.json`을 읽고 `id`가 `0..N-1` 연속 정수인지 검증한다.
2. 각 entry의 `dir` 아래에서 파일명이 `{char_index:04d}.png`인 파일만 모은다.
3. `char_index`는 `0 <= char_index < len(HANGUL_TABLE)` 범위만 허용한다.
4. 각 폰트의 유효 `char_index` 목록을 오름차순으로 정렬한다.
5. 유효 셀이 0개인 폰트는 오류로 처리한다.

이미 `construct-dataset.py`가 빈 glyph의 PNG를 만들지 않는 구조이므로 파일 존재
여부만으로 유효 셀을 판단한다.

## 5. Deterministic 배정 알고리즘

폰트마다 독립적으로 셀을 나눈다.

권장 방식:

1. 각 cell에 대해 안정적인 hash key를 만든다.
2. hash 값으로 해당 폰트의 char 목록을 정렬한다.
3. 정렬된 목록 앞에서부터 test, val, train 순서로 자른다.
4. manifest에는 각 split의 `char_index` 목록을 오름차순으로 저장한다.

hash 입력:

```text
font-classifier-cell-holdout|<seed>|<font_id>|<char_index>
```

hash 함수:

```text
sha256(key.encode("utf-8"))
```

카운트 계산:

```text
test_count = round(valid_count * test_ratio)
val_count = round(valid_count * val_ratio)
train_count = valid_count - val_count - test_count
```

보정 규칙:

- `valid_count >= 20`인 폰트는 val/test가 각각 최소 1개가 되게 한다.
- 보정 후 `train_count`가 1보다 작아지면 train을 우선한다.
- 최종적으로 train/val/test가 서로 겹치면 오류로 처리한다.
- 세 split의 합집합이 해당 폰트의 유효 char 전체와 다르면 오류로 처리한다.

이 방식은 seed와 데이터셋이 같으면 항상 같은 결과를 만들고, 같은 글자가 모든
폰트에서 동시에 val/test로 빠지는 현상을 줄인다.

## 6. Manifest 형식

manifest는 사람이 읽을 수 있는 JSON으로 저장한다. 대용량이지만 split 재사용성과
디버깅을 우선해 폰트별 char 목록을 명시적으로 기록한다.

```json
{
  "schema_version": 1,
  "kind": "font-classifier-cell-holdout",
  "dataset_dir": "data/dataset",
  "created_by": "scripts/generate_split.py",
  "seed": 123,
  "ratios": {
    "train": 0.85,
    "val": 0.1,
    "test": 0.05
  },
  "hash": {
    "algorithm": "sha256",
    "key_format": "font-classifier-cell-holdout|<seed>|<font_id>|<char_index>"
  },
  "dataset_fingerprint": {
    "index_sha256": "<sha256 of index.json>",
    "num_fonts": 3480,
    "num_valid_cells": 8177436
  },
  "totals": {
    "train": 6950811,
    "val": 817744,
    "test": 408881
  },
  "fonts": [
    {
      "font_id": 0,
      "dir": "0000-font-name",
      "name": "font-name",
      "counts": {
        "total": 2350,
        "train": 1997,
        "val": 235,
        "test": 118
      },
      "splits": {
        "train": [0, 1, 2],
        "val": [17, 29],
        "test": [41, 42]
      }
    }
  ],
  "report": {
    "min_cells_per_font": 2100,
    "max_cells_per_font": 2350,
    "empty_split_fonts": [],
    "label_distribution": {
      "train": {
        "cho": [],
        "jung": [],
        "jong": []
      },
      "val": {
        "cho": [],
        "jung": [],
        "jong": []
      },
      "test": {
        "cho": [],
        "jung": [],
        "jong": []
      }
    }
  }
}
```

`name`은 `index.json` entry에 폰트 이름 필드가 있을 때만 채운다. 없으면 생략한다.
예시의 char 목록은 축약 예시이며, 실제 manifest에는 전체 목록을 저장한다.

## 7. 검증 리포트

생성 후 콘솔과 JSON `report`에 다음 정보를 남긴다.

- 전체 font 수
- 전체 유효 cell 수
- split별 cell 수와 비율
- 폰트별 train/val/test 최소, 최대, 평균 cell 수
- train/val/test 중 하나라도 비어 있는 폰트 목록
- cho/jung/jong label 분포
- split 간 중복 cell 수
- manifest 저장 경로

실패 조건:

- `index.json`이 없거나 `id`가 연속 정수가 아니다.
- ratio 합이 1.0에서 너무 멀다. 허용 오차는 `1e-9`.
- 유효 셀이 없는 폰트가 있다.
- split 간 중복 또는 누락이 있다.
- output 파일이 이미 있는데 `--overwrite`가 없다.

## 8. 실행 절차

1. `uv sync`로 의존성을 맞춘다.
2. `data/dataset/index.json`과 폰트별 PNG가 모두 있는지 확인한다.
3. `uv run python scripts/generate_split.py --dataset-dir data/dataset`를 실행한다.
4. 콘솔 리포트에서 split 비율과 empty split이 없는지 확인한다.
5. 생성된 manifest를 `train-model.py`의 `--split-manifest` 입력으로 사용한다.

## 9. 구현 메모

- 새 의존성은 추가하지 않는다. 표준 라이브러리 `argparse`, `hashlib`, `json`,
  `datetime`, `pathlib`, `statistics`만으로 충분하다.
- 데이터셋 스캔 로직은 가능하면 `FontGlyphDataset`의 기준과 동일하게 유지한다.
- manifest를 읽는 학습 코드가 빠르게 flat index를 만들 수 있도록 폰트별
  `char_index` 목록을 저장한다.
- 파일 크기가 커질 수 있으므로 `json.dumps(..., ensure_ascii=False)` 대신
  스트리밍 write를 고려할 수 있다. 첫 구현은 단순한 `json.dump(..., indent=2)`로
  충분하다.
