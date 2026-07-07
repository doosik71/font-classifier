# Dataset Loader — 사용 설명서 및 상세설계서

`font_classifier/dataset_loader.py`는 `construct-dataset.py`가 만든
`data/dataset`(폰트별 디렉터리 + `index.json`)을 모델 학습에 바로 쓸 수 있는
PyTorch `Dataset`으로 읽어 들이는 모듈이다. GUI 도구가 아니라 학습
스크립트가 `import`해서 쓰는 라이브러리 모듈이므로, `bin\*.bat` 런처는
없다.

핵심 설계는 두 가지다.

1. **메모리 안에서 즉석으로 적용하는 augmentation**: 디스크에 결과를
   저장하지 않고 `__getitem__` 호출마다 매번 새로 변형을 적용한다.
2. **기하 변환은 패딩 후 크롭**: `char_extract.py`가 글자의 긴 변을 정확히
   64px로 맞추므로, 원본 그대로 이동/회전/확대/기울이기를 적용하면 획이
   캔버스 경계에 걸려 잘려 나갈 수 있다. 여유 있게 패딩한 뒤 하나의 affine
   변환으로 합성 적용하고, 가운데 64x64를 다시 잘라내 잘림을 원천적으로
   막는다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용법** — 학습 스크립트에서 이 모듈을 쓰는 방법
- **2부. 상세설계서** — prescan/augmentation/라벨 계산의 설계 근거와 트레이드오프

## 1부. 사용법

### 1.1 사전 조건

`data/dataset/index.json`과 그 안에 나열된 폰트 디렉터리/글자 PNG가 이미 만들어져 있어야
한다(먼저 [construct-dataset.md](construct-dataset.md)의 도구를 실행).
`index.json`의 `id`는 0부터 시작하는 빈틈없는 연속 정수여야 하며(현재
`construct-dataset.py`가 항상 이렇게 만든다), 각 항목에는 폰트 디렉터리 이름을 담은
`dir` 필드가 있어야 한다. 아니면 생성 시점에 `ValueError`를 낸다(2.3절).

### 1.2 기본 사용

```python
from torch.utils.data import DataLoader
from font_classifier.dataset_loader import FontGlyphDataset

train_ds = FontGlyphDataset()          # augment=True가 기본값
loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)

for batch in loader:
    images = batch["image"]        # (B, 1, 64, 64) float32, [0, 1]
    font_labels = batch["font_label"]   # (B,) long, 0..num_font_classes-1
    cho = batch["cho_label"]       # (B,) long, 0..18
    jung = batch["jung_label"]     # (B,) long, 0..20
    jong = batch["jong_label"]     # (B,) long, 0..27
    ...
```

일반적인 학습에서는 `DataLoader`의 `batch_size`와 `shuffle=True`를 함께
사용하면 된다.

Windows에서 `num_workers>0`을 쓰려면(2.4절) 학습 스크립트를 반드시
`if __name__ == "__main__":` 블록 안에서 실행해야 한다 — 이는 이 모듈이
아니라 PyTorch의 Windows(spawn 기반 multiprocessing) 요구사항이다.

### 1.3 `__getitem__`이 반환하는 필드

| 키                                        | 자료형                                        | 설명                                                                                                                                            |
| ----------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `image`                                   | `torch.float32`, `(1, 64, 64)`, 범위 `[0, 1]` | augmentation이 적용된(또는 `augment=False`면 정규화만 된) 낱글자 영상                                                                           |
| `font_label`                              | `torch.long`, 스칼라                          | `index.json`의 `id` (이미 0-based이므로 분류기 클래스 인덱스로 바로 사용)                                                                       |
| `cho_label` / `jung_label` / `jong_label` | `torch.long`, 스칼라                          | 초성(0~18) / 중성(0~20) / 종성(0~27, 받침 없음 포함)                                                                                            |
| `font_id`                                 | `int`                                         | `index.json`의 `id`(0-based) — 데이터셋 안의 폰트 디렉터리 번호와 같다                                                                          |
| `char_index`                              | `int`                                         | `HANGUL_TABLE`에서의 글자 인덱스(0~2349)                                                                                                        |

기본 `collate_fn`(딕셔너리를 재귀적으로 배치하는 PyTorch 기본 동작)과
그대로 호환되므로 커스텀 collate가 필요 없다.

### 1.4 평가/검증 시 augmentation 끄기

```python
eval_ds = FontGlyphDataset(augment=False)
```

`augment=False`면 무작위성이 전혀 없고, 같은 인덱스를 여러 번 읽어도
항상 같은 텐서를 반환한다(정규화 `/255.0`만 적용).

### 1.5 `num_workers`와 Windows 주의사항

`DataLoader(num_workers=N)`는 워커마다 별도 **프로세스**를 띄우므로,
`FontGlyphDataset`도 워커별 사본으로 전달된다(2.4절). Windows에서
`num_workers>0`을 쓸 때는 학습 스크립트를 반드시
`if __name__ == "__main__":` 블록 안에서 실행해야 한다.

### 1.6 Augmentation 강도 조정

```python
from font_classifier.dataset_loader import AugmentConfig, FontGlyphDataset

cfg = AugmentConfig(max_rotate_deg=2.0, noise_prob=0.0, jpeg_prob=0.0)
train_ds = FontGlyphDataset(augment_config=cfg)
```

모든 필드와 기본값은 2.5절 표를 참고. 기본값은
[model-design.md](model-design.md) 5.3절의 "공통(약함)" 세트에 해당한다.

### 1.7 클래스 수 / 상수

```python
train_ds.num_font_classes          # index.json에 등록된 폰트 수
from font_classifier.font_dataset import NUM_CHO, NUM_JUNG, NUM_JONG  # 19, 21, 28
```

분류 헤드 크기를 지정할 때 이 값들을 그대로 쓴다.

## 2부. 상세설계서

### 2.1 유효 글자 판정(빈 칸 제외)

`construct-dataset.py`는 추출 실패(annotation 없음/영상 로드 실패/폰트가
그 글자를 지원하지 않는 빈 칸)를 모두 흰 배경(255) PNG로 저장한다
([construct-dataset.md](construct-dataset.md) 2.3절). 반대로
정상적으로 추출된 칸은 잉크 픽셀이 최소 하나는 있으므로, 이 모듈은
"칸 전체가 흰색인지"만으로 유효성을 판정한다. 폰트 하나(2,350칸)를 읽어
numpy로 한 번에 검사한다.

```python
mins = arr.reshape(NUM_CHARS, CHAR_SIZE * CHAR_SIZE).min(axis=1)
valid = mins < 255
```

세 가지 실패 원인을 구분할 수 없다는 한계도 `dataset-browser.py`와
동일하다 — `data/dataset`만으로는 원인을 복원할 수 없다.

### 2.2 초기화 시 사전 스캔(prescan)과 스레드 병렬화

`Dataset.__len__`이 안정적인 길이를 보고하려면, 어떤 (폰트, 글자) 쌍이
유효한지 생성 시점에 전부 알아야 한다. 그래서 `__init__`은 등록된 모든
폰트의 PNG를 한 번씩 읽어 유효 칸 목록(`_valid_index`)을 만든다
(`_prescan`).

- 폰트 하나(PNG, 약 5~6MB 압축)를 열고 디코딩하는 데 실측 약 90ms가
  걸린다. 폰트가 3천여 종이면 순차 처리 시 약 5분이 걸리므로,
  `ThreadPoolExecutor`로 병렬화한다(`prescan_workers`, 기본 8). PIL의
  PNG 압축 해제(zlib)가 GIL을 일부 해제하므로 스레드로도 유의미한
  속도 향상을 얻는다 — 다만 `ProcessPoolExecutor`만큼의 완전한
  병렬성은 아니다(2.8절 알려진 제한사항).
- 디코딩(`_decode_font`)은 입력 폰트 위치만 보고 결과를 만드는 순수 함수라
  스레드 풀에서 안전하게 병렬 호출할 수 있다. 유효 글자 목록
  (`_valid_index`)을 등록하는 작업(`_register_font`)은 항상 메인
  스레드에서 `pool.map`(입력 순서를 보장)의 결과를 순서대로 소비하며
  수행한다.
- 200개 폰트마다(그리고 완료 시) `[FontGlyphDataset] scanned i/N
  font(s)...`를 콘솔에 출력해 진행 상황을 알려준다.

### 2.3 라벨 계산과 `id` 무결성 검증

- `font_label = entry["id"]`: 새 데이터셋에서는 `id` 자체가 이미 0-based
  분류기 클래스 인덱스다.
- 위 방식이 안전하려면 `id`가 0부터 시작하는 빈틈없는 연속 정수여야
  한다(`num_font_classes = len(entries)`와 라벨 값 범위가 맞아떨어지려면
  필수). `_load_index`가 생성 시점에 이를 검증하고, 어긋나면 바로
  `ValueError`를 낸다. 각 항목에 `dir` 필드가 있는지도 함께 검증한다.
- `cho_label`/`jung_label`/`jong_label`은 `font_classifier.font_dataset.
  decompose_hangul_syllable`(이번 작업에서 함께 추가, [model-design.md](model-design.md)
  3.7절 공식)로 계산한다. 학습 라벨 생성과 이후 추론 디코딩이 항상 같은
  함수를 쓰도록 이 한 곳에만 구현했다.

### 2.4 `DataLoader(num_workers>0)`와 피클링

PyTorch는 Windows에서 `num_workers>0`이면 `spawn` 방식으로 워커
**프로세스**를 새로 띄우고, 그 프로세스에 `Dataset` 인스턴스를 pickle해
전달한다. 이 모듈은 `__getstate__`를 오버라이드해, 워커에서 다시 만들
필요가 없는 보조 자료구조는 직렬화 대상에서 제외한다. 학습에 실제로
필요한 `__getitem__` 동작에는 영향이 없고, 워커 시작 시 불필요한 데이터
복사를 줄이기 위한 정리다.

### 2.5 Augmentation 파이프라인

`_augment`가 아래 순서로 적용한다. 기본값은 모두
[model-design.md](model-design.md) 5.3절 "공통(약함)" 세트 수준이다 —
font 분류 손실 학습에도 안전하게 쓸 수 있는 약한 강도를 목표로 한다.

| 순서 | 항목                                         | 기본 확률/범위                                  | 비고                                      |
| ---- | -------------------------------------------- | ----------------------------------------------- | ----------------------------------------- |
| 1    | 기하 변환(이동+회전+확대+기울이기, 1회 합성) | 항상 적용, `±4px` / `±5°` / `0.92~1.08` / `±5°` | 2.7절 — 패딩 후 크롭으로 잘림 방지        |
| 2    | 밝기                                         | 항상, `0.85~1.15`                               |                                           |
| 3    | 대비                                         | 항상, `0.85~1.15`                               |                                           |
| 4    | 감마                                         | 항상, `0.85~1.15`                               | `image ** gamma`                          |
| 5    | 흐림 또는 샤픈(둘 중 최대 하나)              | 흐림 15%, 샤픈 15%                              | 초점 흐림/과샤프닝 흉내                   |
| 6    | 가우시안 노이즈                              | 50%, `std 0.01~0.05`([0,1] 스케일)              |                                           |
| 7    | JPEG 압축 왕복                               | 30%, `quality 30~80`                            | 실제 JPEG 인코딩→디코딩(`_simulate_jpeg`) |
| 8    | random erasing(흰/검 사각형 마스킹)          | 30%, 패치 1~2개, 한 변 5~20%                    | `_random_erasing`, cutout                 |

순서는 실제 사진 촬영 파이프라인을 대략 흉내낸다 — 기하 왜곡과 조명
조건이 먼저 있고, 초점/노이즈 같은 광학적 열화가 그다음, 마지막에 압축과
(렌즈 앞 가림 같은) 가림이 온다고 가정했다. 각 항목을 왜 이 확률/범위로
잡았는지, 그리고 **의도적으로 포함하지 않은** 항목은 아래를 참고한다.

- **좌우/상하 반전은 절대 쓰지 않는다**: 글자를 반전하면 다른(또는
  존재하지 않는) 모양이 되어 라벨이 무효해진다. 일반적인 이미지 분류
  augmentation 라이브러리의 기본값(`RandomHorizontalFlip` 등)을 그대로
  가져오면 안 되는 대표적인 함정이다.
- **획 두께 jitter(dilation/erosion)는 이번 범위에서 뺐다**:
  [model-design.md](model-design.md) 5.3절이 "font 손실에는 쓰지 말 것"으로
  명시한 content 전용 강한 변형이다. 지금 만드는 로더는 font 분류 손실도
  함께 학습하는 baseline(Phase 1)을 겨냥하므로 기본 파이프라인에 넣지
  않았다. Phase 3에서 별도 view로 도입할 때 이 모듈에 새 옵션을 추가하면
  된다.
- **motion blur/그림자/불균일 조명/원근 왜곡(강함)/elastic warp는
  보류했다**: [model-design.md](model-design.md) 5.3절이 "사진 도메인
  모사"로 분류해 로드맵상 Phase 4(실사진 확보 이후)에 배정한 항목이다.
  지금 추가해도 검증할 실사진 데이터가 없어 강도를 가늠할 근거가 없다.

### 2.7 기하 변환: "패딩 후 크롭"으로 잘림 방지 (`_apply_geometric`, `_geometric_pad`)

`char_extract.py`는 글자의 **긴 변**이 정확히 64px가 되도록 정규화한다
(`docs/font-dataset-browser.md` 2.3절). 즉 세로로 길거나 가로로 넓은
글자는 이미 한쪽 변이 캔버스 경계에 딱 붙어 있어(여백 0), 64x64 캔버스
안에서 그대로 이동/회전/확대/기울이기를 적용하면 획이 밀려나가면서
잘릴 수 있다.

해결책은 torchvision의 `Pad + RandomCrop` 관용구와 같은 원리다.

1. 원본 64x64 이미지를 흰색(255, 정규화 후 `1.0`)으로 `pad`px만큼 사방에
   패딩한다 → `(64+2·pad) x (64+2·pad)`.
2. 이동(`translate`)/회전(`angle`)/확대(`scale`)/기울이기(`shear`)를
   하나의 affine 파라미터로 뽑아 `torchvision.transforms.functional.affine`
   로 패딩된 캔버스에 **한 번만** 적용한다. 여러 기하 변환을 따로따로
   적용하면 리샘플링이 누적되어 화질이 흐려지므로, 하나의 호출로 합성한다.
3. 결과에서 가운데 64x64(`[pad:pad+64, pad:pad+64]`)를 그대로 잘라낸다
   (리샘플링 없는 순수 크롭).

`pad`가 설정된 이동/회전/확대/기울이기의 최댓값이 만들어낼 수 있는 최대
변위보다 크기만 하면, 이 범위 안의 변형은 절대 획을 캔버스 밖으로
밀어내지 않는다. `_geometric_pad`는 이를 넉넉하게(정확한 최솟값이 아니라
여유 있는 상한으로) 계산한다.

```python
half = CHAR_SIZE / 2
angle_margin = half * sin(radians(max_rotate_deg + max_shear_deg))
scale_margin = half * max(0, scale_range[1] - 1)
pad = ceil(max_translate_px + angle_margin + scale_margin) + 2
```

기본값(이동 4px, 회전 5°, 기울이기 5°, 확대 최대 1.08배)에서
`pad=15px`가 나온다. 개발 중 획이 네 변에 모두 닿은 극단적인 테스트
글자로 200회 무작위 augmentation을 반복해, 어떤 경우에도 경계의 획이
완전히 사라지지 않음을 직접 확인했다(이 문서 작성 시점의 검증 방법이며
회귀 테스트로 자동화되어 있지는 않다 — 2.9절).

### 2.8 난수 소스와 멀티프로세스 재현성

파라미터를 뽑는 데 Python 표준 `random` 모듈을, 가우시안 노이즈 텐서
생성에 `torch.randn_like`(torch의 전역 RNG)를 함께 쓴다. 두 RNG 모두
Windows의 `spawn` 기반 워커 프로세스가 완전히 새로운 인터프리터로
시작할 때 OS 엔트로피로 새로 시드되므로, `DataLoader(num_workers>0)`의
워커들이 우연히 같은 augmentation 시퀀스를 반복하는 문제는 (이
프로젝트가 대상으로 하는 Windows 환경에서는) 자연히 피해진다. 리눅스의
`fork` 기반 워커로 옮긴다면 `worker_init_fn`에서 `random.seed`/
`torch.manual_seed`를 프로세스별로 다시 호출해야 한다 — 지금은 다루지
않는다(2.9절).

### 2.9 이번 범위에 포함하지 않은 것 / 알려진 제한사항

- **학습/검증 분할 없음**: 이 로더는 유효한 (폰트, 글자) 쌍 전체를
  하나의 데이터셋으로 노출한다. [model-design.md](model-design.md) 5.1절의
  cell-holdout(폰트마다 10~15% 글자를 테스트로 분리) 전략은 이번
  구현에 포함하지 않았다 — 요청 범위가 dataset loader/augmentation이었고,
  분할은 `torch.utils.data.Subset`이나 별도 wrapper로 이 클래스 위에
  얹기 쉬운 독립적인 문제이기 때문이다. 다음 작업에서 다룬다.
- **대조학습/재구성용 다중 view 미지원**: [model-design.md](model-design.md)
  4.2~4.4절의 모드 B(구조/스타일 소스 재조합), 대조학습용 2-view 반환은
  아직 없다. `__getitem__`이 항상 이미지 하나만 반환하므로, Phase 3
  도입 시 별도 wrapper/Dataset이나 이 클래스의 확장이 필요하다.
- **재현 가능한 seed 제어 없음**: `augment=True`일 때 매 호출이 전역
  RNG를 사용하므로, 완전히 같은 augmentation 시퀀스를 재현하려면
  현재는 별도 장치가 없다.
- **`ProcessPoolExecutor` 미사용**: 사전 스캔(2.2절)을 스레드 풀로만
  병렬화했다. 프로세스 풀을 쓰면 GIL 제약 없이 더 빠를 수 있지만, 이
  모듈이 `DataLoader` 워커 프로세스 안에서 다시 쓰일 가능성을 고려하면
  중첩 멀티프로세싱 복잡도가 커져 1차 구현에서는 보류했다.
- **결측 원인 구분 불가**는 `data/dataset` 자체의 한계이며
  ([construct-dataset.md](construct-dataset.md) 2.3절,
  [dataset-browser.md](dataset-browser.md) 2.4절과 동일), 이 로더로는
  해결할 수 없다.

### 2.10 의존성 변경

이번 작업에서 `pyproject.toml`에 `torch`(2.6.0, CUDA 12.4 빌드)와
`torchvision`(0.21.0)을 추가했다. GPU(RTX 3090, 드라이버 CUDA 13.1
지원)가 있는 개발 환경이라 CUDA 12.4 wheel을 `pytorch-cu124`라는
전용(`explicit = true`) uv 인덱스로 받도록 `[tool.uv.sources]`에
등록했다 — 이 인덱스를 일반 인덱스로 등록하면(즉 `explicit`을 빼면)
`numpy` 등 다른 패키지 버전 해석에도 이 인덱스가 끼어들어 해석
실패를 일으킨다(실제로 처음 시도했을 때 이 문제로 실패했다). `torch`
소스만 이 인덱스를 명시적으로 가리키도록 해 문제를 피했다.
`torchvision`은 별도 인덱스 지정 없이 PyPI 기본 인덱스에서 설치되어
`+cpu` 태그가 붙지만, 이 모듈이 쓰는 `functional.affine` /
`adjust_brightness` / `gaussian_blur` 등은 torchvision 자체의 컴파일된
CUDA 커널이 아니라 torch의 텐서 연산을 그대로 호출하므로 CUDA
텐서에도 문제없이 동작한다(직접 GPU 텐서로 확인).

### 2.11 모듈 구조 요약

| 구성 요소                                                              | 역할                                                                                                                 |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `AugmentConfig`                                                        | augmentation 강도/확률 설정 dataclass (2.5절)                                                                        |
| `_geometric_pad`                                                       | 기하 변환용 패딩 폭 계산 (2.7절)                                                                                     |
| `_load_index`                                                          | `index.json` 로딩 + `id` 연속성 검증 (2.3절)                                                                         |
| `FontGlyphDataset`                                                     | 공개 `Dataset` 클래스 — 사전 스캔, 글자 PNG 로딩, 라벨 계산, `__getstate__` (2.2, 2.3, 2.4절)                        |
| `_augment` / `_apply_geometric` / `_simulate_jpeg` / `_random_erasing` | augmentation 파이프라인 각 단계 (2.5, 2.7절)                                                                         |
| `decompose_hangul_syllable`, `NUM_CHO`/`NUM_JUNG`/`NUM_JONG`           | `font_classifier/font_dataset.py`에 추가한 초/중/종성 분해 유틸리티 (2.3절)                                          |
