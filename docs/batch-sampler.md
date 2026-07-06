# Font Group Batch Sampler - 사용 설명서 및 상세설계서

`font_classifier/batch_sampler.py`의 `FontGroupBatchSampler`는
`FontGlyphDataset`(dataset_loader.py)과 짝을 이루는 `DataLoader` 배치
샘플러다. GUI 도구가 아니라 학습 스크립트가 `import`해서 쓰는 라이브러리
모듈이므로 `bin\*.bat` 런처는 없다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용법** - 학습 스크립트에서 이 샘플러를 쓰는 방법
- **2부. 상세설계서** - 왜 필요한지, K/M 기본값 근거, 알고리즘, 트레이드오프

---

## 1부. 사용법

### 1.1 기본 사용

```python
from torch.utils.data import DataLoader
from font_classifier.dataset_loader import FontGlyphDataset
from font_classifier.batch_sampler import FontGroupBatchSampler

train_ds = FontGlyphDataset()
sampler = FontGroupBatchSampler(train_ds)          # 기본값 K=32, M=8, W=4
loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4)

for epoch in range(num_epochs):
    for batch in loader:               # 매 epoch(= for 루프 진입)마다 폰트 묶음이 새로 섞인다
        ...
```

`batch_sampler=`를 넘길 때는 `DataLoader`의 `batch_size`/`shuffle`/`sampler`
인자를 **함께 쓰면 안 된다** - 배치 크기와 순서 전체를
`FontGroupBatchSampler`가 결정한다. `num_workers>0`을 Windows에서 쓰려면
학습 스크립트를 `if __name__ == "__main__":` 블록 안에서 실행해야 한다
([dataset-loader.md](dataset-loader.md) 1.2절과 동일한 PyTorch 요구사항).

### 1.2 K/M/W 조정

```python
sampler = FontGroupBatchSampler(
    train_ds,
    fonts_per_batch=16,
    chars_per_font=4,
    groups_in_flight=2,
)
```

- `fonts_per_batch`(K): 배치 하나에 섞이는 서로 다른 폰트(=분류 클래스)
  수. 크게 잡을수록 폰트 분류 손실의 배치당 클래스 다양성이 좋아지지만,
  캐시에 동시에 붙잡아 둬야 하는 폰트 수도 늘어난다([dataset-loader.md](dataset-loader.md)
  1.5절의 `max_cache_bytes`와 함께 조정한다).
- `chars_per_font`(M): 폰트 하나당 배치에 들어가는 글자 수. 배치 크기는
  `K * M`이다(마지막 폰트 묶음/폰트별 마지막 라운드는 더 작을 수 있다 -
  1.3절).
- `groups_in_flight`(W): 동시에 활성화해 라운드 단위로 섞을 폰트 묶음 수.
  클수록 같은 묶음이 너무 오래 연속으로 나오지 않지만, 동시에 캐시에 남아
  있어야 하는 폰트 수도 `W * K`로 늘어난다. 기본값 `W=4`는 다양성과 캐시
  지역성의 절충안이다.
- 기본값(K=32, M=8, W=4)의 근거는 2.2절 참고.

### 1.3 배치 크기가 항상 K*M은 아니다

폰트 수(K로 나누어떨어지지 않음)나 폰트별 유효 글자 수(M으로
나누어떨어지지 않음, 그리고 폰트마다 빈 칸 개수가 조금씩 다름) 때문에,
마지막 폰트 묶음이나 폰트별 마지막 라운드는 `K*M`보다 작은 배치를 만들
수 있다. 기본값 `drop_last=False`는 이런 배치도 그대로 내보내
(가변적인 배치 크기를 감수하고) 데이터를 하나도 버리지 않는다.
`drop_last=True`로 만들면 정확히 `K*M` 크기인 배치만 내보낸다(그만큼
매 epoch 일부 글자가 누락된다).

### 1.4 `len(sampler)`는 근사치다

`FontGroupBatchSampler.__len__`(진행률 표시줄 등에 쓰인다)은 폰트별 유효
글자 수가 조금씩 다르고 폰트 묶음 구성이 매번 무작위로 바뀌기 때문에
**정확한 배치 수가 아니라 근사치**다. 계산 방법은 2.4절 참고.

---

## 2부. 상세설계서

### 2.1 왜 필요한가 - plain `shuffle=True`는 캐시를 무력화한다

`FontGlyphDataset`의 LRU 캐시([dataset-loader.md](dataset-loader.md) 2.3절)는
같은 폰트가 짧은 시간 안에 반복 조회되어야 값어치가 있다. 그런데 폰트가
약 3,480개나 되는 이 데이터셋에서 완전히 무작위인 `shuffle=True`를 쓰면:

- 배치 하나(예: 64개 표본)를 만들 때, 생일 문제(birthday paradox) 근사로
  거의 항상 64개 **전부 다른 폰트**가 뽑힌다(폰트 수가 배치 크기보다
  훨씬 많으므로 충돌이 드물다).
- 폰트 하나가 평균 ~2,300자를 갖고 전체 표본이 ~700만 개라면, 같은
  폰트를 다시 방문할 때까지의 평균 간격은 약 `7,000,000 / 2,300 ≈
  3,043`개 표본이다. 그 사이에 수천 개의 **다른** 폰트가 캐시를 스쳐
  지나가므로, 캐시 용량이 전체 폰트 수(~3,480개, 약 32GiB)에 근접하지
  않는 한 재방문 전에 반드시 밀려난다.

즉 plain `shuffle=True`에서는 LRU 캐시가 사실상 매번 캐시 미스만
겪는다 - 글자 하나(4KB)를 위해 9MB짜리 PNG 전체를 다시 열고 디코딩하는
셈이라, 캐시가 없는 것과 거의 같아진다.

### 2.2 해법 - PK 샘플링 + windowed interleave

해법의 첫 단계는 한 번에 소수의 폰트(K개)만 활성화해 여러 배치에 걸쳐
재사용하는 것이다. 이는 metric learning(얼굴 인식, person re-id 등)에서
흔히 쓰는 **PK 샘플링**과 같은 발상이다.

하지만 이 프로젝트의 폰트 분류는 클래스 수가 매우 많고, softmax 기반 폰트
헤더가 긴 시간 동안 같은 소수 폰트만 positive로 보게 되면 묶음 바깥
클래스를 과도하게 잊기 쉽다. 그래서 현재 구현은 여기에 한 단계를 더해,
**여러 폰트 묶음을 동시에 활성화하고 라운드 단위로 번갈아 내보내는
windowed interleave**를 사용한다.

중요한 점:

- **M은 캐시 적중률에 직접 영향을 주지 않는다.** 폰트가 캐시에 남아 있는
  한, 디코딩 1회는 그 폰트의 유효 글자 수 전체(M과 무관하게 ~2,300자)에
  상각된다.
- **K는 배치당 클래스 다양성을 결정한다.** K가 너무 작으면 폰트 손실이
  소수 클래스에 치우친다.
- **W는 시간축 다양성을 결정한다.** W가 1이면 예전 구현처럼 한 묶음을
  끝까지 연속 소모하고, W가 커질수록 서로 다른 폰트 묶음이 더 자주
  교차된다.

폰트 약 3,480개, 폰트당 최대 2,350자라는 실제 규모에서, 기본값을 다음과
같이 정했다.

| 파라미터                | 기본값 | 근거 |
| ----------------------- | ------ | ---- |
| `fonts_per_batch`(K)    | 32     | 배치마다 서로 다른 폰트 클래스 32개가 섞여 폰트 분류 손실의 배치당 클래스 다양성을 확보한다. |
| `chars_per_font`(M)     | 8      | 배치 크기 `K*M=256`을 만들고, 폰트 하나당 같은 배치 안에 충분한 동일 폰트 쌍을 확보한다. |
| `groups_in_flight`(W)   | 4      | 총 `W*K=128`개 폰트만 작업 집합으로 유지해 캐시 지역성을 보존하면서도, 같은 그룹이 수백 스텝 연속으로 이어지는 문제를 크게 줄인다. |

절대적인 정답이라기보다 이 데이터셋 규모에 맞춘 합리적인 출발점이다.

### 2.3 알고리즘 (`__iter__`)

매 epoch(= `__iter__` 호출)마다:

1. 유효 글자가 하나 이상 있는 폰트 목록(`_font_positions`)을 무작위로
   섞는다.
2. `fonts_per_batch`(K)개씩 연속으로 묶어 폰트 묶음을 만든다(마지막
   묶음은 `drop_last=False`면 K보다 작을 수 있다).
3. 묶음 안의 폰트마다: 그 폰트의 유효 글자 인덱스 목록을 무작위로 섞고
   `chars_per_font`(M)개씩 잘라 "라운드" 목록을 만든다.
4. 앞에서부터 `groups_in_flight`(W)개 묶음을 활성화한다.
5. 활성 묶음 각각에서 현재 라운드 하나씩을 뽑아 **round-robin**으로 배치를
   내보낸다.
6. 어떤 묶음이 끝나면, 아직 시작하지 않은 다음 묶음을 그 자리에 투입한다.
7. 모든 묶음이 소진될 때까지 5~6을 반복한다.

`drop_last=True`면 정확히 `K*M` 크기가 아닌 배치를 건너뛴다.

이 순서 덕분에, 한 폰트 묶음은 여러 배치에 걸쳐 재사용되되 다른 묶음과
중간중간 섞인다. 그래서 캐시 지역성과 폰트 손실의 시간축 다양성을 동시에
노릴 수 있다.

### 2.4 `__len__` 근사 계산

interleave를 해도 총 배치 수 자체는 크게 바뀌지 않는다. 여전히 "묶음 수 x
묶음당 평균 라운드 수"로 근사하면 충분하다.

```python
num_groups = ceil(유효 폰트 수 / K)
avg_chars = 유효 폰트들의 평균 유효 글자 수
rounds_per_group = ceil(avg_chars / M)
근사 배치 수 = num_groups * rounds_per_group
```

W는 **배치 순서**를 바꾸지만, 총 라운드 수 자체는 바꾸지 않으므로 이 근사에
직접 들어가지 않는다.

### 2.5 알려진 제한사항

- **정확히 균등하지 않은 배치 크기**: `drop_last=False`(기본값)에서는
  배치 크기가 `M`부터 `K*M`까지 다양할 수 있다.
- **작업 집합이 넓어질수록 캐시 요구량이 늘어난다**: W를 크게 잡으면 좋은
  섞임을 얻을 수 있지만, 동시에 캐시에 남아 있어야 하는 폰트 수가 `W*K`로
  증가한다. `max_cache_bytes`와 함께 조정해야 한다.
- **같은 글자 인덱스를 폰트 간 공통으로 맞춘 그리드는 아니다**: 이 샘플러는
  여전히 폰트마다 독립적으로 무작위 M개 글자를 뽑는다.
- **재현 가능한 seed 제어 없음**: `__iter__`가 Python 표준 `random`
  모듈의 전역 상태를 그대로 쓴다.
- **`len()`은 근사치다**: 폰트 묶음 구성이 무작위이므로 실제 배치 수가
  근사치와 약간 다를 수 있다.

### 2.6 모듈 구조 요약

| 구성 요소                               | 역할 |
| -------------------------------------- | ---- |
| `DEFAULT_FONTS_PER_BATCH` / `DEFAULT_CHARS_PER_FONT` | K/M 기본값 |
| `DEFAULT_GROUPS_IN_FLIGHT`             | 동시에 interleave할 그룹 수 W 기본값 |
| `FontGroupBatchSampler`                | `Sampler[list[int]]` - `__iter__`와 `__len__` 제공 |
| `FontGlyphDataset.flat_indices_by_font()` | 폰트별 유효 샘플 인덱스 목록 제공 |
