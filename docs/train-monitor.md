# Train Monitor — 사용 설명서 및 상세설계서

`scripts/train-monitor.py`는 학습 진행 상황을 실시간으로 보는 도구다.
`data/checkpoints/*/metrics.jsonl` 파일들을 주기적으로 폴링해 변경을
감지하고, 각 run(예: `v1`, `v2`)의 손실·정확도 곡선을 하나의 matplotlib
창에 겹쳐 그린다. 목적은 [train-model-v1.md](train-model-v1.md)(baseline)과
[train-model-v2.md](train-model-v2.md)(Top-$k$ Relaxed Negative Learning)의
학습 곡선을 **나란히 비교**하는 것이다.

이 도구는 **읽기 전용**이다. 학습 프로세스(`train-model-v1.py`/
`train-model-v2.py`)가 `metrics.jsonl`에 append하는 것을 옆에서 읽기만
하며, 체크포인트나 metrics 파일을 건드리지 않는다. 학습과 독립적으로
아무 때나 켜고 끌 수 있다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용법** — 실행 방법, 옵션, 화면 읽는 법
- **2부. 상세설계서** — 폴링/갱신 방식, 패널 구성, 알려진 제한사항

## 1부. 사용법

### 1.1 이 도구가 하는 일 (그리고 하지 않는 일)

- `--checkpoints-dir`(기본 `data/checkpoints`) 아래의 `*/metrics.jsonl`을
  모두 찾아, 하위 폴더 이름을 run 이름으로 삼아 곡선을 그린다.
- 파일이 바뀌면(학습이 새 줄을 append하면) 자동으로 다시 읽고 다시
  그린다. 학습 도중 새 run 폴더가 생기면 다음 폴링에서 자동으로 잡힌다.
- 6개 패널(총손실 / 자소손실 / 폰트손실 / 자소정확도 / 폰트 top-1 /
  폰트 top-5)에 v1·v2를 색으로 구분해 겹쳐 그린다.

**하지 않는 것**: 학습 자체(별도 프로세스), held-out 평가(현재 metrics는
전부 학습 배치 기준 — [train-model-v1.md](train-model-v1.md) 2.2절),
metrics 파일 쓰기/삭제, 곡선 이미지 저장(실시간 창만 띄운다).

### 1.2 사전 조건

- `matplotlib`가 설치되어 있어야 한다. 프로젝트 의존성에 포함되어 있으므로
  `uv sync`로 함께 설치된다(별도로 `pip install` 하지 않는다 — AGENTS.md).
- GUI 창을 띄우므로 **디스플레이가 있는 환경**에서 실행해야 한다(다른
  `bin\*-browser` GUI 도구와 같은 조건). 디스플레이가 없는 원격 셸에서는
  창을 띄울 수 없다(2.7절).
- 아직 `metrics.jsonl`이 하나도 없어도 실행할 수 있다 — 빈 화면으로
  시작해 학습이 시작되면 자동으로 곡선이 나타난다.

### 1.3 실행 방법

```bash
bin\train-monitor.bat
```

또는 직접:

```bash
uv run python scripts/train-monitor.py
```

`bin\train-monitor.bat`는 다른 `bin\*.bat`와 동일하게 `VIRTUAL_ENV`를
비운 뒤 프로젝트의 `uv` `.venv`로 실행한다. 보통 학습을 돌리는 창과
별개의 창에서 이 모니터를 함께 띄워 둔다.

### 1.4 주요 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--checkpoints-dir` | `data/checkpoints` | 이 폴더 아래 `<run>/metrics.jsonl`을 찾는다 |
| `--interval` | 5.0 | 파일 변경을 확인하는 폴링 주기(초) |
| `--x-axis` | `step` | 가로축으로 쓸 값 (`step` 또는 `epoch`) |
| `--runs` | (전체) | 그릴 run 이름 제한 (예: `--runs v1 v2`). 생략하면 발견되는 모든 run |

예시:

```bash
# v2만, epoch을 가로축으로, 2초마다 갱신
uv run python scripts/train-monitor.py --runs v2 --x-axis epoch --interval 2
```

> **참고: glob은 하위 폴더만 본다.** `data/checkpoints/*/metrics.jsonl`
> 패턴이라 `data/checkpoints/v1/…`, `.../v2/…`는 잡지만, 최상위
> `data/checkpoints/metrics.jsonl`(구버전 기본 경로)은 포함하지 않는다.
> v1/v2 스크립트는 각각 `data/checkpoints/v1`, `.../v2`에 쓰므로 정상적으로
> 잡힌다.

### 1.5 화면 읽는 법

- **패널(2 x 3)**: 왼쪽 위부터 순서대로
  1. **Total loss** — `loss`
  2. **Jamo loss (CE)** — `loss_jamo`
  3. **Font loss** — `loss_font`(실선), 그리고 v2에만 있는 `warm`(점선)/
     `trn`(파선). 패널 하단에 "스케일이 달라 절대값 비교 금물"이라는
     주석이 붙는다(2.5절).
  4. **Jamo accuracy** — `syllable_acc`(실선)와 `cho`/`jung`/`jong`
     (점선·일점쇄선 등)
  5. **Font top-1 accuracy** — `font_acc`
  6. **Font top-5 accuracy** — `font_top5_acc`
- **색 = run, 선 스타일 = metric.** 오른쪽 위 범례가 run→색(v1, v2, …)을,
  각 패널 안의 작은 범례가 선 스타일→metric을 알려준다.
- **폰트 정확도 패널의 오른쪽 축(`alpha (curriculum)`)**: v2의 curriculum
  계수 `alpha`를 옅은 선으로 겹쳐 그린다(0~1). warm-up이 끝나고 alpha가
  0→1로 오르는 구간에서 `font_acc`/`font_top5_acc`가 어떻게 반응하는지
  바로 볼 수 있다([train-model-v2.md](train-model-v2.md) 2.4~2.5절). v1은
  alpha가 없으므로 이 선을 그리지 않는다.
- **상단 제목(suptitle)**: 마지막 갱신 시각과 각 run의 최신 step을 보여
  준다. 파일이 아직 없으면 "학습이 시작되면 나타난다"는 안내가 뜬다.
- 모든 곡선은 `metrics.jsonl`에 기록된 값 그대로이며, 각 점은 이미 학습
  스크립트의 `--log-every` 구간 평균이다(추가 평활화는 하지 않는다).
- **주의**: 여기 표시되는 정확도는 전부 **학습 배치 기준**이지 held-out
  지표가 아니다([train-model-v1.md](train-model-v1.md) 2.2절). 최종 성능
  비교가 아니라 학습이 정상적으로 진행되는지 보는 용도다.

### 1.6 종료

창을 닫으면(또는 실행 터미널에서 `Ctrl+C`) 종료된다. 학습 프로세스에는
아무 영향이 없다(별개 프로세스이고 읽기 전용이므로).

## 2부. 상세설계서

### 2.1 왜 watchdog 대신 폴링인가

파일 변경 감지에 `watchdog` 같은 inotify 기반 라이브러리 대신, 각 파일의
`(mtime, size)`를 주기적으로 비교하는 단순 폴링을 쓴다.

- 새 의존성을 추가하지 않는다(AGENTS.md "단순함 최우선"). metrics는 수 초
  단위로 append되므로 5초 폴링이면 충분하고, 실시간성 요구도 낮다.
- matplotlib 이벤트 루프(`FuncAnimation`)가 이미 주기적으로 콜백을 부르는
  구조라, 그 콜백 안에서 파일을 확인하면 별도 스레드/감시자가 필요 없다.

### 2.2 변경 감지와 다시 그리기

`FuncAnimation`이 `--interval`마다 `Monitor.update`를 부른다. 이때
`Monitor.poll()`이:

1. `checkpoints_dir.glob("*/metrics.jsonl")`로 파일 목록을 새로 구한다
   (학습 도중 생긴 새 run 폴더가 자동으로 포함된다).
2. 각 파일의 `(mtime_ns, size)`를 직전 값과 비교해, 바뀐 파일만 다시
   파싱한다.
3. 파일이 사라진 run은 데이터에서 제거한다.
4. 무언가 바뀌었으면 `True`를 돌려주고, 그때만 `redraw()`가 전체 패널을
   다시 그린다(변경이 없으면 다시 그리지 않아 CPU를 아낀다).

`--interval`을 5초로 둔 것은 데이터 양(수천~수만 줄) 대비 재파싱/재드로우
비용이 작아 5초면 부담이 없기 때문이다. 아주 잦은 갱신이 필요하면 값을
줄이면 된다.

### 2.3 부분 기록·깨진 줄 처리

학습 스크립트가 `metrics.jsonl`의 마지막 줄을 쓰는 도중에 모니터가 그
파일을 읽으면 마지막 줄의 JSON이 깨져 있을 수 있다. `load_records`는
줄 단위로 `json.loads`를 시도하고 **실패한 줄은 조용히 건너뛴다** — 다음
폴링에서 그 줄이 완성되면 정상적으로 읽힌다. 그래서 파일을 잠그거나
학습과 동기화할 필요가 없다.

### 2.4 metric별 독립 x/y 매칭

run마다, 그리고 시간에 따라 기록되는 key가 다를 수 있다.

- v1에는 `alpha`/`loss_font_warm`/`loss_font_trn`이 없다.
- `font_top5_acc`는 나중에 추가되었으므로, 오래된 실행의 앞부분 기록에는
  없을 수 있다.

그래서 `series()`는 **metric마다 (x, y)가 모두 존재하는 점만** 골라
독립적으로 배열을 만든다. 없는 key는 그 run/구간에서 그냥 그리지 않으며,
있는 구간부터 곡선이 시작된다. 이렇게 하면 서로 다른 run·서로 다른
버전의 metrics를 한 화면에 문제없이 겹칠 수 있다.

### 2.5 패널 구성과 색/선 규칙

- **색 = run, 선 스타일 = metric.** run 수만큼 색을 matplotlib 기본 색
  순환에서 배정하고(정렬 순서 기준: v1→C0, v2→C1 …), 같은 패널 안에서는
  metric별로 선 스타일을 다르게 준다. run(색) 범례는 그림 상단에, metric
  (선 스타일) 범례는 각 패널 안에 둔다.
- **Font loss 패널의 절대값 비교 금지 주석**: v1은 softmax CE, v2는 sigmoid
  기반 손실이라 `loss_font`의 스케일 자체가 다르다
  ([train-model-v2.md](train-model-v2.md) 2.5절). 오해를 막기 위해 패널에
  주석을 박아 둔다. **비교는 `font_acc`/`font_top5_acc`로 한다.**
- **alpha twin 축**: 폰트 정확도 두 패널에만 오른쪽 y축(0~1)을 두어 v2의
  `alpha`를 겹쳐 그린다. `Axes.clear()`가 twin 축을 매번 기본값(왼쪽)으로
  되돌리므로, 다시 그릴 때마다 오른쪽 배치를 재지정한다(구현 주석 참고).

### 2.6 그림 안 텍스트는 영어

matplotlib 기본 폰트(DejaVu Sans)는 한글 글리프가 없어, 그림 안에 한글을
쓰면 tofu(□)로 깨진다. 한글 폰트를 찾아 지정하는 방식은 환경마다 설치된
폰트가 달라 취약하므로, **그림 안 텍스트(제목·범례·주석·상단 상태줄)는
영어로** 두었다. 콘솔 출력과 이 문서는 한글 그대로다. (한글 라벨이 꼭
필요해지면 프로젝트에 CJK 폰트를 포함하고 `matplotlib.rcParams`에 지정하는
방식으로 확장할 수 있다.)

### 2.7 matplotlib 백엔드와 디스플레이

실시간 창은 대화형 backend(TkAgg 등)와 디스플레이가 필요하다. 이
프로젝트에는 tkinter가 있어 보통 TkAgg가 선택된다. 만약 비대화형
backend(`agg`)로 떨어지면(디스플레이 없음) 창을 띄울 수 없으므로 시작 시
경고를 출력한다. 다른 GUI 도구(`scan-font-browser` 등)가 도는 환경이면
이 모니터도 정상 동작한다.

### 2.8 알려진 제한사항

- **held-out 지표 없음**: 표시되는 정확도는 전부 학습 배치 기준이다
  ([train-model-v1.md](train-model-v1.md) 2.2절). 최종 성능 판단용이
  아니다.
- **매 변경 시 파일 전체 재파싱**: 증분 파싱(append된 부분만 읽기)을 하지
  않고 바뀐 파일을 통째로 다시 읽는다. metrics.jsonl이 수십 MB 수준까지
  커지지 않는 한 문제가 없다고 보았다(단순함 우선). 파일이 매우 커지면
  증분 파싱으로 바꿔야 한다.
- **색 안정성**: run이 추가/삭제되어 run 집합이 바뀌면 색을 다시 배정하므로
  기존 run의 색이 바뀔 수 있다. 정렬 순서 기준이라 같은 run 집합에서는
  일관된다.
- **한글 미표시**(2.6절): 그림 안에는 한글을 쓰지 않는다.
- **곡선의 톱니파 해석 주의**: `FontGroupBatchSampler`가 폰트 그룹을 돌기
  때문에 손실/정확도가 그룹 경계에서 출렁일 수 있다. 이는 모니터가 아니라
  학습 데이터 구성에서 오는 현상이다(batch-sampler.md).

### 2.9 검증 방법

디스플레이가 없는 환경에서도 확인할 수 있도록, matplotlib `Agg` backend로
합성 `metrics.jsonl`(v1 = alpha/warm/trn 없음, 중간부터 `font_top5_acc`
추가·마지막 줄 일부러 깨뜨림 / v2 = 전체 key 포함)을 만들어 다음을
확인했다.

- `load_records`가 깨진 마지막 줄을 건너뛰고 나머지를 정상 파싱하는지.
- `series`가 metric별로 존재하는 구간만 뽑는지(예: `font_top5_acc`는
  중간부터, v1의 `alpha`는 빈 배열).
- `poll`이 두 run을 발견하고, 변경이 없을 때 `False`(재드로우 생략)를
  돌려주는지.
- `redraw`가 전체 패널을 오류 없이 그리고 PNG로 저장되는지, 두 번 다시
  그려도 alpha twin 축이 오른쪽에 유지되는지.
- `--runs` 필터와 `--x-axis epoch` 경로.

**아직 확인하지 못한 것**: 실제 디스플레이에서의 창 렌더링/상호작용,
장시간 실행 시 메모리 추이. 처음 실제로 띄울 때 창이 정상적으로 갱신되는지
직접 관찰해야 한다.

### 2.10 모듈 구조 요약

| 구성 요소 | 역할 |
| --- | --- |
| `parse_args` | CLI 인자 정의 (1.4절) |
| `PANELS` | 패널·metric·선 스타일 구성표 (2.5절) |
| `load_records` | metrics.jsonl 한 파일을 dict 리스트로 (깨진 줄 skip, 2.3절) |
| `series` | records에서 metric별 (x, y) 추출 (2.4절) |
| `Monitor.poll` | 파일 변경 감지 + 데이터 갱신 (2.2절) |
| `Monitor.redraw` | 전체 패널 다시 그리기 (2.5절) |
| `Monitor.update` | `FuncAnimation` 콜백: 변경 시에만 redraw |
| `main` | 인자 파싱, figure 구성, `FuncAnimation` 시작 |
