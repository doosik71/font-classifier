# Construct Dataset — 사용자 설명서 및 상세설계서

`scripts/construct-dataset.py`는 `data/annotation`에 저장된 annotation
정보를 이용해 `data/scan`의 원본 스캔 영상에서 완성형 한글 2,350자 낱글자
영상을 폰트 단위로 추출·정규화하여, 실제로 학습에 쓸 수 있는 파일로
저장하는 배치 도구이다.

`font-dataset-browser.py`가 annotation과 추출 로직이 올바른지 화면으로
**검증**만 하고 아무것도 저장하지 않는 도구였다면, 이 도구는 그 검증을
통과한 같은 추출 로직을 이용해 실제로 **파일 데이터셋을 만드는**
단계다(README "진행 상황" 참고). GUI 없이 콘솔에서 끝까지 실행되는
일괄 처리 스크립트다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용자 설명서** — 도구를 실행하고 결과와 출력을 읽는 방법
- **2부. 상세설계서** — 파일 포맷 설계, 실패 판정, 재실행 시 동작 등
  유지보수에 필요한 설계 정보

---

## 1부. 사용자 설명서

### 1.1 이 도구가 하는 일

1. `data/annotation`의 모든 JSON을 읽어 폰트 단위로 묶고, 폰트 이름을
   가나다(알파벳)순으로 정렬한다(`font_classifier.font_dataset.
   build_font_entries()` 재사용 — annotation이 100자 이상 통째로 누락된
   폰트는 여기서 이미 제외된다. 자세한 내용은
   [font-dataset-browser.md](font-dataset-browser.md) 2.1~2.2절 참고).
2. 정렬된 순서대로 폰트를 하나씩 처리한다. 폰트마다 완성형 2,350자를
   `data/scan`의 zip/jpg에서 잘라내 64x64로 정규화한 뒤, 인쇄 순서대로
   세로로 이어붙여 **폰트당 PNG 파일 한 장**(64 x 150,400px)을 만든다.
3. 글자 추출에 5회 이상 실패한 폰트는 품질이 낮다고 보고 통째로
   건너뛴다(1.4절, 2.3절).
4. 성공한 폰트만 `data/dataset/0001.png`부터 순서대로 번호를 매겨
   저장하고, 같은 폴더의 `index.json`에 번호·폰트 이름·파일 이름 매핑을
   기록한다.

폰트 하나당 2,350개의 개별 파일을 만들면 3천여 종의 폰트에서 6백만 건이
넘는 파일이 생겨 저장소와 메모리 처리에 부담이 크므로, 폰트 단위로 한
파일에 모든 글자를 모아 저장한다(2.1절).

### 1.2 실행 방법

```bash
bin\construct-dataset.bat
```

또는 직접:

```bash
uv run python scripts/construct-dataset.py
```

`bin\construct-dataset.bat`는 다른 가상환경이 활성화되어 있어도 이를
무시하고 항상 프로젝트의 `uv` `.venv`만 사용하도록 `VIRTUAL_ENV`를 비운
뒤 실행한다(다른 `bin\*.bat`와 동일한 방식). 별도의 명령줄 인자는 없다.

### 1.3 실행 흐름과 콘솔 출력

```text
Found 3541 font(s) with annotation.
Processing '!백묵-바람과세월체B(종)' (1/3541)...
Processing '!백묵-바람과세월체B(횡)' (2/3541)...
...
[WARNING] 0016-!백묵-스트체(종서): '똠' (idx=101) looks like a blank cell.
...
[SKIP] Font 'YDOL05N': 5 char(s) failed to extract - skipping this font.
...
Done: 3312 font(s) written to D:\dev\font-classifier\data\dataset
```

- 시작하면 처리 대상 폰트 수를 출력하고, 폰트마다 `Processing
  '<이름>' (진행/전체)...`를 출력한다.
- 글자 추출에 실패할 때마다(annotation 없음/빈 칸/영상 로드 실패)
  `[WARNING]`을 출력한다(2.3절 — 어떤 글자인지 알 수 있도록 문자와
  인덱스를 함께 보여준다).
- 한 폰트에서 실패가 5회에 이르면 `[SKIP]`을 출력하고 그 폰트는
  건너뛴다. 남은 글자는 시도조차 하지 않으므로 실패가 누적된 폰트는
  빠르게 다음 폰트로 넘어간다.
- 모든 폰트를 처리한 뒤 `Done: N font(s) written to ...`으로 실제 저장된
  폰트 수를 알려준다.
- 콘솔 메시지는 영어로 출력한다(다른 도구와 동일한 관례).

### 1.4 결과물 구조

```text
data/dataset/
  0001.png    # 64 x 150,400, 그레이스케일(L 모드) — 폰트 한 종
  0002.png
  ...
  index.json
```

`index.json`은 성공적으로 저장된 폰트만 등록된 배열이다.

```json
[
  { "id": 1, "font_name": "!백묵-바람과세월체B(종)", "file": "0001.png" },
  { "id": 2, "font_name": "!백묵-바람과세월체B(횡)", "file": "0002.png" }
]
```

- `id`와 파일 이름의 번호는 항상 같다(`id=1` → `0001.png`).
- 번호는 **건너뛴 폰트를 빼고** 저장에 성공한 순서대로 1부터 다시
  매긴다 — annotation 폰트 목록 안에서의 원래 순번이 아니다.
- 하나의 PNG 안에서 2,350자는 `font_classifier.font_dataset.
  HANGUL_TABLE` 순서(완성형 한글 코드값 순)대로 `idx`번째 글자가
  `[idx*64 : (idx+1)*64]` 행 구간에 놓인다. 즉 N번째(0-based) 글자를
  꺼내려면 세로로 `64*N`px 지점부터 64px를 자르면 된다.
- 추출에 실패한 낱칸(annotation 없음, 빈 칸, 영상 로드 실패)은 다른
  칸과 구분 없이 흰색(255) 그대로 남는다 — PNG 자체에는 "이 칸은
  실패했다"는 표시가 없으므로, 필요하면 콘솔 로그(1.3절)나 원본
  annotation의 `char_pages` 커버리지로 역추적해야 한다.

### 1.5 실행 전 확인할 점 / 재실행 시 동작

- **읽기 전용 입력**: `data/annotation`, `data/scan`은 읽기만 하며 전혀
  수정하지 않는다.
- **재실행은 항상 처음부터 다시 만든다**: 이어하기(resume) 기능은 없다.
  다시 실행하면 폰트 목록을 처음부터 다시 정렬하고, 1번부터 번호를 다시
  매겨 `data/dataset`의 같은 이름 파일을 덮어쓰며 `index.json`도 새로
  쓴다. annotation이 추가/변경되어 폰트 목록 순서나 성공/실패 결과가
  달라지면 번호-폰트 대응도 바뀔 수 있다는 뜻이다.
- **이전 실행보다 파일 수가 줄면 뒷번호 파일이 남을 수 있다**: 이
  도구는 실행 전에 `data/dataset`을 비우지 않는다. 예를 들어 이전
  실행에서 `0001.png`~`0100.png`가 만들어졌는데, 이번 실행에서는 98개
  폰트만 성공했다면 `0099.png`/`0100.png`는 이전 실행의 내용을 그대로
  간직한 채 `index.json`에는 등록되지 않은 고아 파일로 남는다.
  annotation을 갱신한 뒤 다시 만들 때는 `data/dataset`을 먼저 비우고
  실행하는 것을 권장한다(2.6절 "알려진 제한사항"도 참고).
- **소요 시간**: 폰트 3,541종 기준 실측 결과(1분간 테스트)로 미루어
  전체 실행에 대략 1~2시간 이상 걸릴 수 있다(폰트 하나당 최대 5장의
  원본 영상을 열고 2,350칸을 자르고 리사이즈하기 때문). 콘솔을 켜 둔
  채로 충분한 시간을 두고 실행하거나 백그라운드로 돌리는 것을 권장한다.
  `Ctrl+C`로 중단해도 그 시점까지 저장된 파일과 `index.json`은 남아
  있지만(2.5절), 위 재실행 유의사항이 그대로 적용된다.

---

## 2부. 상세설계서

### 2.1 배경 및 설계 목표

3천 종이 넘는 폰트마다 완성형 2,350자를 각각 파일 하나로 저장하면
6백만 건이 넘는 파일이 생긴다. 이는 (1) 파일 시스템에 개별 파일이 너무
많아지고, (2) 학습 시 수백만 개의 작은 파일을 여닫는 I/O 부담이 크며,
(3) 파일 하나당 디스크 블록 오버헤드로 저장 공간도 낭비된다. 이를
피하기 위해 **폰트 하나 = PNG 파일 하나**로 묶고, 2,350자를 세로로
이어붙여 하나의 큰 이미지로 저장한다. 학습 시에는 `index.json`으로
폰트를 찾고, 정해진 오프셋(`idx*64`)으로 필요한 글자만 슬라이싱해 읽으면
된다.

### 2.2 폰트 목록과 처리 순서

`font_classifier.font_dataset.build_font_entries()`를 그대로 재사용한다
(font-dataset-browser.py와 완전히 같은 로직 — 자세한 내용은
[font-dataset-browser.md](font-dataset-browser.md) 2.1~2.2절 참고).

- annotation이 100자 이상(`MISSING_PAGE_THRESHOLD`) 통째로 누락된
  폰트는 여기서 이미 제외된다.
- 나머지 폰트는 `entry.font_name` 가나다순으로 정렬되어 반환된다.
  `main()`은 이 순서 그대로 앞에서부터 처리한다(README/요청사항의
  "알파벳순으로 정렬하고 첫번째 폰트부터 처리" 요구를 그대로 만족한다).
- `missing_count`가 1~99인("부분 누락") 폰트도 이 목록에 포함되어
  일단 시도된다 — 다만 실제로는 2.3절의 5회 실패 기준 때문에 대부분
  건너뛰게 된다(missing_count가 5 이상이면 annotation 누락만으로도
  실패 5회에 도달한다).

### 2.3 글자 추출과 실패 판정 (`_build_font_image`)

폰트 하나(2,350자)를 순서대로 순회하며, 글자 하나마다 다음을 시도한다.

1. `entry.char_pages.get(idx)`로 이 글자가 속한 annotation(페이지)을
   찾는다. **없으면 실패**로 센다(annotation 자체가 없는 경우).
2. 페이지가 있으면 해당 페이지의 원본 영상을 zip에서 읽고 저장된
   `rotation_deg`만큼 회전시킨다(`_load_rotated_page_image` —
   font-dataset-browser.py의 동일 함수와 같은 로직). 같은 페이지의
   글자를 반복해서 다시 열지 않도록 폰트 하나를 처리하는 동안
   `page_image_cache`에 캐시한다. **영상을 열지 못하면 실패**로 센다
   (zip 손상, entry 없음 등).
3. 영상을 열었으면 `font_classifier.char_extract.extract_char_cell`로
   그 칸을 64x64 그레이스케일로 정규화한다(Otsu 이진화로 잉크
   바운딩박스를 찾고, 종횡비를 유지한 채 리사이즈 — 알고리즘 세부사항은
   [font-dataset-browser.md](font-dataset-browser.md) 2.3절 참고, 이
   도구는 그 로직을 그대로 재사용할 뿐 다시 구현하지 않는다).
   **`None`이 반환되면(칸에 잉크가 거의 없음 — 폰트가 그 글자를
   지원하지 않아 빈 칸으로 인쇄된 경우) 실패**로 센다.

세 가지 실패 원인(annotation 없음/영상 로드 실패/빈 칸) 모두 **똑같이
1회의 실패로 취급**한다 — 이 도구의 목적은 annotation 검증이 아니라
학습용 데이터셋의 품질 관리이므로, 원인을 따지지 않고 "이 글자를 쓸 수
없다"는 사실 자체만 센다.

한 폰트에서 실패가 `MAX_FAILURES`(5)에 도달하면 **그 즉시** 처리를
중단하고 폰트 전체를 버린다(`[SKIP]` 로그). 남은 글자는 시도조차 하지
않으므로 annotation이 부실한 폰트일수록 더 빨리 다음 폰트로 넘어간다
— 3,541종 전체를 유한한 시간 안에 처리하기 위한 의도적인 조기 종료
(early-exit) 최적화다.

> **왜 5인가**: 사용자가 명시적으로 지정한 값이다. 완성형 2,350자 중
> 극히 일부(빈 칸 포함 5자 미만)만 빠진 경우는 학습 데이터로 여전히
> 쓸 만하다고 보고 흰 배경으로 채운 채 포함시키지만, 그 이상이면
> annotation 품질이나 폰트 커버리지에 문제가 있다고 보고 아예
> 제외한다. `font-dataset-browser.py`의 `MISSING_PAGE_THRESHOLD`(100,
> "페이지 통째 누락" 판정)와는 목적이 다른 별개의 기준이다 — 그
> 쪽은 목록에서 아예 뺄지 말지를 annotation 존재 여부만으로
> 판단하고, 이 도구는 실제 학습 데이터 품질을 위해 빈 칸까지
> 포함해서 훨씬 엄격하게(5자) 거른다.

실패한 낱칸은 `canvas`에 아무것도 그리지 않으므로 초기값인 흰색(255)
그대로 남는다(1.4절).

### 2.4 이미지 레이아웃

```python
canvas = Image.new("L", (CHAR_SIZE, CHAR_SIZE * len(HANGUL_TABLE)), color=255)
...
canvas.paste(glyph, (0, idx * CHAR_SIZE))
```

- `CHAR_SIZE`(64)는 `font_classifier.char_extract.CHAR_SIZE`를 그대로
  가져와 쓴다 — 낱글자 크기가 두 도구 사이에서 어긋나지 않도록 상수를
  공유한다.
- 가로 64px, 세로 `64 * 2350 = 150,400px`, 그레이스케일(`"L"` 모드)
  단일 캔버스이며, `idx`번째 글자(HANGUL_TABLE 순서)는 세로로
  `[idx*64, (idx+1)*64)` 구간에 놓인다.
- 폭이 64px 고정인 이유는 `extract_char_cell`이 이미 긴 변을 64px로
  맞춰 반환하기 때문이며(`char_extract.py` 참고), 별도의 리사이즈를 더
  하지 않고 그대로 붙여넣는다.

### 2.5 파일 저장과 `index.json` (`main`)

- 폰트가 성공(실패 5회 미만)하면 그 시점의 `next_id`(1부터 시작하는
  카운터)로 파일 이름(`f"{next_id:04d}.png"`)을 만들어 `data/dataset`에
  저장하고, `index.json`에 `{id, font_name, file}` 레코드를 추가한 뒤
  `next_id`를 늘린다. 실패한 폰트는 `next_id`를 소비하지 않는다 —
  번호가 annotation 폰트 목록의 순번이 아니라 **저장에 성공한 순서**를
  나타낸다는 뜻이다.
- 파일 이름은 4자리 0채움이 기본이지만, 폰트 수가 9,999를 넘어가면
  Python의 `:04d` 서식이 자릿수를 그대로 늘려(`10000.png`) 잘리지
  않는다 — 현재 3,541종 규모에서는 해당하지 않는다.
- `index.json`은 폰트 하나를 저장할 때마다(`_write_index`) 그 시점까지
  누적된 전체 목록으로 **매번 다시 씀**(append가 아니라 전체
  덮어쓰기). 파일 하나 분량(수백 KB 이내)이라 매번 다시 써도 비용이
  작으며, 그 대신 중간에 스크립트가 죽거나 `Ctrl+C`로 중단되어도
  그 직전까지 저장된 파일과 `index.json`이 서로 어긋나지 않는다(=
  `index.json`에 없는데 이미 디스크에 있는 PNG는 있을 수 있어도,
  `index.json`에 있는데 PNG가 없는 경우는 없다 — 아래 2.6절의 예외
  상황만 빼면).

### 2.6 알려진 제한사항

- **이어하기(resume) 미지원**: 재실행하면 항상 처음부터 다시 만든다
  (1.5절). annotation이 자주 갱신되는 지금 단계에서는 매번 전체를 다시
  만드는 것이 결과의 일관성 면에서 더 안전하다고 보고 단순하게
  구현했다 — 이어하기를 지원하려면 폰트 이름 기준으로 기존
  `index.json`을 읽어 이미 처리된 폰트를 건너뛰는 로직이 필요하지만,
  이번 범위에서는 다루지 않는다.
- **이전 실행의 고아 파일**: `data/dataset`을 실행 전에 비우지 않으므로,
  이전 실행보다 이번 실행에서 성공한 폰트 수가 적으면 뒷번호의 이전
  파일이 `index.json`에 등록되지 않은 채 디스크에 남을 수 있다(1.5절).
  annotation을 크게 갱신한 뒤에는 `data/dataset`을 먼저 비우고
  실행하는 것을 권장한다.
- **중단 시점에 따라 파일 하나가 `index.json`보다 앞서갈 수 있다**:
  폰트 하나를 처리하는 절차는 `image.save()` → `index.append()` →
  `_write_index()` 순서다. 이 사이(파일은 이미 저장했지만 `index.json`
  갱신 전)에 강제 종료되면, 디스크에는 있지만 `index.json`에는 없는
  PNG 파일 하나가 생길 수 있다(실제로 개발 중 `timeout`으로 강제
  종료해 재현했다). 정상 종료(`Done: ...` 메시지)까지 기다리거나,
  `Ctrl+C`처럼 프로세스가 각 반복 사이에서 스스로 멈추는 방식이라면
  거의 발생하지 않는다.
- **폰트 그룹핑/추출 로직 자체의 한계**는 이 도구가 재사용하는
  `font_classifier` 모듈의 한계를 그대로 물려받는다 — 자세한 내용은
  [font-dataset-browser.md](font-dataset-browser.md) 2.7절 참고(폰트
  이름 문자열이 다르면 다른 폰트로 취급, `grid`/`rotation_deg`를
  검증 없이 신뢰 등).

### 2.7 모듈 구조 요약

| 구성 요소                                           | 파일                                  | 역할                                                                                                          |
| --------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `HANGUL_TABLE` / `FontEntry` / `build_font_entries` | `font_classifier/font_dataset.py`     | 폰트 목록/정렬, 글자→페이지 매핑 (2.2절, [font-dataset-browser.md](font-dataset-browser.md) 2.1~2.2절)        |
| `CHAR_SIZE` / `extract_char_cell`                   | `font_classifier/char_extract.py`     | 칸 하나에서 글자 하나를 64x64로 추출/정규화 (2.3절, [font-dataset-browser.md](font-dataset-browser.md) 2.3절) |
| `GridParams`                                        | `font_classifier/grid_autocorrect.py` | annotation에 저장된 격자 좌표를 담는 자료구조                                                                 |
| `_get_zip` / `_load_rotated_page_image`             | `scripts/construct-dataset.py`        | zip에서 원본 영상을 읽고 회전 보정 적용 (2.3절)                                                               |
| `_build_font_image`                                 | `scripts/construct-dataset.py`        | 폰트 하나의 2,350자 이미지 생성 + 실패 판정 (2.3, 2.4절)                                                      |
| `_write_index` / `main`                             | `scripts/construct-dataset.py`        | 전체 폰트 순회, 파일/인덱스 저장, 진행 로그 (2.5절)                                                           |

`font_classifier/` 패키지의 세 모듈은 `scan-font-browser.py`,
`auto-correct-annotation.py`, `font-dataset-browser.py`와 이미 공유하는
로직이며, 이 도구는 새 알고리즘을 추가하지 않고 검증이 끝난 같은 로직을
"파일로 저장하는 마지막 단계"에 재사용한다.
