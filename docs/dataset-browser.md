# Dataset Browser — 사용자 설명서 및 상세설계서

`scripts/dataset-browser.py`는 `construct-dataset.py`가 만들어 둔
`data/dataset`(폰트별 디렉터리 안에 2,350개의 64x64 PNG가 들어 있는 구조)을
폰트 단위로 열람하는 Tkinter GUI 도구다.

`font-dataset-browser.py`가 `data/annotation` + `data/scan`에서 매번 새로
격자를 찾고 글자를 잘라 정규화해 보여주는 도구였다면, 이 도구는
`construct-dataset.py`가 이미 완성해 둔 결과물 `data/dataset/<font_id>/<hangul_id>.png`를
그대로 읽어 보여준다. `data/dataset`을 읽기만 하며 아무것도 쓰지 않는다 —
실제 학습에 쓰일 최종 데이터셋 파일이 올바른지 눈으로 훑어보기 위한 용도다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용자 설명서** — 도구를 실행하고 화면을 읽는 방법
- **2부. 상세설계서** — `font-dataset-browser.py`와의 차이, 글자 추출/빈
  칸 판정 방식 등 유지보수에 필요한 설계 정보

---

## 1부. 사용자 설명서

### 1.1 실행 방법

```bash
bin\dataset-browser.bat
```

또는 직접:

```bash
uv run python scripts/dataset-browser.py
```

`bin\dataset-browser.bat`는 다른 가상환경이 활성화되어 있어도 이를
무시하고 항상 프로젝트의 `uv` `.venv`만 사용하도록 `VIRTUAL_ENV`를 비운
뒤 실행한다(다른 `bin\*.bat`와 동일한 방식). 별도의 명령줄 인자는 없다.

이 도구는 `data/dataset/index.json`과 그 안에 나열된 폰트 디렉터리/글자 PNG가 이미
만들어져 있어야 의미가 있다. 아직 없다면 먼저
[construct-dataset.md](construct-dataset.md)의 도구를 실행해야 한다.

### 1.2 화면 구성

```text
┌───────────────────────┬──────────────────────────────────────────────┐
│ 폰트 목록 (N)         │ 가  각  간  갇  갈  ...  (20열, 스크롤 가능) │
│                       │ [64x64 영상 격자]                            │
│                       │                                              │
│ 범례 설명             │                                              │
└───────────────────────┴──────────────────────────────────────────────┘
                    상태 표시줄 (하단)
```

- **왼쪽 – 폰트 목록**: `data/dataset/index.json`에 등록된 폰트 이름을
  가나다순으로 나열한다. `construct-dataset.py`가 실패 5회 이상인
  폰트는 애초에 `index.json`에 넣지 않으므로(자세한 내용은
  [construct-dataset.md](construct-dataset.md) 2.3절 참고), 이 목록에
  나온 폰트는 이미 최소한의 품질 기준을 통과한 것들이다 —
  `font-dataset-browser.py`처럼 완비/부분누락을 색으로 구분해서
  보여주지 않는다(1.4절 참고).
- **오른쪽 – 글자 격자**: 폰트를 선택하면 해당 폰트 디렉터리의
  `0000.png`~`2349.png`를 원본 페이지와 같은 20열 격자로 2,350자 순서대로
  보여준다. 각 칸은 이미 저장되어 있는 64x64 그레이스케일 영상을 그대로
  보여주고, 그 아래에 이 자리에 있어야 할 글자를 레이블로 표시한다.
  화면이 길어 마우스 휠이나 세로 스크롤바로 스크롤한다.
- **글자 칸 테두리 색**: 정상적으로 추출된 칸은 옅은 회색 테두리만
  표시된다. 이 자리의 글자를 `construct-dataset.py`가 추출하지 못해
  흰 배경(255) 그대로 남겨 둔 칸은 눈에 잘 띄도록 빨간 테두리로
  표시한다.
- **상태 표시줄**: 선택한 폰트 이름과 함께 "빈 칸(추출 실패) N자"를
  보여준다.

### 1.3 사용 순서

1. 왼쪽 목록에서 확인하고 싶은 폰트를 클릭한다.
2. 오른쪽 화면에 2,350자가 격자로 표시될 때까지 기다린다 — 폰트
   디렉터리 안의 PNG 2,350개를 그대로 읽어 붙이므로 추가적인 추출 계산은
   없다(2.1절).
3. 격자 위 각 칸의 영상과 그 아래 예상 글자 레이블을 눈으로 대조해
   `construct-dataset.py`의 결과물이 올바른지 확인한다.
4. 빨간 테두리 칸이 많은 폰트가 있다면, 그 폰트의 annotation 품질을
   `font-dataset-browser.py`나 원본 annotation에서 다시 확인해 본다
   (1.4절 — 이 도구만으로는 실패 원인을 알 수 없다).

### 1.4 콘솔 메시지 / 이 도구가 알려주지 못하는 것

- `index.json`을 읽지 못하면(파일 손상 등) `[ERROR]`를 한 줄 출력하고
  빈 목록으로 시작한다.
- 그 외에는 콘솔에 출력하는 정보가 거의 없다 — `font-dataset-browser.py`
  와 달리 이 도구는 **왜** 특정 칸이 비어 있는지(annotation 없음 /
  폰트가 그 글자를 지원하지 않음 / 영상 로드 실패) 구분할 방법이 없다.
  `construct-dataset.py`가 세 경우를 모두 흰 배경으로 남기기 때문에
  PNG만 봐서는 원인을 알 수 없기 때문이다(2.4절). 원인까지 알고 싶으면
  `font-dataset-browser.py`로 같은 폰트를 열어 연한 빨강(annotation
  없음)과 진한 빨강(빈 칸)을 구분해서 봐야 한다.

### 1.5 데이터셋이 없을 때

`data/dataset/index.json`이 없으면 상태 표시줄에 안내 문구를 띄우고
목록을 비워 둔다. `construct-dataset.py`를 먼저 실행해 데이터셋을 만든
뒤 다시 열면 된다.

---

## 2부. 상세설계서

### 2.1 `font-dataset-browser.py`와의 차이

| 구분              | `font-dataset-browser.py`                                  | `dataset-browser.py`                        |
| ----------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| 입력 데이터       | `data/annotation`(격자/회전 좌표) + `data/scan`(원본 zip/jpg) | `data/dataset`(이미 정규화된 폰트별 디렉터리/글자 PNG) + `index.json` |
| 글자 추출 방식    | 매번 zip에서 페이지를 열고 회전 보정 후 Otsu로 바운딩박스를 찾아 정규화(`extract_char_cell`) | 폰트 디렉터리에서 `0000.png`~`2349.png`를 그대로 읽음 |
| 폰트 목록 판정    | annotation 존재 여부로 완비/부분누락/제외를 판정, 색으로 구분 | `construct-dataset.py`가 이미 필터링한 `index.json`을 그대로 사용, 색 구분 없음 |
| 빈 칸 원인 구분   | 연한 빨강(annotation 없음) / 진한 빨강(빈 칸) 2단계 구분      | 구분 불가 — 흰 배경이면 무조건 빨간 테두리 1단계(2.4절) |
| 폰트 하나 여는 속도 | 최대 5장의 원본 영상 열기 + 2,350회 Otsu/바운딩박스/리사이즈 (약 1초 안팎) | PNG 1장 열기 + 2,350회 crop(그 이하, Otsu/리사이즈 없음) |
| 쓰기 여부         | 읽기 전용                                                      | 읽기 전용                                        |

두 도구 모두 `data/annotation`·`data/scan`·`data/dataset`을 전혀
수정하지 않는다. `dataset-browser.py`는 `font_classifier.char_extract`에서
`CHAR_SIZE` 상수만 가져다 쓸 뿐, `extract_char_cell`이나
`grid_autocorrect`, zip 처리 로직은 전혀 참조하지 않는다 — 정규화 자체는
`construct-dataset.py`가 이미 끝냈기 때문이다.

### 2.2 폰트 목록 로딩 (`load_index`)

`data/dataset/index.json`을 통째로 읽어 `font_name` 기준으로 정렬한다.
`construct-dataset.py`가 이미 알파벳순으로 처리하며 파일을 쓰므로
사실상 이미 정렬되어 있지만, 방어적으로 다시 한번 정렬해 목록이 항상
가나다순으로 보이도록 보장한다(다른 도구가 `index.json`을 재정렬
없이 편집할 가능성에 대비).

`index.json`이 없거나 JSON 파싱에 실패하면 `[ERROR]`를 출력하고 빈
목록을 반환한다 — GUI는 그대로 뜨되 폰트 목록이 비어 있고 상태
표시줄에 안내 문구가 나온다(1.5절).

### 2.3 글자 영상 추출 (`_render_font`)

```python
cell = image.crop((0, idx * CHAR_SIZE, CHAR_SIZE, (idx + 1) * CHAR_SIZE))
```

`construct-dataset.py`가 `HANGUL_TABLE`의 `idx`번째 글자를 세로로
`[idx*64, (idx+1)*64)` 구간에 저장해 두었으므로([construct-dataset.md](construct-dataset.md)
2.4절), 그 구간만큼만 잘라내면 바로 정규화된 64x64 글자 영상이다.
격자 좌표 추정, 회전 보정, Otsu 이진화, 바운딩박스 계산, 리사이즈 등
`font-dataset-browser.py`가 매번 반복하던 계산이 전혀 필요 없다 — 이
모든 계산은 `construct-dataset.py`가 데이터셋을 만들 때 이미 끝냈다.

폰트를 선택할 때마다 그 폰트의 PNG 파일 하나만 `Image.open`으로 열고
(`image.load()`로 즉시 읽어들여 이후 crop이 지연 디코딩에 걸리지 않게
한다), 2,350번 `crop`을 반복한다. 페이지 이미지를 여러 장 열거나
캐시할 필요가 없다 — 폰트 하나 = 파일 하나이기 때문이다.

### 2.4 빈 칸 판정 (`Image.getextrema`)

```python
if cell.getextrema() == (255, 255):
    blank_count += 1
    self._draw_placeholder(x, y)
    continue
```

`construct-dataset.py`는 추출에 실패한 칸(annotation 없음 / 영상 로드
실패 / 폰트가 그 글자를 지원하지 않는 빈 칸)을 모두 흰 배경(255) 그대로
남겨 둔다([construct-dataset.md](construct-dataset.md) 2.3절). 반대로
정상적으로 추출된 칸은 `extract_char_cell`이 잉크 픽셀을
`MIN_INK_PIXELS`(5) 초과일 때만 글자로 인정하므로, 캔버스에 그려졌다면
반드시 255보다 어두운 픽셀이 하나 이상 있다. 따라서 그레이스케일(`"L"`)
영상의 최소/최대값이 `(255, 255)`(완전히 흰색, 즉 어두운 픽셀이 전혀
없음)인지만 확인하면 "이 칸은 채워지지 않았다"를 정확히 판정할 수
있다.

다만 이 방법으로는 **왜** 비어 있는지 구분할 수 없다 — 세 가지 실패
원인이 모두 같은 흰 배경으로 저장되기 때문이다(1.4절). 원인을 구분해서
보여주는 `font-dataset-browser.py`(연한 빨강/진한 빨강 2단계,
[font-dataset-browser.md](font-dataset-browser.md) 2.4절)와 대비되는
이 도구의 근본적인 한계이며, `data/dataset`만 갖고는 원인을 복원할 수
없다는 점에서 의도된 트레이드오프다(저장 공간을 아끼려고 원인 메타데이터를
따로 저장하지 않았다).

### 2.5 화면 배치와 렌더링

칸 배치 상수(`COLS`, `CELL_PITCH_X`/`CELL_PITCH_Y`, `LABEL_HEIGHT`,
`CELL_MARGIN`)와 격자를 그리는 방식은 `font-dataset-browser.py`의
`_render_font`와 동일하다 — 같은 화면 레이아웃으로 두 도구의 결과를
나란히 비교하기 쉽도록 일부러 맞췄다(모듈로 공유하기엔 GUI 상수라
`font_classifier/`에 두기 애매해, 다른 GUI 스크립트와 마찬가지로 각
스크립트에 그대로 둔다). `ImageTk.PhotoImage` 객체를 `self.tk_images`
리스트에 보관해 가비지 컬렉션을 막는 것도 동일하다.

### 2.6 모듈 구조 요약

| 구성 요소                                    | 파일                                 | 역할                                                        |
| ----------------------------------------------- | --------------------------------------- | --------------------------------------------------------------- |
| `HANGUL_TABLE`                                  | `font_classifier/font_dataset.py`       | 완성형 2,350자 순서 (idx ↔ 글자 매핑)                        |
| `CHAR_SIZE`                                     | `font_classifier/char_extract.py`       | 낱글자 한 변의 픽셀 크기(64) — 오프셋 계산에 사용             |
| `load_index()`                                  | `scripts/dataset-browser.py`            | `index.json` 로딩/정렬 (2.2절)                                |
| `DatasetBrowser._render_font()`                 | `scripts/dataset-browser.py`            | PNG crop, 빈 칸 판정, 격자 렌더링 (2.3~2.5절)                 |

### 2.7 알려진 제한사항

- **실패 원인 구분 불가**: 2.4절에서 설명한 대로, 이 도구는 빈 칸의
  원인(annotation 없음 / 빈 칸 / 로드 실패)을 구분하지 못한다.
- **`index.json`은 실행 시점의 스냅샷**: 실행 중 `construct-dataset.py`
  가 `data/dataset`을 갱신하고 있어도 이 도구는 그 변화를 자동으로
  반영하지 않는다. 최신 상태를 보려면 도구를 껐다가 다시 실행해야
  한다.
- **PNG 파일이 예상 크기와 다르면 조용히 깨질 수 있다**: PNG의 세로
  길이가 `CHAR_SIZE * len(HANGUL_TABLE)`(150,400px)과 다르면(예:
  손상되었거나 다른 도구가 다른 규격으로 만든 파일), 뒤쪽 글자의
  `crop`이 이미지 범위를 벗어나 빈 영상을 반환할 수 있다 — 파일 크기를
  검증하거나 오류를 알려주지 않으므로, 이상하게 비어 보이는 폰트가
  있다면 파일 자체가 정상인지(`Image.open(...).size`) 먼저 확인한다.
