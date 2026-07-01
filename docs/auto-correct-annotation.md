# Auto Correct Annotation — 사용자 설명서 및 상세설계서

`scripts/auto-correct-annotation.py`는 `data/annotation`에 이미 저장된
annotation(대부분 자동 보정 기능이 만들어지기 전 사람이 직접 입력한 값)에,
`scripts/scan-font-browser.py`가 사용하는 자동 격자/회전 보정 기능
(`grid_autocorrect` 모듈)을 일괄로 다시 적용하는 텍스트 기반(GUI 없음)
배치 도구이다.

이 문서는 두 부분으로 구성된다.

- **1부. 사용자 설명서** — 도구를 실행하고 출력을 읽는 방법
- **2부. 상세설계서** — 비교 전략, 임계값, 파일 처리 방식 등 유지보수에
  필요한 설계 정보

---

## 1부. 사용자 설명서

### 1.1 이 도구가 하는 일

`data/annotation` 폴더의 annotation JSON 파일마다:

1. JSON에 적힌 `zip`/`entry`로 원본 스캔 영상을 찾아 연다.
2. `grid_autocorrect`의 자동 보정(1차 위상 분석 + 2차 바운딩 박스 + 회전
   감지, 원리는 [docs/scan-font-browser.md](scan-font-browser.md) 2.9절
   참고)을 새로 적용해 격자 시작 좌표(origin_x, origin_y)와 회전 보정
   각도(rotation_deg)를 추정한다.
3. 기존에 저장되어 있던 값과 새로 추정한 값을 비교한다.
   - 차이가 작으면 **묻지 않고** 자동 보정 값으로 갱신한다.
   - 차이가 크면 파일 이름/기존 값/자동 보정 값/차이를 화면에 출력하고
     적용 여부를 물어본다.

실행 전에는 항상 `data/annotation` 폴더 전체를 타임스탬프가 붙은 폴더로
백업하므로, 결과가 마음에 들지 않으면 그 폴더로 되돌릴 수 있다.

### 1.2 실행 방법

```bash
bin\auto-correct-annotation.bat
```

또는 직접:

```bash
uv run python scripts/auto-correct-annotation.py
```

`bin\auto-correct-annotation.bat`는 다른 가상환경이 활성화되어 있어도
이를 무시하고 항상 프로젝트의 `uv` `.venv`만 사용하도록 `VIRTUAL_ENV`를
비운 뒤 실행한다 (`scan-font-browser.bat`와 동일한 방식).

이 도구는 `data/scan`의 zip에서 영상을 읽고 `data/annotation`의 파일을
갱신하므로, `scripts/scan-font-browser.py`를 실행할 때와 같은 위치
(프로젝트 루트)에서 실행해야 한다. 별도의 명령줄 인자는 없다.

### 1.3 실행 흐름과 화면 출력

실행하면 다음 순서로 진행된다.

1. `data/annotation` 안의 전체 파일 개수를 출력한다.
2. 전체 폴더를 `data/annotation_backup_YYYYMMDD_HHMMSS/`로 백업하고
   경로를 출력한다.
3. 파일을 하나씩 처리하면서, 50개마다 `진행 i/N ...`을 출력해 진행
   상황을 알려준다.
4. 기존 값과 자동 보정 값의 차이가 **큰** 파일을 만나면 처리를 멈추고
   다음 형식으로 출력한 뒤 입력을 기다린다.

   ```text
   00000002.json  (001.zip / 001/00000002.jpg)
     기존 값   : origin_x=8.00 origin_y=241.00 rotation_deg=0.00
     자동 보정 : origin_x=12.25 origin_y=243.35 rotation_deg=0.00
     차이      : dx=+4.25 dy=+2.35 d회전=+0.00
   자동 보정 값을 적용할까요? [y/N]
   ```

   `y`(또는 `yes`)를 입력하면 그 파일에 자동 보정 값을 적용하고, 그 외
   (빈 입력 포함)에는 건너뛰고 기존 값을 그대로 둔다.
5. 모든 파일을 처리한 뒤 아래와 같은 요약을 출력한다.

   ```text
   === 완료 ===
   자동 적용        : 1802건
   확인 후 적용     : 12건
   확인 후 건너뜀   : 3건
   변화 없음        : 5건
   오류             : 0건
   백업 위치        : data\annotation_backup_20260701_193000
   ```

- **자동 적용**: 차이가 작아 묻지 않고 바로 갱신한 파일 수
- **확인 후 적용**: 차이가 커서 물어봤고, `y`로 답해 적용한 파일 수
- **확인 후 건너뜀**: 물어봤지만 적용하지 않기로 한 파일 수(기존 값 유지)
- **변화 없음**: 자동 보정 값이 기존 값과 정확히 같아 아무것도 하지 않은
  파일 수
- **오류**: JSON을 읽을 수 없거나, zip/entry 정보가 없거나, 원본 영상을
  열 수 없는 등 처리에 실패한 파일 수 (해당 파일은 손대지 않는다)

### 1.4 얼마나 많은 파일이 확인을 요구할까

1,900여 개의 annotation 중 정말 초반에 수동으로 입력한 파일 몇몇은 자동
보정 값과 4~9px 정도 차이가 나는 경우가 있었다(수동 보정 방식이
안정되기 전에 입력된 것으로 보인다). 대부분은 자동 보정 값과 매우
가깝게 일치한다. 현재 임계값(1.5절 참고)에서는 이런 파일 대부분이
"자동 적용"으로 처리되고, 소수만 확인을 거친다.

### 1.5 허용 오차(임계값) 조정

`scripts/auto-correct-annotation.py` 상단의 두 상수로 "차이가 크다"의
기준을 정한다.

```python
ORIGIN_DIFF_THRESHOLD_PX = 20.0
ROTATION_DIFF_THRESHOLD_DEG = 1.5
```

- `origin_x` 또는 `origin_y`의 차이가 `ORIGIN_DIFF_THRESHOLD_PX`(px)를
  넘거나, `rotation_deg`의 차이가 `ROTATION_DIFF_THRESHOLD_DEG`(도)를
  넘으면 확인을 요구한다.
- 값을 낮추면 더 엄격해져 확인해야 할 파일이 늘어나고, 높이면 더 관대해져
  자동 적용되는 파일이 늘어난다. 설계 배경과 값 변경 이력은 2.3절 참고.

### 1.6 실행 전 확인할 점 / 되돌리는 방법

- 이 도구는 `data/annotation`의 파일을 **직접 수정**한다. 실행 전 자동
  백업(`data/annotation_backup_<타임스탬프>/`)이 만들어지지만, 다른
  프로그램(예: scan-font-browser.py)이 같은 폴더를 열어 저장 중이라면
  동시 실행을 피하는 것이 좋다.
- 결과가 마음에 들지 않으면 백업 폴더의 내용을 `data/annotation`에 다시
  복사해 되돌릴 수 있다.
- `data/scan`은 읽기만 하며 전혀 수정하지 않는다.
- 실행 도중 `Ctrl+C`로 중단해도 그때까지 처리된 파일은 이미 갱신되어
  있을 수 있다(파일 단위로 즉시 저장하기 때문). 백업이 있으므로 필요하면
  되돌리면 된다.

---

## 2부. 상세설계서

### 2.1 배경 및 설계 목표

`data/scan`을 annotation하는 초기에는 자동 보정 기능이 없어 사람이 격자
시작 좌표와 회전 보정 각도를 직접 입력했다. 이후 `scan-font-browser.py`에
자동 보정(위상 분석 + 바운딩 박스 + 회전 감지, 2.9절 참고)이 추가되어
정확도가 크게 좋아졌으므로, 이미 저장된 1,000개 이상의 annotation에도
같은 로직을 일괄로 적용해 품질을 끌어올리자는 것이 이 도구의 목적이다.

핵심 설계 원칙은 다음과 같다.

- **GUI와 로직을 분리한다**: 자동 보정 알고리즘 자체는
  `scripts/grid_autocorrect.py`라는 별도 모듈에 있고, GUI 상태(Tkinter)에
  전혀 의존하지 않는다. `scan-font-browser.py`(하이픈이 있어 일반적인
  `import`로 불러올 수 없다)와 이 배치 도구가 같은 함수를
  `from grid_autocorrect import ...`로 그대로 재사용한다. 이렇게 분리한
  이유는 두 도구가 서로 다른 구현을 갖게 되어 시간이 지나며 결과가
  달라지는 것을 막기 위함이다.
- **모든 변경은 되돌릴 수 있어야 한다**: 실행 전 항상 전체 백업을 만든다
  (2.6절).
- **애매한 경우는 사람이 판단한다**: 자동 보정 결과가 기존 값과 크게
  다르면 그 이유(예: 예외적인 스캔, 다른 폰트로 착각한 페이지 등)를 알 수
  없으므로 무조건 덮어쓰지 않고 사람에게 보여주고 확인을 받는다.

### 2.2 비교 대상을 만드는 방법 (`build_search_params`)

자동 보정 결과를 "새로 추정한 값"으로 취급하려면, 격자 탐색을 **기존
(수동) 시작 좌표에서 출발시키면 안 된다.** 그렇게 하면 위상 분석의 탐색
범위(`AUTO_CORRECT_RANGE`, ±20px)가 기존 값 근처로 편향되어, 자동 보정
결과가 기존 값과 비슷하게 나오는 것이 당연해져 버려 비교 자체가 의미를
잃는다.

그래서 `build_search_params`는 `scan-font-browser.py`가 **새 영상**을 열
때와 완전히 동일한 조건을 만든다.

- 시작 좌표(`origin_x`, `origin_y`)는 항상 `DEFAULT_GRID`의 표준값에서
  출발한다.
- 칸 크기(`cell_w`, `cell_h`)와 열/행 수(`cols`, `rows`)는 annotation에
  저장된 값을 그대로 쓴다. 모든 영상이 같은 인쇄 양식을 쓰므로 이
  값들은 사실상 항상 `DEFAULT_GRID`와 같지만(실제로 검사해 본 결과
  1,956개 파일 모두 기본값과 일치했다), 혹시 개별 영상에 맞춰 수동으로
  조정한 값이 있다면 그 계산을 존중한다.

### 2.3 허용 오차 임계값

```python
ORIGIN_DIFF_THRESHOLD_PX = 20.0
ROTATION_DIFF_THRESHOLD_DEG = 1.5
```

처음에는 `ORIGIN_DIFF_THRESHOLD_PX = 3.0`, `ROTATION_DIFF_THRESHOLD_DEG =
0.3`으로 시작했으나, 실제로 여러 폰트에 걸쳐 자동 보정과 수동 값을 비교해
본 결과(1.4절)를 반영해 각각 10.0px, 1.5도로 완화했다. 이 값이 클수록
"자동 적용"되는 파일이 늘어나고 확인을 요구하는 파일이 줄어든다 — 즉
자동 보정 결과를 더 신뢰한다는 뜻이다. 반대로 값을 낮추면 더 많은
파일에서 사람의 확인을 거치게 된다.

두 임계값은 **또는(OR)** 조건으로 결합된다: `origin_x`, `origin_y`,
`rotation_deg` 중 하나라도 자기 임계값을 넘으면 확인을 요구한다
(`process_one`의 `large_diff` 계산 참고).

### 2.4 파일 처리 흐름 (`process_one`)

파일 하나를 처리하는 절차는 다음과 같다.

1. JSON을 읽는다. 읽기 실패 시 `"error"`.
2. `zip`/`entry` 필드가 없으면 `"error"`.
3. 저장된 `grid.origin_x`/`grid.origin_y`/`rotation_deg`(기존 값)를
   읽는다. 없으면 `DEFAULT_GRID`/`0.0`을 기본값으로 쓴다.
4. `zip`에서 원본 영상을 읽는다(같은 zip은 `zip_cache`에 열어 둔 채
   재사용해 반복해서 열지 않는다). 실패 시 `"error"`.
5. `build_search_params`로 만든 `GridParams`를 `grid_autocorrect.
   estimate_origin_and_rotation`에 넘겨 자동 보정 값을 얻는다.
6. 기존 값과의 차이(`diff_x`, `diff_y`, `diff_rot`)를 계산한다.
   - 세 값이 모두 사실상 0이면(부동소수점 오차 이내) `"unchanged"`이며
     파일을 건드리지 않는다.
   - 하나라도 임계값(2.3절)을 넘으면(`large_diff`) 파일 이름/기존
     값/자동 보정 값/차이를 출력하고 `ask_yes_no`로 확인을 받는다.
     거절하면 `"skipped"`이며 파일을 건드리지 않는다.
7. 여기까지 왔다면(변화가 있고, 확인이 필요 없거나 확인을 통과함)
   `grid.origin_x`, `grid.origin_y`, `rotation_deg` 세 필드만 갱신하고
   JSON을 다시 저장한다. `font_name`, `first_char`, `last_char`,
   `char_count`, `zip`, `entry`, `image_width`/`image_height`,
   `cols`/`rows`/`cell_w`/`cell_h`는 건드리지 않는다.
8. 확인 없이 적용됐으면 `"updated"`, 확인 후 적용됐으면 `"confirmed"`를
   반환한다.

### 2.5 백업 메커니즘 (`backup_annotation_dir`)

`main()`은 파일을 하나라도 처리하기 전에 `shutil.copytree`로
`data/annotation` 전체를 `data/annotation_backup_<YYYYMMDD_HHMMSS>/`에
복사한다. 이 백업 폴더는 `data/` 아래에 생기므로 `.gitignore`의
`/data` 규칙에 의해 자동으로 git 추적에서 제외된다. 도구 자체는 오래된
백업을 자동으로 지우지 않으므로, 여러 번 실행하면 백업 폴더가 여러 개
쌓인다 — 필요 없어진 백업은 직접 정리한다.

### 2.6 모듈 구조 요약

| 구성 요소                                                  | 역할                                                           |
| ---------------------------------------------------------- | -------------------------------------------------------------- |
| `ORIGIN_DIFF_THRESHOLD_PX` / `ROTATION_DIFF_THRESHOLD_DEG` | "차이가 크다"의 기준 (모듈 상수, 2.3절)                        |
| `PROGRESS_INTERVAL`                                        | 진행 상황을 출력하는 간격(파일 개수, 기본 50)                  |
| `backup_annotation_dir()`                                  | `data/annotation` 전체를 타임스탬프 폴더로 백업 (2.5절)        |
| `build_search_params(annotation_grid)`                     | 비교용 `GridParams` 생성 — 시작 좌표는 항상 기본값에서 (2.2절) |
| `load_image(zip_cache, zip_name, entry)`                   | zip에서 원본 영상을 읽어 옴(zip 핸들 재사용)                   |
| `ask_yes_no(prompt)`                                       | `y`/`n` 확인을 받을 때까지 반복 질문                           |
| `process_one(path, zip_cache)`                             | annotation 파일 하나를 비교·갱신 (2.4절)                       |
| `main()`                                                   | 전체 목록을 순회하며 백업 → 처리 → 요약 출력                   |

`GridParams`, `DEFAULT_GRID`, `estimate_origin_and_rotation`은
`scripts/grid_autocorrect.py`에서 가져와 그대로 사용한다 (자세한 원리는
[docs/scan-font-browser.md](scan-font-browser.md) 2.9절 참고).

### 2.7 알려진 제한사항

- `grid_autocorrect`의 회전 추정과 마찬가지로, 대략 2도 이상 기울어진
  영상에서는 자동 보정 값 자체가 부정확할 수 있다(원리는
  scan-font-browser.md 2.9.4절 참고). 이런 영상은 큰 차이로 표시되어
  확인을 거치게 되므로, 화면에 나온 차이 값을 보고 신중하게 판단해야
  한다.
- 이 도구는 `grid`와 `rotation_deg`만 갱신한다. `first_char`/`last_char`
  등 글자 범위나 `font_name`이 애초에 잘못 입력된 경우는 이 도구로 고칠
  수 없다(scan-font-browser.py에서 직접 수정해야 한다).
- 진행 중 오류가 난 파일은 건너뛰고 계속 진행하지만, 오류의 원인(예:
  zip 파일 손상, 존재하지 않는 entry)은 별도로 조사해야 한다.
- 대량의 annotation을 처리하므로 실행 시간이 수 분 이상 걸릴 수 있다
  (파일당 회전이 감지되지 않으면 약 35ms, 감지되면 영상 재회전 때문에
  약 200ms 정도 소요된다 — scan-font-browser.md 2.9.4절 참고).
