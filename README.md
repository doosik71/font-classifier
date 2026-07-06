# Font Classifier

한글 폰트 인식 모델 개발 프로젝트. 최종 목표는 사용자가 제시한 사진 영상에서
한글 글자를 추출하고, 그 글자가 어떤 폰트로 쓰였는지 인식하는 앱을 만드는
것이다. 폰트 인식 모델은 딥러닝으로 학습한다.

> 현재는 annotation 작성·검증 도구와 학습용 낱글자 데이터셋 생성 도구,
> 그리고 폰트 인식 모델 학습 스크립트와 학습
> 진행 상황을 실시간으로 보는 모니터까지 갖춰졌으며, 모델 학습을 진행하는
> 단계다.

## 개발 환경

- Python, [uv](https://docs.astral.sh/uv/)로 패키지 및 가상환경을 관리한다.
- 이 프로젝트에서는 miniforge/conda 등 시스템 가상환경을 사용하지 않는다.
  오직 `uv`가 프로젝트 루트에 생성하는 `.venv`만 사용한다.

```bash
uv sync            # 의존성 설치 (.venv 생성/갱신)
uv run python ...  # .venv 안의 파이썬으로 스크립트 실행
```

## 프로젝트 구조

```text
data/                          # git 저장소에서 제외됨 (실행 중 생성/참조되는 데이터)
  scan/                        # 학습용 원천 스캔 영상 (zip). 원본이므로 절대 수정하지 않는다.
  annotation/                  # scan-font-browser로 작성한 영상별 annotation(json)
  dataset/                     # construct-dataset가 만든 폰트별 폴더/글자별 PNG + index.json (학습 입력)
  checkpoints/                 # 학습 산출물. 체크포인트(.pt) + metrics.jsonl
  results/                     # eval-model이 기록하는 평가 결과(eval.json)
scripts/                       # 사용자가 직접 실행하는 파이썬 스크립트(진입점)
  scan-font-browser.py         # 스캔 영상 열람 + annotation 작성 GUI
  auto-correct-annotation.py   # 기존 annotation에 자동 보정을 일괄 적용하는 배치 도구
  font-dataset-browser.py      # annotation을 이용해 폰트별 낱글자 영상을 추출/검증하는 GUI
  construct-dataset.py         # annotation+scan에서 폰트별 폴더/글자별 PNG 데이터셋 생성
  dataset-browser.py           # 생성된 학습 데이터셋(data/dataset)을 열람/검증하는 GUI
  train-model.py               # 폰트 인식 모델 학습(softmax cross entropy)
    train-monitor.py           # metrics.jsonl 변경을 감지해 학습 곡선을 실시간 그래프로 표시
  eval-model.py                # 체크포인트의 한글/폰트 인식 성능·속도를 평가해 eval.json에 기록
  font-classifier.py           # 학습된 모델로 낱글자의 한글/폰트를 인식하는 GUI 앱
font_classifier/               # 여러 스크립트가 import해서 쓰는 공유 모듈 패키지
  grid_autocorrect.py          # 자동 격자/회전 보정 로직
  char_extract.py              # 칸 하나에서 글자 하나를 64x64로 추출/정규화하는 로직
  font_dataset.py              # annotation을 폰트 단위로 묶어 글자→페이지를 매핑하는 로직
  dataset_loader.py            # 폰트 중심 학습 데이터셋(PNG+index.json)을 읽는 PyTorch Dataset
  batch_sampler.py             # "K개 폰트 x M개 글자" 폰트 기준 배치 샘플러
  char_batch_sampler.py        # "K개 글자 x M개 폰트" 글자 기준 배치 샘플러
  model.py                     # 폰트 인식 모델(공유 인코더 + 자소/폰트 헤더 + 디코더)
bin/                           # 스크립트 실행용 런처 (각 스크립트마다 .bat=Windows, .sh=Linux/macOS)
```

## 원천 데이터 (data/scan)

- zip 파일마다 1,000장의 jpg가 들어 있고, 파일명은 전체 데이터셋 안에서
  겹치지 않는 8자리 일련번호이다.
- 폰트 하나당 연속된 5장의 영상으로 구성된다 (500자 x 5장 = 2,350자).
  마지막 5번째 영상은 500자에 못 미칠 수 있다.
- 영상 한 장의 구성: 첫 줄에 폰트 이름이 인쇄되어 있고, 그 아래 가로 20열 x
  세로 25행 격자에 글자가 배치되어 있다.
- 글자 순서는 KS X 1001(완성형) 한글 2,350자 순서를 따른다. 이 순서는
  유니코드 한글 음절(U+AC00~U+D7A3) 중 `iso2022_kr` 코덱으로 인코딩 가능한
  글자만 코드값 순으로 골라내면 그대로 재현되며, 실제 스캔 영상과 대조하여
  확인했다 (`euc_kr`/`cp949`는 확장된 11,172자 전체를 허용하므로 사용할 수
  없다).

## Scan Font Browser

스캔 영상에서 jpg/글자 격자가 올바르게 추출되는지 확인하고, 학습에 필요한
annotation을 만드는 GUI 도구. 사용법과 내부 설계는
[docs/scan-font-browser.md](docs/scan-font-browser.md)에 자세히 정리되어
있다.

```bash
bin\scan-font-browser.bat
# 또는
uv run python scripts/scan-font-browser.py
```

주요 기능:

- 왼쪽에 ZIP 선택 콤보박스와 jpg 파일 목록을 표시한다. annotation이 저장된
  영상은 초록색, 저장되지 않은 영상은 빨간색으로 표시되어 진행 상황을 한눈에
  볼 수 있다.
- 메인 화면에 선택한 영상과 20열 x 25행 격자를 overlay로 표시하며, 격자
  좌표(시작 위치/칸 크기)는 영상마다 조금씩 다를 수 있어 화면에서 직접
  조정할 수 있다.
- 영상이 기울어진 경우를 위한 회전 보정 각도를 입력하면 그 값만큼 영상을
  회전시켜 화면에 보여준다.
- 폰트 이름과 첫 글자를 입력하면 격자 크기로부터 마지막 글자와 총 글자 수를
  자동 계산하고, 격자 각 칸 위에 예상 글자를 작게 표시해 인쇄된 실제 글자와
  일치하는지 바로 확인할 수 있다.
- "Annotation 저장" 버튼을 누르면 `data/annotation/<8자리번호>.json`에 폰트
  이름, 첫/마지막 글자, 글자 수, 회전 보정 각도, 격자 좌표를 저장한다. 같은
  영상을 다시 선택하면 저장된 값을 그대로 불러온다.
- 폰트 이름은 같은 폰트의 연속된 5장에서 매번 다시 입력하지 않도록 다음
  영상으로 넘어가도 값이 유지된다. 그 외 항목(첫 글자, 회전각, 격자)은
  영상마다 다르므로 새 영상을 선택하면 초기화된다.

## Auto Correct Annotation

`data/annotation`에 이미 저장된 annotation(대부분 자동 보정 기능이
생기기 전 사람이 직접 입력한 값)에, Scan Font Browser의 자동 격자/회전
보정 기능을 일괄로 다시 적용하는 텍스트 기반(GUI 없음) 배치 도구.
사용법과 내부 설계는
[docs/auto-correct-annotation.md](docs/auto-correct-annotation.md)에
자세히 정리되어 있다.

```bash
bin\auto-correct-annotation.bat
# 또는
uv run python scripts/auto-correct-annotation.py
```

실행 전 `data/annotation` 전체를 타임스탬프 폴더로 백업한 뒤, 각
annotation의 원본 영상에 자동 보정을 다시 적용해 기존 값과 비교한다.
차이가 작으면 바로 갱신하고, 차이가 크면 파일 이름과 기존/자동 보정
값, 차이를 보여주고 적용 여부를 물어본다.

## Font Dataset Browser

`data/annotation`에 저장된 annotation을 폰트 단위로 모아, 해당 스캔
영상에서 완성형 한글 2,350자 낱글자 영상을 추출/정규화해 보여주는 GUI
도구. annotation 정보의 무결성과, 그 정보로 낱글자 영상을 올바르게 잘라낼
수 있는지 확인하기 위한 것이며 읽기 전용이다(아무것도 저장하지 않는다).
사용법과 내부 설계는
[docs/font-dataset-browser.md](docs/font-dataset-browser.md)에 자세히
정리되어 있다.

```bash
bin\font-dataset-browser.bat
# 또는
uv run python scripts/font-dataset-browser.py
```

주요 기능:

- 왼쪽에 annotation이 있는 폰트 목록을 가나다순으로 표시한다. annotation
  이 상당량(100자 이상) 누락되어 검증할 수 없는 폰트는 목록에서 제외하고
  콘솔에 오류로 알린다. 목록에 포함된 폰트 중에도 일부(100자 미만) 글자가
  누락된 경우 주황색으로 표시한다.
- 폰트를 선택하면 원본 페이지와 같은 20열 격자로 2,350자를 이어붙여
  스크롤 가능한 화면에 표시한다. 각 칸은 Otsu 방법으로 찾은 잉크
  바운딩 박스를 종횡비를 유지한 채 긴 변이 64px가 되도록 확대 또는
  축소해 정규화한 그레이스케일 영상이며, 그 아래에 이 자리에 있어야
  할 글자를 레이블로 보여준다(이진화는 바운딩 박스를 찾는 데만 쓰고
  화면에는 원본에 가까운 그레이스케일로 표시한다. 원본에 작게
  인쇄된 폰트도 확대되어 비슷한 크기로 비교할 수 있다. 종횡비와,
  글자 크기 대비 획 두께의 상대적 비율은 한글 폰트 형태 분류에
  중요한 특징이므로 유지한다).
- annotation이 없어 표시할 수 없는 글자는 연한 빨강 테두리로, annotation은
  있지만 해당 폰트에 그 글자가 없어 빈 칸으로 인쇄된 경우는 눈에 잘
  띄도록 진한 빨강 테두리로 표시하고 콘솔에 경고를 출력한다.
- 한 페이지가 중복 스캔되어 같은 글자가 여러 페이지에 걸쳐 나타나면 먼저
  스캔된(파일명이 더 앞선) 페이지를 우선한다.

## Train Model

`data/dataset` 또는 `data/dataset-2`의 낱글자 데이터셋으로 한글 폰트 인식 모델을 학습하는
스크립트. `data/dataset-2`는 먼저 `data/dataset`을 만든 뒤
`construct-dataset-2.py`로 재배치해 준비한다.

학습 결과(체크포인트 + `metrics.jsonl`)는 `data/checkpoints` 아래에 저장한다.

- 폰트 헤더를 일반적인 softmax cross entropy로 학습.
  결과는 `data/checkpoints`에 저장. 설계/사용법:
  [docs/train-model.md](docs/train-model.md).

```bash
bin\train-model.bat     # 또는  uv run python scripts/train-model.py
uv run python scripts/train-model.py --dataset-layout char --dataset-dir data/dataset-2
```

- 자소(초/중/종성) 인식과 폰트 인식을 함께 학습하며, `--log-every` 스텝마다
  손실과 정확도(자소 개별·음절, 폰트 top-1/top-5)를 콘솔과
  `data/checkpoints/metrics.jsonl`에 기록한다.

## Train Monitor

학습 진행 상황을 실시간으로 보는 도구. `data/checkpoints/metrics.jsonl`의
변경을 주기적으로 감지해, 손실·정확도 곡선을 하나의
matplotlib 창에 겹쳐 그린다(읽기 전용 — 학습과 독립적으로 아무 때나 켜고
끌 수 있다). 사용법과 내부 설계는
[docs/train-monitor.md](docs/train-monitor.md)에 자세히 정리되어 있다.

```bash
bin\train-monitor.bat
# 또는
uv run python scripts/train-monitor.py
```

- 총손실 / 자소손실 / 폰트손실 / 자소정확도 / 폰트 top-1 / 폰트 top-5
  패널을 두고, 한눈에 비교한다.
- 파일 변경이 있을 때만 다시 그리며, 학습 도중 새로 생기는 run 폴더도
  자동으로 잡는다.

## Eval Model

명령행에서 지정한 체크포인트(기본 `data/checkpoints/latest.pt`)를, 지정한
데이터셋(기본 `data/dataset`) 전체에 대해 평가해 한글/폰트 인식 성능과 인식
속도를 측정하고 결과를 `data/results/eval.json`에 기록하는 스크립트. 사용법과
내부 설계는 [docs/eval-model.md](docs/eval-model.md)에 자세히 정리되어 있다.

```bash
bin\eval-model.bat
# 또는
uv run python scripts/eval-model.py
uv run python scripts/eval-model.py --checkpoint data/checkpoints/checkpoint-epoch-0003.pt
```

- `train-model.py`와 같은 `FontGlyphDataset`을 쓰되 augmentation을 끄고
  데이터 전체를 한 번 순회한다. 지표 정의(자소 개별/음절, 폰트 top-1/5)는
  학습 로그와 맞추고, 추가로 제한/개방 디코딩 글자 정확도와 폰트 top-10을
  낸다.
- 인식 속도는 encode+디코딩+폰트 top-k의 순수 연산 시간으로 측정한다(첫
  배치 워밍업 제외, 데이터 로딩/정답 대조 제외).

## Font Classifier (인식 앱)

학습된 폰트 인식 모델(`data/checkpoints/latest.pt`)로 **낱글자 하나**의
한글과 폰트를 인식해 결과를 보여 주는 Tkinter GUI 앱. 사용법과 내부 설계는
[docs/font-classifier.md](docs/font-classifier.md)에 자세히 정리되어 있다.

```bash
bin\font-classifier.bat
# 또는
uv run python scripts/font-classifier.py
```

- 입력은 **데이터셋**(폰트 목록 + 2,350자 격자에서 글자 칸 선택, 정답 표시)
  또는 **파일**(임의 영상을 열고 글자 하나를 박스로 선택; 박스가 없으면 영상
  전체를 낱글자로 처리)에서 고른다. 고른 영역은 학습 때와 같은
  `char_extract.normalize_glyph`로 64x64 정규화한다.
- 한글은 제한/개방 디코딩 결과를 함께 보여 주고, 폰트는 softmax 상위 10위의
  이름·확률과 "인식된 글자를 그 폰트로 쓴 글리프"(데이터셋에서 읽음)를 나란히
  표시한다. augmentation 적용 여부는 체크박스로 켜고 끌 수 있다.
- 폰트 이름 매핑은 `latest.pt`의 폰트 클래스 순서가 `data/dataset/index.json`의
  `id` 순서와 일치한다고 전제한다(자세한 주의사항은 docs 4절 참고).

## 진행 상황

- [x] 스캔 영상 브라우저 및 격자 확인 도구
- [x] annotation 입력/저장 기능 (폰트 이름, 첫/끝 글자, 격자 좌표, 회전 보정각)
- [x] 격자 시작 좌표·회전 자동 보정 및 기존 annotation 일괄 재보정 도구
- [x] annotation을 이용한 폰트별 낱글자 영상 추출/검증 도구 (Font Dataset Browser)
- [x] 학습용 낱글자 데이터셋 생성/검증 도구 (Construct Dataset / Dataset Browser)
- [x] 폰트 인식 모델 학습 스크립트와 실시간 학습 모니터 (Train Monitor)
- [ ] 한글 폰트 인식 모델 학습 완료
- [x] 낱글자 한글/폰트 인식 앱 (Font Classifier) — 파일(박스 선택)·데이터셋 입력
- [ ] 사용자 사진에서 여러 글자 자동 추출(세그멘테이션) + 폰트 인식
