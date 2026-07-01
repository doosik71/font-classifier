# Font Classifier

한글 폰트 인식 모델 개발 프로젝트. 최종 목표는 사용자가 제시한 사진 영상에서
한글 글자를 추출하고, 그 글자가 어떤 폰트로 쓰였는지 인식하는 앱을 만드는
것이다. 폰트 인식 모델은 딥러닝으로 학습한다.

> 현재는 학습 데이터(스캔 영상)를 확인하고 annotation을 작성하는 도구까지
> 만들어진 초기 단계이며, 모델 학습은 아직 시작하지 않았다.

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
scripts/                       # 사용자가 직접 실행하는 파이썬 스크립트
  scan-font-browser.py         # 스캔 영상 열람 + annotation 작성 GUI
  auto-correct-annotation.py   # 기존 annotation에 자동 보정을 일괄 적용하는 배치 도구
  grid_autocorrect.py          # 두 도구가 함께 쓰는 자동 격자/회전 보정 로직(GUI 비의존)
bin/                           # 스크립트 실행용 런처
  scan-font-browser.bat
  auto-correct-annotation.bat
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

## 진행 상황

- [x] 스캔 영상 브라우저 및 격자 확인 도구
- [x] annotation 입력/저장 기능 (폰트 이름, 첫/끝 글자, 격자 좌표, 회전 보정각)
- [x] 격자 시작 좌표·회전 자동 보정 및 기존 annotation 일괄 재보정 도구
- [ ] 전체 스캔 영상에 대한 annotation 작업
- [ ] annotation을 이용한 개별 글자 이미지 잘라내기(cropping) 및 학습 데이터셋 구성
- [ ] 한글 폰트 인식 모델 학습
- [ ] 사용자 사진에서 한글 글자 추출 + 폰트 인식 앱 개발
