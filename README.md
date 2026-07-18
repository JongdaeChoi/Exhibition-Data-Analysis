# Data Analysis by API + Colab

Google Colab에서 Gemini API를 사용해 CSV/Excel 데이터를 탐색하는 노트북 프로젝트입니다.

## Notebook

- `데이터분석_API.ipynb`
- `데이터분석_API.py`: GitHub/Codex에서 Python 코드를 쉽게 확인하기 위한 읽기용 스크립트
- 실행 환경: Google Colab
- API 인증: Colab Secrets의 `exhibition` 키

노트북이 실제 원본(source of truth)입니다. `.py` 파일은 코드 검토용이므로 실행과 수정은 노트북을 기준으로 합니다.

## Gradio 테스트 앱

`데이터분석_API(gradio).py`는 Colab 전용 `ipywidgets` UI를 Windows 브라우저에서도 시험할 수 있도록 만든 별도 앱입니다.

```powershell
python -m pip install -r requirements.txt
python ".\데이터분석_API(gradio).py"
```

실행 후 브라우저가 자동으로 열립니다. 화면에서 Gemini API 키와 CSV/Excel 파일을 등록한 뒤 질문할 수 있습니다. API 키는 파일에 저장되지 않습니다.

모드에서 `파이썬 코드 자동 실행`을 선택하면 Gemini가 반환한 Python 코드 블록을 제한된 분석 환경에서 실행합니다. 실행 상태, 표, 차트는 모델의 코드 응답 바로 다음에 채팅 메시지로 순서대로 표시됩니다. 파일·프로세스·네트워크 접근과 임의 모듈 import는 차단됩니다.

## Streamlit Business Insight

`streamlit_app`의 **인사이트** 단계는 현재 `df_clean`, 전처리 이력, 기술통계와 저장된 시각화 통계자료를 Gemini에 전달합니다.

- 지원 모델: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.1-pro-preview`
- 텍스트 요청: 설명, 분석 또는 보고형 Markdown 답변
- 차트 요청: Gemini가 구조화한 Pydantic 설정을 기존 통계·차트 함수로 검증 및 실행
- 저장·복원: 대화와 차트가 포함된 JSON, 읽기용 Markdown 다운로드 및 기존 인사이트 업로드

Colab에서는 Secrets에 `exhibition` 이름으로 Gemini API 키를 등록합니다. 로컬 실행에서는 화면의 비밀번호 입력란이나 `GEMINI_API_KEY` 환경변수를 사용할 수 있습니다. API 키는 다운로드 파일에 저장되지 않습니다.

## 처음 한 번: GitHub 저장소 연결

GitHub에서 빈 저장소를 만든 뒤, 이 폴더의 PowerShell에서 아래 명령을 실행합니다.

```powershell
git remote add origin https://github.com/YOUR_GITHUB_ID/YOUR_REPOSITORY.git
git add .
git commit -m "Add Gemini data analysis notebook"
git push -u origin main
```

GitHub가 비밀번호를 요청하면 계정 비밀번호 대신 Personal Access Token을 사용하거나 Git Credential Manager의 브라우저 로그인을 완료합니다.

## Colab에서 열고 저장하기

1. [Google Colab](https://colab.research.google.com/)을 엽니다.
2. **File > Open notebook > GitHub**에서 저장소 URL을 입력합니다.
3. 노트북을 선택합니다.
4. Colab 왼쪽의 **Secrets**에서 이름이 `exhibition`인 Gemini API 키를 등록하고 **Notebook access**를 켭니다.
5. 수정 후 **File > Save a copy in GitHub**를 선택합니다.
6. 대상 저장소와 `main` 브랜치를 선택하고 의미 있는 commit message를 입력합니다.

> Colab의 **Save a copy in GitHub**는 새 커밋을 GitHub에 직접 만듭니다. 이후 Codex에서 작업하기 전에 반드시 `git pull`로 최신 커밋을 받으세요.

## 매 작업의 권장 순서

### Codex에서 시작할 때

```powershell
git pull --rebase origin main
git status
```

Codex로 수정한 뒤 변경 내용을 확인하고 커밋합니다.

```powershell
git diff
git add .
git commit -m "Describe the notebook change"
git push origin main
```

### Colab에서 시작할 때

GitHub 탭에서 노트북을 다시 열어 최신 버전을 사용합니다. 작업을 마친 뒤에는 **Save a copy in GitHub**로 커밋합니다.

## 충돌을 피하는 핵심 규칙

- Codex와 Colab에서 같은 노트북을 동시에 수정하지 않습니다.
- 작업 시작 전에 항상 최신 GitHub 버전을 가져옵니다.
- 한 작업 단위마다 작게 커밋하고 바로 push합니다.
- API 키와 원본 데이터 파일은 GitHub에 올리지 않습니다.
- Colab 출력이 불필요하게 커졌다면 **Edit > Clear all outputs** 후 저장합니다.

## 파일 정책

`.gitignore`는 로컬 데이터 파일, API 키 파일, 가상환경, Jupyter 체크포인트, 생성된 채팅 JSON을 제외합니다. 재현에 필요한 작은 샘플 데이터가 있다면 `*.csv` 등의 무시 규칙에 예외를 명시한 뒤 추가하세요.
