# Streamlit 데이터 분석 앱 구현 계획

## 1. 조사 범위와 개발 원칙

이 문서는 기존 Colab/Gradio 프로그램을 변경하지 않고 Streamlit 버전을 설계하기 위한 사전 조사 결과다. 이번 단계에서는 실행 코드 전체를 구현하지 않는다.

- 기존 파일(`데이터분석_API.ipynb`, `데이터분석_API.py`, `데이터분석_API(gradio).py`)은 수정하지 않는다.
- 새 코드와 문서는 `streamlit_app/` 아래에만 둔다.
- 업로드 원본은 항상 `df`로 보존하고, 전처리 작업본은 반드시 `df_clean = df.copy()`로 시작한다.
- Streamlit UI, 데이터 입출력·전처리, 시각화, Gemini 연동, 실행 안전성 코드를 분리한다.
- 기존 함수는 그대로 복사하기보다 UI 의존성을 제거하고 테스트 가능한 순수 함수로 옮긴다.

## 2. 현재 저장소 구조

```text
Data Analysis by API+Colab/
├── 데이터분석_API.ipynb          # Colab 실행 원본(source of truth)
├── 데이터분석_API.py             # 노트북에서 생성한 코드 검토용 미러
├── 데이터분석_API(gradio).py     # Windows/브라우저용 Gradio 확장판
├── requirements.txt              # Gemini, Gradio, pandas, 시각화 의존성
├── README.md                     # 실행·GitHub/Colab 작업 안내
└── streamlit_app/
    └── IMPLEMENTATION_PLAN.md     # 본 문서
```

현재 브랜치는 `main`이며 조사 시점의 작업 트리는 깨끗했다. 별도의 `AGENTS.md`, 테스트 폴더, 패키지 구조는 없다.

## 3. 기존 프로그램의 주요 기능

### 3.1 Colab 노트북과 검토용 Python 파일

노트북은 다음 네 단계로 구성된다.

1. Colab 라이브러리·나눔 글꼴 설정
2. Colab Secret `exhibition`에서 Gemini API 키 로드
3. CSV/Excel 업로드, 인코딩 순차 판독, `df_clean = df.copy()` 생성
4. ipywidgets 기반 분석 채팅 UI

주요 기능은 다음과 같다.

- Gemini 모델 선택과 일반 분석/파이썬 자동 실행 모드
- 데이터 구조, 타입, 결측치, 기술통계, 상위 5행을 최대 60,000자로 요약
- 질문 및 이미지 첨부
- 대화 이력 저장·복원과 일반 텍스트 참고자료 등록
- 모델이 반환한 Python 코드 블록 실행
- 원본 `df` 수정 및 무관한 샘플 데이터로 `df_clean` 교체 방지
- 생성된 차트를 PNG로 보관하고 대화 문맥에 연결
- 출력만 지우기와 새 대화 시작을 구분

`데이터분석_API.py`는 노트북에서 생성된 검토용 파일이므로 새 Streamlit 코드가 직접 import할 대상으로 삼지 않는다. Colab 전용 `google.colab`, `ipywidgets`, `IPython.display` 의존성이 있기 때문이다.

### 3.2 Gradio 프로그램

Gradio 버전은 Colab 기능을 로컬 브라우저 환경으로 옮기면서 다음을 강화했다.

- API 키 입력 또는 `GEMINI_API_KEY` 환경변수 사용
- CSV/Excel 로딩과 5행 미리보기
- 세션별 DataFrame·파일명·대화 메시지 관리
- 정확한 컬럼명과 고유값 예시를 포함한 Gemini 컨텍스트
- 사용자가 질문에 명시한 컬럼과 생성 코드의 실제 컬럼 사용 여부 검증
- 제한된 built-in만 허용하는 Python 실행 namespace
- import, 파일, 프로세스, 네트워크, 특수 속성 접근 차단
- DataFrame/Series/빈도 사전을 표 결과로 변환
- MultiIndex·중복 컬럼·날짜·NumPy 값·NaN/Inf의 JSON 안전 변환
- 초광폭 표의 표시 열 제한, 여러 표와 여러 차트 순서 보존
- 실행 환경에 맞춘 한글 차트 글꼴 교정
- JSON 대화 저장 및 이전 JSON/텍스트 대화 불러오기

## 4. 재사용 가능 로직과 UI 전용 코드 구분

### 4.1 높은 우선순위로 재사용할 로직

| 기존 함수/영역 | 판단 | Streamlit 이관 방향 |
|---|---|---|
| `read_table` | 재사용 가능 | 업로드 bytes/파일 객체를 받도록 확장하고 CSV 인코딩·Excel 판독을 유지 |
| `load_dataset`의 `frame.copy()` 원칙 | 재사용 필수 | `df`와 `df_clean`을 별도 세션 키로 저장; 둘이 같은 객체가 되지 않도록 검증 |
| `make_data_context` | 재사용 가능 | 파일명 전역 의존 없이 순수 함수로 유지; 대용량/고카디널리티 컬럼 제한 추가 |
| `make_system_prompt` | 재사용 가능 | 프롬프트 템플릿을 별도 모듈로 이동 |
| `content_history` | 재사용 가능 | Streamlit 세션 메시지를 Gemini SDK 형식으로 변환 |
| `requested_column_names` | 재사용 가치 높음 | 정확한 컬럼명 검증에 사용하고 단위 테스트 추가 |
| `table_for_gradio` | 로직 재사용 | `normalize_table_for_display`로 이름 변경해 UI 중립화 |
| `captured_value_table` | 재사용 가능 | 자동 실행의 `display()` 결과 수집에 사용 |
| `displayable_wide_table` | 재사용 가능 | Streamlit 표시 정책과 다운로드 정책을 분리 |
| `decode_readable_file` | 재사용 가능 | 이전 대화/텍스트 참고자료 로딩에 사용 |
| `normalize_saved_messages` | 재사용 가능 | 저장 스키마 버전 호환 계층으로 이동 |
| `execute_python_blocks`의 AST 안전검사 | 핵심 재사용 | 실행기 모듈로 격리하고 보안 테스트 후 사용 |
| `apply_korean_chart_font` | 재사용 가능 | 시각화 유틸리티로 이동, 사용 가능 폰트 탐색 유지 |

### 4.2 수정 후 재사용할 로직

- `new_session`/`clone_session`: Gradio `gr.State` 대신 `st.session_state` 초기화 함수로 바꾼다.
- `execute_python_blocks`: 현재 namespace에서 `df_clean`이 입력 `frame`과 같은 객체를 참조한다. Streamlit에서는 실행 전 `df = source.copy(deep=True)`, `df_clean = df.copy(deep=True)`를 명시하여 원본과 작업본을 확실히 분리한다.
- `save_conversation`: 임시 파일 경로를 만드는 대신 JSON bytes를 반환해 `st.download_button`에서 내려받게 한다.
- 표 Markdown 변환: 채팅 기록 호환에는 사용할 수 있지만 화면 표시는 `st.dataframe`을 우선한다.
- Gemini 호출: UI 반환값 묶음 대신 서비스 결과 객체(답변, 갱신 DataFrame, 표, 차트, 오류)를 반환하도록 바꾼다.

### 4.3 재사용하지 않을 Gradio/Colab 전용 코드

- `gr.Blocks`, `gr.Row`, `gr.Dropdown`, `gr.File`, `gr.Chatbot`, `gr.State` 선언
- `.click()`, `.submit()` 이벤트 연결과 Gradio 입출력 tuple
- `gr.Error`, `gr.DownloadButton`, `demo.launch()`
- Gradio DOM을 겨냥한 `APP_CSS`
- ipywidgets 컴포넌트, `.observe()`, `.on_click()` 콜백
- Colab `files.upload/download`, `userdata.get`, `IPython.display`, JavaScript 스크롤 코드
- 전역 위젯과 전역 채팅 세션에 직접 접근하는 이벤트 핸들러

## 5. 제안하는 Streamlit 폴더 구조

```text
streamlit_app/
├── app.py                         # 진입점과 페이지 조립만 담당
├── requirements.txt              # Streamlit 앱 전용 의존성
├── README.md                      # 로컬 실행·환경변수·배포 안내
├── IMPLEMENTATION_PLAN.md         # 본 계획 문서
├── config/
│   └── settings.py                # 모델 목록, 크기/표시 제한, 기본값
├── core/
│   ├── models.py                  # 서비스 결과·전처리 단계 데이터 구조
│   ├── session.py                 # st.session_state 초기화/리셋
│   └── exceptions.py              # 사용자 표시용 예외
├── data/
│   ├── loader.py                  # CSV/Excel 판독과 원본 보존
│   ├── profiler.py                # 타입·결측·고유값·기술통계 프로파일
│   ├── preprocessing.py           # 전처리 미리보기/적용/되돌리기
│   └── validation.py              # 스키마, 컬럼, 변환 가능성 검사
├── analysis/
│   ├── prompts.py                 # 시스템/요청 프롬프트
│   ├── gemini_service.py          # Gemini SDK 호출과 대화 변환
│   ├── safe_executor.py           # 생성 Python AST 검사·격리 실행
│   └── history.py                 # 대화 저장/복원/버전 호환
├── visualization/
│   ├── builder.py                 # 차트 사양에서 Figure 생성
│   ├── recommendations.py         # 컬럼 타입별 권장 차트
│   └── fonts.py                   # 한글 글꼴 설정
├── ui/
│   ├── sidebar.py                 # API 키, 모델, 파일, 모드
│   ├── overview.py                # 원본/정제 데이터 요약
│   ├── preprocessing_view.py      # 전처리 UI
│   ├── visualization_view.py      # 상세 시각화 UI
│   └── chat_view.py               # Gemini 채팅·실행 결과 UI
└── tests/
    ├── test_loader.py
    ├── test_preprocessing.py
    ├── test_safe_executor.py
    ├── test_history.py
    └── test_visualization.py
```

`app.py`는 화면 배치와 호출 순서만 담당한다. `data/`, `analysis/`, `visualization/`은 Streamlit을 import하지 않는 것을 기본 규칙으로 삼아 독립 테스트가 가능하게 한다.

## 6. 화면 정보 구조 제안

- 사이드바: API 키, Gemini 모델, 데이터 파일, 이전 대화, 세션 초기화
- 탭 1 `데이터 개요`: 원본 크기, 컬럼 타입, 결측치, 중복, 고유값, 기술통계, 미리보기
- 탭 2 `전처리`: 작업 단계 선택, 변경 전/후 비교, 예상 영향, 적용·되돌리기
- 탭 3 `시각화`: 차트 유형, X/Y/색상/집계/정렬/필터, 스타일, 미리보기, PNG 다운로드
- 탭 4 `AI 분석`: 일반 분석과 안전한 코드 실행, 이미지 첨부, 표·차트 결과, 대화 저장

원본 `df`는 읽기 전용으로만 표시한다. 모든 전처리 및 AI 실행은 `df_clean`에만 적용하며, 적용 이력을 세션에 순서대로 기록한다.

## 7. 전처리 기능 개발 단계

### 1단계: 로딩·진단 기반

- CSV 인코딩(`utf-8-sig`, `cp949`, `utf-8`)과 XLSX/XLS 로딩
- `df` 적재 직후 `df_clean = df.copy(deep=True)` 생성
- 파일명, 행·열 수, 메모리, dtype, 결측치, 중복행, 고유값 요약
- 날짜/숫자/범주 후보 타입 탐지(자동 적용하지 않고 제안만 표시)
- 원본 해시 또는 구조 서명을 저장하여 불변성 검증

### 2단계: 기본 전처리

- 컬럼명 공백·중복 검사와 이름 변경
- 행/열 선택 및 불필요 컬럼 제거
- dtype 변환(숫자, 날짜, 문자열, 범주형)
- 결측치 제거 또는 상수/평균/중앙값/최빈값 대체
- 완전 중복 및 선택 컬럼 기준 중복 제거
- 문자열 공백 정리와 대소문자/표기 통일

### 3단계: 상세 전처리

- 숫자 이상치 탐지(IQR, z-score)와 제거/클리핑/표시
- 값 매핑·범주 병합, 구간화(bin), 파생 컬럼 생성
- 날짜 요소(연/월/분기/요일) 파생
- 필터 조건을 AND/OR로 조합
- 단계별 변경 행 수, 결측 변화, dtype 변화 비교

### 4단계: 이력·안전성

- 각 변환을 선언적 작업 사양으로 저장
- 미리보기 후 명시적 적용
- 한 단계 되돌리기, 전체 초기화, 원본 대비 diff 요약
- 정제 데이터 CSV/XLSX 다운로드
- 업로드 원본 불변성 자동 테스트

## 8. 시각화 기능 개발 단계

### 1단계: 기본 차트

- 막대, 선, 산점도, 히스토그램, 박스플롯
- 컬럼 타입에 따른 사용 가능한 축 자동 제한
- count/sum/mean/median 등 집계와 정렬
- 제목, 축 레이블, 범례, 색상 팔레트, 크기 설정
- 한글 폰트와 음수 기호 처리

### 2단계: 분석용 상세 차트

- 범주별 분포, 누적/그룹 막대, 시계열 추세
- 상관행렬, 결측치 맵, pair plot 또는 표본 기반 관계 탐색
- 상위 N개, 날짜 범위, 범주/수치 필터
- facet(소그룹) 및 색상/크기 인코딩
- 데이터가 큰 경우 표본 추출 사실과 건수를 명시

### 3단계: 품질·내보내기

- 차트별 필수 컬럼·dtype·집계 결과 검증
- 빈 결과, 단일값, 과도한 범주 수, NaN/Inf 경고
- 차트 PNG 다운로드와 사용한 필터/집계 사양 저장
- 같은 사양으로 재생성 가능한 시각화 설정 JSON
- 대표 데이터셋으로 시각 회귀/스모크 테스트

## 9. 권장 구현 순서와 완료 기준

1. **기반 구조**: 폴더, 설정, 세션, 의존성, 최소 앱 진입점 — 1차 완료
2. **데이터 계층**: 로더·프로파일러·원본 불변성 테스트 — 1차 완료
3. **전처리 MVP**: 결측, 노이즈 후보, 특정값 변경, 날짜 요소 분리, 결과 비교·다운로드 — 1차 완료
4. **시각화 MVP**: 6개 차트 계열, Pydantic 검증, n×n subplot, 텍스트/메뉴 요청, 통계 source·다운로드 — 1차 완료
5. **Gemini 연동**: 컨텍스트, 대화, 이미지, 저장/복원
6. **안전 실행기**: 기존 AST 방어 로직 이관 및 공격/회귀 테스트
7. **상세 기능**: 이상치, 파생변수, 고급 차트, 설정 저장
8. **통합 QA**: 대용량 파일, 한글 인코딩/폰트, 세션 초기화, 오류 메시지
9. **배포 준비**: Streamlit Community Cloud용 secrets/requirements/README

첫 구현 마일스톤의 완료 기준은 다음과 같다.

- 기존 세 프로그램의 diff가 비어 있다.
- 업로드 직후 `df is not df_clean`이며, 전처리 후에도 원본 값이 동일하다.
- UI 모듈 없이 데이터/분석 함수 테스트가 실행된다.
- CSV 세 인코딩과 XLSX를 읽고 명확한 오류를 표시한다.
- 전처리 변경 전/후 영향과 되돌리기가 작동한다.
- 차트가 한글·결측·빈 필터 결과를 안전하게 처리한다.
- 자동 실행기가 import, 파일, 프로세스, 네트워크, 원본 `df` 변경을 차단한다.

## 10. 구현 전 확인할 설계 결정

- Streamlit에서도 Gemini만 지원할지, 향후 다른 모델 공급자를 위한 인터페이스를 미리 둘지
- 자동 실행 모드를 기본 비활성화할지
- 최대 업로드 크기와 대용량 데이터 샘플링 기준
- 전처리 이력을 현재 세션에만 둘지 JSON으로 내보낼지
- 차트 라이브러리를 Matplotlib/Seaborn 중심으로 유지할지 Plotly 상호작용을 추가할지

기본 권장은 **Gemini 단일 공급자, 자동 실행 기본 비활성화, Matplotlib/Seaborn + 필요한 차트만 Plotly, 전처리 사양 JSON 내보내기**다. 이 선택은 기존 기능을 보존하면서 Streamlit 상세 전처리·시각화를 단계적으로 확장하기 쉽다.
