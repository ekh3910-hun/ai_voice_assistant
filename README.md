# [나만의 AI 음성 비서 — 한국어 음성 명령 기반 메모·할 일 관리 서비스]

## 1. 프로젝트 개요
- 팀원 및 담당 기능: 팀장: 문윤성 / 팀원: 이태규, 엄기훈, 정유아, 최준혁
- 주요 사용자 시나리오:
1. 사용자가 마이크에 “내일 할 일에 발표 준비 추가해 줘”라고 말하면<br/>
2. 시스템이 STT로 사용자의 음성을 텍스트로 전사합니다.<br/>
3. 핵심 AI 기능으로 전사된 텍스트의 의도를 ADD_TASK로 분류하고, 규칙 기반 슬롯 파싱을 통해 날짜와 내용을 추출합니다.<br/>
4. TTS가 "내일 할 일에 발표 준비를 추가했습니다."라고 음성으로 피드백을 제공하고<br/>
5. DB에 추출된 할 일 데이터를 PostgreSQL에 저장합니다.
- 서비스 화면:

## 2. 기능
- STT: `faster-whisper`를 활용한 고정밀 한국어 음성 전사
- 핵심 AI: `TfidfVectorizer` 및 분류기를 이용한 9개 의도 분류와 규칙 기반 슬롯 추출
- 입력: WAV·MP3, 한국어 음성 및 텍스트 명령
- 확장 명령: `30일`, `이십구일` 날짜 표현, 대상 날짜를 지정한 수정, 할 일 일괄 완료·삭제, 최근 메모 삭제
- 장애 처리: 불명확한 명령 보류, DB 실패 시 화면 상태 보존, STT·TTS 실패 안내
- TTS: `edge-tts`를 활용한 자연스러운 안내 음성 생성
- DB: PostgreSQL(Neon)을 활용한 메모, 할 일 및 처리 이력 관리

## 3. 실행 환경
- Python: 3.10 이상
- 운영체제: Windows
- 외부 API 및 비용/접근 조건: `edge-tts` 사용을 위한 인터넷 연결 필요

## 4. 설치
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 5. 환경변수 및 DB
1. `DATABASE_URL`을 환경변수 또는 Streamlit Secrets에 설정합니다. 로컬에서는 `.env`도 사용할 수 있습니다.
2. `sql/schema.sql`을 적용합니다.
3. `DATABASE_URL`: PostgreSQL 연결 문자열. Git에 업로드하지 않습니다.
4. DB가 없으면 목록은 빈 상태이며, 화면에서 추가한 세션 데이터는 새 세션에 보존되지 않습니다. 음성·텍스트 명령으로 영구 저장하려면 DB 연결이 필요합니다.

## 6. 실행
```bash
streamlit run app.py
```

## 7. 테스트·평가 재현
- 회귀 테스트: `python -m unittest discover -s tests -p test_acceptance.py -v`
- 통합 기능 테스트: `python -m unittest discover -s tests -p test_integration.py -v`
- 화면 테스트: `python -m unittest discover -s tests -p test_ui.py -v`
- 테스트 범위와 미검증 항목: [테스트 결과서](reports/07_테스트계획_결과서.md)
- 모델 평가 명령: `python -m models.model`
- 데이터셋 출처: [ ]

## 8. 주의사항·제약
- 음성 데이터 저장 정책: [ ]
- 알려진 한계: 처리 목록은 최대 100개 기준입니다. 실제 STT·TTS·DB 연동 및 브라우저 음성 청취는 배포 환경에서 별도 확인이 필요합니다.
- 기존 배포 명령 `streamlit run app1_b.py`도 `app.py`로 연결됩니다.
- 배포 링크: [ ]
