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
- 핵심 AI: `TfidfVectorizer` 및 분류기를 이용한 5개 의도 분류(`ADD_TASK`, `ADD_MEMO` 등) 및 슬롯 추출
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
1. `.env.example`을 참고해 실제 비밀키를 별도 설정합니다.
2. `sql/schema.sql`을 적용합니다.
3. 환경변수 이름과 의미: [ ]

## 6. 실행
```bash
streamlit run app.py
```

## 7. 테스트·평가 재현
- 테스트 명령: [ ]
- 모델 평가 명령: [ ]
- 데이터셋 출처: [ ]

## 8. 주의사항·제약
- 음성 데이터 저장 정책: [ ]
- 알려진 한계: [ ]
- 배포 링크: [ ]
