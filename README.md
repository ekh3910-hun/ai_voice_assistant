# 나만의 AI 음성 비서

한국어 음성·문자로 메모와 할 일을 등록하고 관리하는 Streamlit 앱입니다.

## 설치 및 실행

Python 3.10 이상, Windows 또는 Linux에서 실행합니다.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app1_b.py
```

기본 앱 `app1_b.py`는 규칙 기반 명령 처리와 로컬 SQLite를 사용합니다.
`assistant_data.sqlite3`에 저장하며 별도 DB 자격증명이 필요하지 않습니다.
테스트 등에서 저장 위치를 분리하려면 `ASSISTANT_DB_PATH`를 지정합니다.
`app1.py`는 같은 UI와 SQLite를 사용하지만 학습된 의도 분류 모델을 사용합니다.
`pages/`와 `src/core.py`, `src/db.py`는 별도의 이전 구현으로, 기본 앱의 처리 경로와 다릅니다.
PostgreSQL 구현을 사용할 때만 `DATABASE_URL`과 `sql/schema.sql` 적용이 필요합니다.

## 사용 범위

- 대화: 메모 추가, 날짜가 있는 할 일 추가, 조회 안내, 도움말
- 메모 관리: 추가·조회·수정·삭제
- 할 일 관리: 날짜별 조회·추가·완료·삭제
- 처리 이력: 대화 처리 결과 및 CSV 내려받기
- 수정·완료·삭제 대화 명령은 관리 탭으로 안내합니다. 일정 수정 UI는 아직 없습니다.
- SQLite 데이터는 서버 인스턴스가 공유합니다. 사용자별 인증·데이터 분리는 구현되지 않았습니다.

예: `오늘 회의 내용 메모해 줘`, `내일 오후 3시 병원 예약 추가해 줘`.
녹음은 브라우저의 마이크 권한과 HTTPS(또는 로컬 개발 환경)가 필요합니다.
지원 음성은 0.5~15초, 10MB 이하이며 녹음·파일 업로드가 가능합니다.

## 음성 인식 및 합성

STT는 faster-whisper CPU/int8을 사용합니다. 기본 모델은 한국어 정확도를 우선하는 `small`입니다.
기존 `base`보다 모델 용량과 추론 비용이 커집니다. 실제 품질 향상 폭은 데이터로 검증해야 합니다.
`WHISPER_MODEL_SIZE=base`로 가벼운 모델을 선택하거나 `medium` 등으로 변경할 수 있습니다.
`WHISPER_MODEL_PATH`로 이미 다운로드한 CTranslate2 Whisper 모델 디렉터리를 지정하면 로컬 모델을 사용합니다.
한국어 명령 어휘 힌트와 VAD를 적용하며, 기본 경로에서는 중복 무음 제거로 어미가 잘리는 것을 피합니다.

최초 모델 다운로드에는 Hugging Face 접근이 필요합니다. 클라우드에서는 `HF_HOME=/workspace/.cache/huggingface`처럼 쓰기 가능한 캐시를 지정하세요.
TTS는 edge-tts를 사용하며 `speech.platform.bing.com` 연결이 필요합니다. 시스템 HTTPS 프록시를 따르고 30초 제한을 둡니다.
TTS가 실패해도 이미 처리한 명령과 저장 결과는 유지합니다.
클라우드 도메인 초안 저장만으로 실행 중 네트워크 정책이 변경되지는 않습니다.

## 검증 및 STT 비교

```bash
python -m unittest discover -s tests -v
python -m pip check
python test_stt.py --models base small --limit 6 --output /tmp/stt-comparison.csv
```

마지막 명령은 실제 음성을 인식하므로 모델 다운로드/캐시가 필요합니다.
모델별 첫 샘플 지연에는 로딩 시간이 포함됩니다. 실행 실패나 파일이 없는 경우 종료 코드 1을 반환합니다.
정확도는 사람이 검수한 정답 파일을 제공할 때만 계산합니다.

```csv
filename,text
sample.wav,사람이 녹음을 듣고 확인한 실제 문장
```

```bash
python test_stt.py --audio-dir data/audio_processed --models base small \
  --references /path/to/references.csv --output /tmp/stt-evaluation.csv
```

정답 파일에는 선택한 모든 WAV의 `filename,text`가 필요합니다. WER/CER는 낮을수록 좋습니다.
기존 `reports/stt_results.csv`는 이전 인식 결과이며 정답 전사로 간주하면 안 됩니다.
기존 의도 분류 보고서 역시 STT 또는 기본 규칙 기반 앱의 종단 성능을 나타내지 않습니다.
음성 테스트는 실제 오디오 전송·브라우저 마이크 권한까지 자동 검증하지는 않습니다.

학습 모델을 다시 만들려면 `python train_intent.py`를 실행합니다.
이 명령은 모델과 평가 보고서를 갱신하므로 변경 사항을 확인하세요.
전체 점검 결과는 [EVALUATION.md](EVALUATION.md)를 참고하세요.

## Render 배포

[Render에서 배포 설정 열기](https://render.com/deploy?repo=https://github.com/ekh3910-hun/ai_voice_assistant)

Render에 로그인한 뒤 위 링크에서 저장소 연결을 허용하고 Blueprint를 생성하세요.
`render.yaml`은 main 브랜치를 사용하고, Render가 제공하는 PORT에 Streamlit을 연결합니다.
서비스가 Live 상태가 되면 Render가 발급한 실제 `https://…onrender.com` 주소로 접속하세요.
이 링크 자체는 실행 중인 앱 주소가 아닙니다.

기본 설정은 요금 발생을 피하기 위한 무료 데모입니다. 유의 사항:

- SQLite와 모델 캐시는 임시 파일 시스템에 있습니다. 재배포·재시작 시 데이터와 캐시를 잃을 수 있으므로 중요한 메모를 저장하지 마세요.
- 무료 인스턴스는 유휴 시 중지될 수 있으며, 첫 접속 및 최초 모델 다운로드가 느릴 수 있습니다.
- small 모델 STT는 무료 인스턴스의 메모리 한도를 초과할 수 있습니다. 음성 시연이 필요하면 충분한 메모리의 인스턴스를 선택하고 비용을 확인하세요. 모델을 base/tiny로 줄이면 품질과 속도 조건도 바뀝니다.
- 영구 데이터가 필요하면 유료 영구 디스크에 ASSISTANT_DB_PATH를 지정하거나 별도 영구 저장소 구현이 필요합니다. 현재 SQLite 앱이 DATABASE_URL만으로 PostgreSQL로 전환되지는 않습니다.
- 인증·사용자별 데이터 분리가 없으므로 공개 URL에서는 다른 방문자와 데이터가 공유됩니다. 개인 정보 없이 데모 용도로 사용하세요.
- 배포 성공 후 실제 녹음, STT 및 TTS 연결은 별도로 확인해야 합니다.
