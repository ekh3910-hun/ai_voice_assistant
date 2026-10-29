"""app1 전용 의도 분류. DB/STT/TTS 의존성 없이 로컬 모델만 사용."""
from pathlib import Path
from functools import lru_cache

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'app1_intent_model.joblib'

@lru_cache(maxsize=1)
def _load_model(path, modified_ns, size):
    import joblib
    return joblib.load(path)

def classify_intent(text):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('먼저 프로젝트 폴더에서 python train_intent.py를 실행해 주세요.')
    stat = MODEL_PATH.stat()
    model = _load_model(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)
    probabilities = model.predict_proba([text])[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), float(probabilities[index])
