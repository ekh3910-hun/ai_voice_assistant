"""Real audio STT benchmark; optional human transcripts enable WER/CER scoring."""
import argparse
from pathlib import Path
import unicodedata

import pandas as pd
from src.stt import transcript_wav, evaluate_score


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='한국어 WAV STT 모델 비교 및 WER/CER 평가')
    parser.add_argument('--audio-dir', type=Path, default=root / 'data/audio_processed')
    parser.add_argument('--output', type=Path, default=root / 'reports/stt_results.csv')
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--models', nargs='+', default=[None], help='예: --models base small (미지정 시 앱 기본 모델)')
    parser.add_argument('--references', type=Path, help='사람이 검수한 filename,text 열의 CSV')
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error('--limit은 1 이상이어야 합니다')
    files = sorted(args.audio_dir.glob('*.wav'))[:args.limit]
    if not files:
        print('WAV 파일이 없습니다.')
        return 1
    references = {}
    if args.references:
        frame = pd.read_csv(args.references, keep_default_na=False)
        if not {'filename', 'text'} <= set(frame.columns) or frame['filename'].duplicated().any():
            parser.error('정답 CSV에는 중복 없는 filename,text 열이 필요합니다')
        references = dict(zip(frame['filename'], frame['text']))
        if any(not isinstance(references.get(p.name), str) or not references[p.name].strip() for p in files):
            parser.error('선택한 모든 WAV에 비어 있지 않은 정답 전사가 필요합니다')
    import os
    if len(args.models) > 1 and os.getenv('WHISPER_MODEL_PATH'):
        parser.error('모델 비교 시 WHISPER_MODEL_PATH를 해제하세요. 같은 로컬 모델로 중복 평가할 수 없습니다')
    results = []
    for model in args.models:
        for file_path in files:
            row = {'filename': file_path.name, 'model': model or os.getenv('WHISPER_MODEL_SIZE', 'small'),
                   'stt_text': '', 'elapsed_sec': None, 'error': ''}
            print(f"\n처리 중: {file_path.name} ({row['model']})")
            try:
                text, elapsed = transcript_wav(file_path, model_size=model, use_preprocess=False)
                row.update(stt_text=text, elapsed_sec=round(elapsed, 3))
                if references:
                    reference, prediction = normalize(references[file_path.name]), normalize(text)
                    row.update(reference=reference, **evaluate_score(reference, prediction))
                print('인식 결과:', text, '시간:', elapsed)
            except Exception as exc:
                row['error'] = str(exc)
                print('오류:', exc)
            results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False, encoding='utf-8-sig')
    passed = sum(not row['error'] for row in results)
    print(f'\nSTT {passed}/{len(results)} 성공. 결과: {args.output}')
    if not references:
        print('정답 전사가 없어 정확도(WER/CER)는 평가하지 않았습니다.')
    print('모델별 첫 샘플 시간에는 모델 로딩/다운로드가 포함됩니다.')
    return int(passed != len(results))


if __name__ == '__main__':
    raise SystemExit(main())
