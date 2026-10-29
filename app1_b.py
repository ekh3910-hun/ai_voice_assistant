"""기존 배포 서비스의 실행 명령과 호환되는 진입점입니다."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('app.py')), run_name='__main__')
