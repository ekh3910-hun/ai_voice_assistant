# 아래 코드를 app.py의 st.set_page_config(...) 바로 아래에 추가하세요.
# app.py 전체를 이 파일로 교체하지 마세요.
from pathlib import Path


def apply_common_style():
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(
            "<style>" + css_path.read_text(encoding="utf-8") + "</style>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("assets/style.css 파일을 찾을 수 없습니다.")


apply_common_style()
