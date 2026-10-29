# pages/model_evaluate.py
# 모델 평가 페이지 (의도별 지표, 오류 사례, 혼동 행렬 시각화)

import streamlit as st
import pandas as pd
import altair as alt                                    # [추가] 시각화
from models.model import train_model, REPORT_PATH


# ════════════════════════════════════════════════════════════════════════════
# [추가] 시각화 설정
# ════════════════════════════════════════════════════════════════════════════
LOW_F1 = 0.88       # 이 값 미만인 의도는 주황색으로 강조

COLOR = {
    "Precision": "#3B6FE0",   # 정밀도 (파랑)
    "Recall":    "#2BA48A",   # 재현율 (초록)
    "F1":        "#F0904A",   # F1 (주황)
    "correct":   "#3B6FE0",   # 혼동행렬 정답 칸
    "wrong":     "#F0904A",   # 혼동행렬 오분류 칸
    "empty":     "#F0F2F6",   # 혼동행렬 0건 칸
}


# ════════════════════════════════════════════════════════════════════════════
# [추가] 데이터 가공
# ════════════════════════════════════════════════════════════════════════════
def build_confusion(metrics: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """intent_metrics.csv + intent_errors.csv로 혼동행렬을 다시 만듭니다.

    정답 칸 = 의도별 문장 수(count) − 그 의도에서 틀린 문장 수
    오답 칸 = intent_errors.csv의 (정답, 예측) 조합별 건수
    """
    intents = metrics["intent"].tolist()
    wrong = errors.groupby(["true_intent", "predict_intent"]).size()
    wrong_by_true = errors.groupby("true_intent").size()

    rows = []
    for t in intents:
        for p in intents:
            if t == p:
                n = int(metrics.loc[metrics["intent"] == t, "count"].iloc[0]) - int(wrong_by_true.get(t, 0))
            else:
                n = int(wrong.get((t, p), 0))
            rows.append({"실제": t, "예측": p, "건수": n, "구분": "정답" if t == p else "오분류"})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# [추가] 화면 구성 요소
# ════════════════════════════════════════════════════════════════════════════
def render_summary(metrics: pd.DataFrame):
    """정확도 · 재현율 · F1 요약 카드 (Macro 평균)"""
    total = metrics["count"].sum()
    accuracy = (metrics["Recall"] * metrics["count"]).sum() / total   # 맞힌 문장 수 / 전체

    c1, c2, c3 = st.columns(3)
    c1.metric("정확도 (Accuracy)", f"{accuracy:.1%}", border=True)
    c2.metric("재현율 (Macro Recall)", f"{metrics['Recall'].mean():.3f}", border=True)
    c3.metric("F1 score (Macro F1)", f"{metrics['F1'].mean():.3f}", border=True)


def render_per_intent_chart(metrics: pd.DataFrame):
    """의도별 정밀도 · 재현율 · F1 묶음 막대"""
    st.subheader('의도별 성능')
    low = metrics.loc[metrics["F1"] < LOW_F1, "intent"].tolist()
    st.caption("의도마다 정밀도 · 재현율 · F1을 나란히 비교합니다"
               + (f" · F1 {LOW_F1} 미만: {', '.join(low)}" if low else ""))

    long_df = metrics.melt(id_vars="intent", value_vars=["Precision", "Recall", "F1"],
                           var_name="지표", value_name="점수")
    order = ["Precision", "Recall", "F1"]

    chart = alt.Chart(long_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("intent:N", title=None, sort=metrics["intent"].tolist(),
                axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
        xOffset=alt.XOffset("지표:N", sort=order),
        y=alt.Y("점수:Q", scale=alt.Scale(domain=[0, 1.05]), title="점수",
                axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], format=".1f")),
        color=alt.Color("지표:N", sort=order, title=None,
                        scale=alt.Scale(domain=order, range=[COLOR[k] for k in order]),
                        legend=alt.Legend(orient="top")),
        tooltip=["intent", "지표", alt.Tooltip("점수:Q", format=".2f")],
    ).properties(height=320)

    st.altair_chart(chart, width="stretch")


def render_confusion_chart(metrics: pd.DataFrame, errors: pd.DataFrame):
    """마우스를 올리면 건수가 보이는 혼동행렬 히트맵"""
    st.subheader('혼동 행렬')
    st.caption("행 = 실제 의도, 열 = 예측 의도 · 파랑 = 정답, 주황 = 오분류")

    cm_df = build_confusion(metrics, errors)
    intents = metrics["intent"].tolist()

    base = alt.Chart(cm_df).encode(
        x=alt.X("예측:N", sort=intents, title="예측 의도",
                axis=alt.Axis(labelAngle=-40, labelOverlap=False, orient="bottom")),
        y=alt.Y("실제:N", sort=intents, title="실제 의도"),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        color=alt.condition(
            "datum.건수 == 0",
            alt.value(COLOR["empty"]),
            alt.Color("구분:N", legend=None,
                      scale=alt.Scale(domain=["정답", "오분류"],
                                      range=[COLOR["correct"], COLOR["wrong"]])),
        ),
        tooltip=["실제", "예측", "건수"],
    )
    numbers = base.mark_text(fontSize=13, fontWeight="bold").encode(
        text="건수:Q",
        color=alt.condition("datum.건수 == 0", alt.value("#A3A8B8"), alt.value("white")),
    )

    st.altair_chart((cells + numbers).properties(height=420), width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# 페이지 (기존 구성 유지 · [추가] 표시한 줄만 새로 넣음)
# ════════════════════════════════════════════════════════════════════════════
def render_evaluate_view():
    st.subheader('📃 모델 평가')

    metrics_path = REPORT_PATH / 'intent_metrics.csv'
    errors_path = REPORT_PATH / 'intent_errors.csv'

    has_report = metrics_path.exists()
    btn_label = '모델 재학습 및 평가 실행' if has_report else '모델 학습 및 평가 실행'

    if st.button(btn_label, type='primary'):
        with st.spinner('모델 학습 및 평가 중...'):
            train_model()
        st.success('모델 평가가 완료되었습니다')
        st.rerun()

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        render_summary(df)                                          # 재현율 · F1 요약
        st.subheader('의도별 평가 지표')
        st.dataframe(df, width="stretch", hide_index=True)
        render_per_intent_chart(df)                                 # 의도별 성능

        if errors_path.exists():
            errors = pd.read_csv(errors_path)
            st.subheader('대표 오류 사례')
            if errors.empty:
                st.info('오분류 사례가 없습니다')
            else:
                st.dataframe(errors, width="stretch", hide_index=True)

            render_confusion_chart(df, errors)                     # 혼동 행렬 시각화
    else:
        st.info('아직 평가 결과가 없습니다. 위의 버튼을 눌러 모델 평가를 실행해 보세요.')

if __name__ == '__main__':
    st.set_page_config(page_title="모델 평가", layout="centered")
    render_evaluate_view()