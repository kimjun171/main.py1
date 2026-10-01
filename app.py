import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="서울 기온 예측기", layout="centered")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울의 과거 기온 데이터를 바탕으로 선형 회귀 분석을 진행하고, 미래 기온을 예측합니다.")

# 1. 데이터 로드
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    # CSV 데이터 읽기 (UTF-8 인코딩)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 날짜 컬럼을 datetime 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df["연도"] <= 2025]
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일이 300일 이상인 해만 필터링
    valid_data = yearly_stats[yearly_stats["관측일수"] >= 300].copy()
    return valid_data

data = load_data()

# 2. 선형 회귀 계산 (1908년 기준 경과 연수 사용)
# X: 1908년 기준 경과 연수 (Year - 1908)
data["경과연수"] = data["연도"] - 1908
X = data["경과연수"]
y = data["연평균기온"]

slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
r_squared = r_value ** 2

# 3. 주요 정보 안내 표시
min_year = int(data["연도"].min())
max_year = int(data["연도"].max())
total_years = len(data)

st.markdown(f"""
---
### 📊 데이터 요약 정보
- **분석에 사용된 연도 개수:** `{total_years}`개 연도
- **분석 시작 연도:** `{min_year}`년
- **분석 종료 연도:** `{max_year}`년
---
""")

# 4. 연도 선택 슬라이더 및 예측
st.subheader("🔮 연도별 예상 기온 예측")
selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

# 회귀 방정식을 통한 기온 예측: y = slope * (연도 - 1908) + intercept
pred_elapsed = selected_year - 1908
predicted_temp = slope * pred_elapsed + intercept

st.metric(label=f"{selected_year}년 예상 연평균 기온", value=f"{predicted_temp:.2f} °C")

# 5. Plotly 그래프 그리기
fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["연평균기온"],
    mode='markers',
    name='실제 연평균 기온',
    marker=dict(color='#1f77b4', size=8)
))

# 회귀 직선 (1900년 ~ 2100년 영역)
line_years = np.array(range(1900, 2101))
line_elapsed = line_years - 1908
line_pred = slope * line_elapsed + intercept

fig.add_trace(go.Scatter(
    x=line_years,
    y=line_pred,
    mode='lines',
    name='선형 회귀선',
    line=dict(color='#ff7f0e', width=2)
))

# 선택된 연도의 예측점 강조
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode='markers',
    name=f'{selected_year}년 예측점',
    marker=dict(color='red', size=12, symbol='star')
))

# 레이아웃 설정
fig.update_layout(
    title=f"서울 연평균 기온 및 추세선 (상관계수 R: {r_value:.4f}, R²: {r_squared:.4f})",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
