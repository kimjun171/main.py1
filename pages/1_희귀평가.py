import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="서울 기온 예측기 & 모델 평가", layout="wide")

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 평가 및 비교")
st.write("학습 기간(전체 / 과거 100년 / 과거 50년)에 따른 선형회귀 모델의 기울기 변화와 최근 20년(2006~2025년) 테스트 데이터에 대한 예측 성능을 비교합니다.")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
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
    valid_data["경과연수"] = valid_data["연도"] - 1908
    return valid_data

data = load_data()

# 2. 데이터 세트 분할
# 공통 테스트 데이터: 최근 20년 (2006~2025)
test_df = data[(data["연도"] >= 2006) & (data["연도"] <= 2025)].copy()

# 훈련 데이터 세트
train_full = data.copy()
train_100y = data[(data["연도"] >= 1906) & (data["연도"] <= 2005)].copy()
train_50y = data[(data["연도"] >= 1956) & (data["연도"] <= 2005)].copy()

# 3. 모델 학습 및 평가 함수
def evaluate_model(train_data, test_data, name):
    # 학습 (1908년 기준 경과연수 사용)
    slope, intercept, r_val, _, _ = stats.linregress(train_data["경과연수"], train_data["연평균기온"])
    
    # 훈련 데이터 자체 평가
    y_train_pred = slope * train_data["경과연수"] + intercept
    train_mae = np.mean(np.abs(train_data["연평균기온"] - y_train_pred))
    train_mse = np.mean((train_data["연평균기온"] - y_train_pred) ** 2)
    train_r2 = r_val ** 2
    
    # 공통 테스트 데이터(2006~2025) 예측 평가
    y_test_true = test_data["연평균기온"]
    y_test_pred = slope * test_data["경과연수"] + intercept
    
    test_mae = np.mean(np.abs(y_test_true - y_test_pred))
    test_mse = np.mean((y_test_true - y_test_pred) ** 2)
    
    # 테스트 R2 계산: 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_test_true - y_test_pred) ** 2)
    ss_tot = np.sum((y_test_true - np.mean(y_test_true)) ** 2)
    test_r2 = 1 - (ss_res / ss_tot)
    
    return {
        "모델명": name,
        "학습 연도 범위": f"{int(train_data['연도'].min())}~{int(train_data['연도'].max())}",
        "학습 데이터 수": len(train_data),
        "기울기 (100년당)": slope * 100,
        "절편": intercept,
        "slope_raw": slope,
        "Train R²": train_r2,
        "Test MAE": test_mae,
        "Test MSE": test_mse,
        "Test R²": test_r2
    }

res_full = evaluate_model(train_full, test_df, "전체 데이터 모델")
res_100y = evaluate_model(train_100y, test_df, "과거 100년 모델 (1906~2005)")
res_50y = evaluate_model(train_50y, test_df, "과거 50년 모델 (1956~2005)")

results = [res_full, res_100y, res_50y]

# 4. 성능 비교 표 출력
st.subheader("📊 모델별 기울기 및 테스트 데이터(2006~2025) 예측 성능 평가")

summary_df = pd.DataFrame([{
    "모델 구분": r["모델명"],
    "학습 기간": r["학습 연도 범위"],
    "학습 데이터 수": f"{r['학습 데이터 수']}개",
    "기울기 (100년당 상승 폭)": f"+{r['기울기 (100년당)']:.2f} °C",
    "Train R²": f"{r['Train R²']:.4f}",
    "Test MAE (°C)": f"{r['Test MAE']:.4f}",
    "Test MSE (°C²)": f"{r['Test MSE']:.4f}",
    "Test R²": f"{r['Test R²']:.4f}"
} for r in results])

st.dataframe(summary_df, use_container_width=True)

# 5. 핵심 인사이트 요약 메트릭
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🌐 전체 데이터 모델 기울기",
        value=f"+{res_full['기울기 (100년당)']:.2f} °C/100년",
        delta=f"Test MAE: {res_full['Test MAE']:.2f}°C"
    )

with col2:
    st.metric(
        label="📜 과거 100년 모델 (1906~2005)",
        value=f"+{res_100y['기울기 (100년당)']:.2f} °C/100년",
        delta=f"Test MAE: {res_100y['Test MAE']:.2f}°C",
        delta_color="off"
    )

with col3:
    diff_slope = res_50y['기울기 (100년당)'] - res_100y['기울기 (100년당)']
    st.metric(
        label="🔥 과거 50년 모델 (1956~2005)",
        value=f"+{res_50y['기울기 (100년당)']:.2f} °C/100년",
        delta=f"100년 모델 대비 +{diff_slope:.2f}°C 가속",
        delta_color="inverse"
    )

st.markdown("""
> **💡 분석 요약**
> - **기울기 비교**: 최근 50년(1956~2005) 데이터로 학습한 모델의 기울기가 과거 100년(1906~2005) 모델보다 현저히 가파릅니다. 이는 20세기 후반 이후 **기온 상승 속도가 가속화**되었음을 나타냅니다.
> - **테스트 데이터 예측 성능**: 지구 온난화가 가속되는 경향 때문에, **최근 50년 데이터를 학습한 모델**이 최근 20년(2006~2025) 실제 기온을 오차(MAE, MSE)가 더 작고 높은 $R^2$ 수치로 **가장 잘 예측**함을 확인할 수 있습니다.
""")

st.markdown("---")

# 6. 연도 선택 슬라이더 및 각 모델별 예측 기온 비교
st.subheader("🔮 모델별 미래/과거 기온 예측 비교")
selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

pred_elapsed = selected_year - 1908
p_full = res_full["slope_raw"] * pred_elapsed + res_full["절편"]
p_100y = res_100y["slope_raw"] * pred_elapsed + res_100y["절편"]
p_50y = res_50y["slope_raw"] * pred_elapsed + res_50y["절편"]

p_col1, p_col2, p_col3 = st.columns(3)
p_col1.metric(f"{selected_year}년 예상 (전체 모델)", f"{p_full:.2f} °C")
p_col2.metric(f"{selected_year}년 예상 (과거 100년 모델)", f"{p_100y:.2f} °C")
p_col3.metric(f"{selected_year}년 예상 (과거 50년 모델)", f"{p_50y:.2f} °C")

# 7. Plotly 시각화 (회귀선 3개 + 실제 데이터 구분)
fig = go.Figure()

# 훈련 데이터 산점도 (2005년 이전)
train_pts = data[data["연도"] <= 2005]
fig.add_trace(go.Scatter(
    x=train_pts["연도"],
    y=train_pts["연평균기온"],
    mode='markers',
    name='학습 데이터 (~2005년)',
    marker=dict(color='#1f77b4', size=7, opacity=0.7)
))

# 테스트 데이터 산점도 (2006~2025년)
fig.add_trace(go.Scatter(
    x=test_df["연도"],
    y=test_df["연평균기온"],
    mode='markers',
    name='테스트 데이터 (2006~2025년)',
    marker=dict(color='red', size=9, symbol='diamond')
))

# 회귀 직선 그리기 (1900년 ~ 2100년)
line_years = np.array(range(1900, 2101))
line_elapsed = line_years - 1908

# 1. 전체 모델 선
fig.add_trace(go.Scatter(
    x=line_years,
    y=res_full["slope_raw"] * line_elapsed + res_full["절편"],
    mode='lines',
    name=f'전체 모델 (+{res_full["기울기 (100년당)"]:.2f}°C/100년)',
    line=dict(color='#ff7f0e', width=2)
))

# 2. 과거 100년 모델 선
fig.add_trace(go.Scatter(
    x=line_years,
    y=res_100y["slope_raw"] * line_elapsed + res_100y["절편"],
    mode='lines',
    name=f'과거 100년 모델 (+{res_100y["기울기 (100년당)"]:.2f}°C/100년)',
    line=dict(color='#2ca02c', width=2, dash='dash')
))

# 3. 과거 50년 모델 선
fig.add_trace(go.Scatter(
    x=line_years,
    y=res_50y["slope_raw"] * line_elapsed + res_50y["절편"],
    mode='lines',
    name=f'과거 50년 모델 (+{res_50y["기울기 (100년당)"]:.2f}°C/100년)',
    line=dict(color='#e377c2', width=2, dash='dot')
))

# 선택 연도 예측점 표시
fig.add_trace(go.Scatter(
    x=[selected_year, selected_year, selected_year],
    y=[p_full, p_100y, p_50y],
    mode='markers',
    name=f'{selected_year}년 예측 위치',
    marker=dict(color='black', size=10, symbol='star')
))

fig.update_layout(
    title="서울 연평균 기온 선형 회귀선 비교 (1900년 ~ 2100년)",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
