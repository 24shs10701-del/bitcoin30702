import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# 페이지 설정 (웹 브라우저 탭 제목 및 레이아웃)
st.set_page_config(page_title="비트코인 가격 분석 대시보드", layout="wide")

@st.cache_data
def load_data(file_path):
    """
    CSV 데이터를 로드하고 전처리하는 함수
    """
    # 데이터 불러오기 (세미콜론 구분자 처리)
    df = pd.read_csv(file_path, sep=';')
    
    # 시간 관련 컬럼을 datetime 객체로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 날짜순으로 정렬
    df = df.sort_values('timestamp')
    
    # 분석에 필요한 숫자형 데이터 타입 변환 (오류 발생 시 NaN 처리)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

def predict_next_day(df):
    """
    선형회귀 모델을 사용하여 내일 가격을 예측하는 함수
    """
    # 최근 30일 데이터 사용
    model_df = df.tail(30).copy()
    
    # 날짜를 숫자로 변환 (학습용)
    model_df['day_index'] = np.arange(len(model_df))
    
    X = model_df[['day_index']]
    y = model_df['close']
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일(다음 인덱스) 예측
    next_day_index = np.array([[len(model_df)]])
    prediction = model.predict(next_day_index)[0]
    
    return prediction

# 실제 데이터 파일명 (동일 폴더 내 coin.csv)
DATA_FILE = 'coin.csv'

try:
    # 데이터 로드
    df = load_data(DATA_FILE)
    
    # 사이드바 설정: 날짜 범위 선택 필터
    st.sidebar.header("📊 필터 설정")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    # 기간 선택 (기본값은 전체 기간)
    date_range = st.sidebar.date_input(
        "분석 기간 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # 시작일과 종료일이 모두 선택되었는지 확인
    if len(date_range) == 2:
        start_date, end_date = date_range
        # 선택한 기간에 맞춰 데이터 필터링
        filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)].copy()
    else:
        filtered_df = df.copy()
        start_date, end_date = min_date, max_date

    # 메인 타이틀 영역
    st.title("🪙 Bitcoin (BTC) 실시간 가격 분석 및 예측")
    st.markdown(f"**현재 분석 기간:** `{start_date}` ~ `{end_date}`")

    # 1. 상단 주요 지표 (Metrics)
    if not filtered_df.empty:
        latest = filtered_df.iloc[-1]  # 가장 최근 데이터
        previous = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest
        
        # 변동액 및 변동률 계산
        price_diff = latest['close'] - previous['close']
        price_diff_pct = (price_diff / previous['close']) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("현재 종가", f"₩{latest['close']:,.0f}", f"{price_diff_pct:.2f}%")
        m2.metric("기간 내 최고가", f"₩{filtered_df['high'].max():,.0f}")
        m3.metric("기간 내 최저가", f"₩{filtered_df['low'].min():,.0f}")
        m4.metric("누적 거래량", f"{filtered_df['volume'].sum():,.0f}")

    # 1.5 내일 가격 예측 섹션
    st.divider()
    st.subheader("🔮 AI 내일 가격 예측 (Linear Regression)")
    
    # 전체 데이터를 기준으로 예측 (최근 트렌드 반영)
    predicted_price = predict_next_day(df)
    current_actual_price = df.iloc[-1]['close']
    pred_diff = predicted_price - current_actual_price
    pred_diff_pct = (pred_diff / current_actual_price) * 100
    
    p_col1, p_col2 = st.columns([1, 2])
    
    with p_col1:
        if pred_diff > 0:
            st.success(f"### **상승 예측** 📈")
            st.write(f"내일 예상 가격: **₩{predicted_price:,.0f}**")
            st.write(f"현재가 대비 약 **{pred_diff_pct:+.2f}%** 변화가 예상됩니다.")
        else:
            st.error(f"### **하락 예측** 📉")
            st.write(f"내일 예상 가격: **₩{predicted_price:,.0f}**")
            st.write(f"현재가 대비 약 **{pred_diff_pct:+.2f}%** 변화가 예상됩니다.")
        
        st.caption("※ 최근 30일간의 선형 추세를 기반으로 한 예측이며, 실제 투자 결과와 다를 수 있습니다.")

    with p_col2:
        # 예측 시각화 차트
        recent_30 = df.tail(30).copy()
        next_date = recent_30['timestamp'].max() + timedelta(days=1)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=recent_30['timestamp'], y=recent_30['close'], name='최근 30일 실거래가'))
        fig_pred.add_trace(go.Scatter(x=[recent_30['timestamp'].iloc[-1], next_date], 
                                     y=[current_actual_price, predicted_price],
                                     name='내일 예측 지점',
                                     line=dict(dash='dash', color='yellow'),
                                     marker=dict(size=10, color='red')))
        
        fig_pred.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pred, use_container_width=True)
    st.divider()

    # 2. 메인 가격 차트
    st.subheader("📈 가격 추세 및 이동평균선 (MA)")
    
    # 이동평균선(MA) 계산 (20일, 50일)
    filtered_df['MA20'] = filtered_df['close'].rolling(window=20).mean()
    filtered_df['MA50'] = filtered_df['close'].rolling(window=50).mean()

    fig = go.Figure()
    # 종가 선 그래프
    fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['close'], 
                             name='종가 (Close)', line=dict(color='#00E676', width=2)))
    # 이동평균선 추가
    fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['MA20'], 
                             name='20일 이평선', line=dict(dash='dot', color='orange')))
    fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['MA50'], 
                             name='50일 이평선', line=dict(dash='dot', color='red')))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="날짜",
        yaxis_title="가격 (KRW)",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. 거래량 및 시장 지표 하단 배치
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📊 거래량 분석")
        fig_vol = go.Figure(data=[go.Bar(x=filtered_df['timestamp'], y=filtered_df['volume'], marker_color='#1E88E5')])
        fig_vol.update_layout(template="plotly_dark", height=400, xaxis_title="날짜", yaxis_title="거래량")
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with c2:
        st.subheader("💰 시가총액 추이")
        fig_cap = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['marketCap'], 
                                           fill='tozeroy', line_color='#FFD600')])
        fig_cap.update_layout(template="plotly_dark", height=400, xaxis_title="날짜", yaxis_title="Market Cap")
        st.plotly_chart(fig_cap, use_container_width=True)

    # 4. 데이터 원본 조회
    with st.expander("📝 원본 데이터 상세 보기"):
        st.write(f"총 {len(filtered_df)}개의 레코드가 검색되었습니다.")
        st.dataframe(filtered_df.sort_values('timestamp', ascending=False), use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ 파일을 찾을 수 없습니다: '{DATA_FILE}' 파일이 파이썬 실행 파일과 같은 경로에 있는지 확인해 주세요.")
except Exception as e:
    st.error(f"⚠️ 오류가 발생했습니다: {e}")
