import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


# ==========================================
# 1. 모델 및 데이터 불러오기
# ==========================================

direction_model = tf.keras.models.load_model(
    "ATCH_direction_model.keras"
)

return_model = tf.keras.models.load_model(
    "ATCH_return_model.keras"
)

direction_scaler = joblib.load(
    "ATCH_direction_scaler.pkl"
)

return_scaler = joblib.load(
    "ATCH_return_scaler.pkl"
)

data = pd.read_csv(
    "ATCH_AI_final_dataset.csv"
)


# ==========================================
# 2. 모델 입력 변수 48개
# ==========================================

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Volume_MA20",
    "Volume_Change",
    "Return_1D",
    "MA5",
    "MA20",
    "MA60",
    "Volatility_20D",
    "RSI14",
    "SPY_Return",
    "QQQ_Return",
    "Filing_Count",
    "Filing_8_K",
    "Filing_10_Q",
    "Filing_10_K",
    "Filing_S_1",
    "Filing_S_3",
    "Filing_DEF 14A",
    "Daily_Event_Score",
    "Revenue",
    "Net_Income",
    "Assets",
    "Liabilities",
    "Stockholders_Equity",
    "Cash",
    "Operating_Cash_Flow",
    "Market_Cap",
    "PS_Ratio",
    "PB_Ratio",
    "Debt_Ratio",
    "Cash_to_Liabilities",
    "ROA",
    "ROE",
    "Revenue_Growth",
    "Short_Interest",
    "Short_Interest_Ratio",
    "Short_Volume",
    "Short_Volume_Ratio",
    "Short_Interest_to_Volume",
    "Short_Ratio_Change",
    "Days_To_Earnings",
    "Earnings_Within_5D",
    "Earnings_Within_10D",
    "Earnings_Within_20D"
]


# ==========================================
# 3. 예측 함수
# ==========================================

def predict():

    try:

        # 데이터 날짜 정렬
        data_sorted = data.copy()

        if "Date" in data_sorted.columns:

            data_sorted["Date"] = pd.to_datetime(
                data_sorted["Date"]
            )

            data_sorted = data_sorted.sort_values(
                "Date"
            ).reset_index(drop=True)

        # 가장 최근 데이터 선택
        latest = data_sorted.iloc[-1:].copy()

        # 필요한 변수 확인
        missing = [
            col
            for col in features
            if col not in latest.columns
        ]

        if missing:

            return (
                "## 오류 발생\n\n"
                "다음 변수가 데이터에 없습니다:\n\n"
                + "\n".join(
                    f"- {col}"
                    for col in missing
                )
            )

        # 숫자형으로 변환
        X = latest[features].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # 결측치 확인
        if X.isna().any().any():

            missing_values = X.columns[
                X.isna().any()
            ].tolist()

            return (
                "## 오류 발생\n\n"
                "예측 데이터에 결측치가 있습니다:\n\n"
                + "\n".join(
                    f"- {col}"
                    for col in missing_values
                )
            )

        # ======================================
        # 상승 확률 예측
        # ======================================

        X_direction = direction_scaler.transform(X)

        probability = direction_model.predict(
            X_direction,
            verbose=0
        )[0][0]

        probability = float(probability)

        # ======================================
        # 다음날 수익률 예측
        # ======================================

        X_return = return_scaler.transform(X)

        expected_return = return_model.predict(
            X_return,
            verbose=0
        )[0][0]

        expected_return = float(expected_return)

        # ======================================
        # 현재 주가
        # ======================================

        current_price = float(
            latest["Close"].iloc[0]
        )

        # ======================================
        # 날짜
        # ======================================

        if "Date" in latest.columns:

            date_text = str(
                latest["Date"].iloc[0]
            )[:10]

        else:

            date_text = "최근 데이터"

        # ======================================
        # 상승 / 하락 판단
        # ======================================

        if probability >= 0.5:

            direction = "상승 가능성"

        else:

            direction = "하락 가능성"

        # ======================================
        # 결과
        # ======================================

        result = f"""
# ATCH AI Stock Predictor

### 기준 날짜

**{date_text}**

### 최근 종가

**${current_price:.2f}**

---

## 상승 / 하락 예측

**상승 확률:** {probability * 100:.2f}%

**모델 판단:** {direction}

---

## 다음날 수익률 예측

**예상 수익률:** {expected_return * 100:.2f}%

---

### 프로그램 설명

본 프로그램은 공개된 주가 및 기업 관련 데이터를
입력값으로 사용하여 인공신경망(ANN)을 학습시키고,

1. 다음 거래일의 상승 가능성
2. 다음 거래일의 예상 수익률

을 예측합니다.

※ 본 결과는 인공지능 모델의 예측값이며
실제 주가 변동이나 미래 수익을 보장하지 않습니다.
"""

        return result

    except Exception as e:

        return (
            "## 오류 발생\n\n"
            f"`{str(e)}`"
        )


# ==========================================
# 4. Gradio 인터페이스
# ==========================================

demo = gr.Interface(
    fn=predict,
    inputs=[],
    outputs=gr.Markdown(),
    title="ATCH AI Stock Predictor",
    description=(
        "공개 데이터를 이용한 "
        "인공신경망 기반 AtlasClear Holdings "
        "(ATCH) 주가 예측 프로그램"
    ),
    theme=gr.themes.Soft()
)


# ==========================================
# 5. 실행
# ==========================================

demo.launch()
