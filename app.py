import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


# =========================
# 1. 모델 불러오기
# =========================

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


# =========================
# 2. 사용할 변수
# =========================

features = [
    "Open","High","Low","Close",
    "Volume","Volume_MA20","Volume_Change",
    "Return_1D","MA5","MA20","MA60",
    "Volatility_20D","RSI14",
    "SPY_Return","QQQ_Return",
    "Filing_Count","Filing_8_K","Filing_10_Q",
    "Filing_10_K","Filing_S_1","Filing_S_3",
    "Filing_DEF_14A","Daily_Event_Score",
    "Revenue","Net_Income","Assets","Liabilities",
    "Stockholders_Equity","Cash","Operating_Cash_Flow",
    "Market_Cap","PS_Ratio","PB_Ratio",
    "Debt_Ratio","Cash_to_Liabilities",
    "ROA","ROE","Revenue_Growth",
    "Short_Interest","Short_Interest_Ratio",
    "Short_Volume","Short_Volume_Ratio",
    "Short_Interest_to_Volume",
    "Short_Ratio_Change",
    "Days_To_Earnings",
    "Earnings_Within_5D",
    "Earnings_Within_10D",
    "Earnings_Within_20D"
]


# =========================
# 3. ATCH 데이터 가져오기
# =========================

def get_stock_data():

    data = yf.download(
        "ATCH",
        period="1y",
        auto_adjust=False,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    data = data[
        ["Date","Open","High","Low","Close","Volume"]
    ]

    return data


# =========================
# 4. 기술적 지표 계산
# =========================

def make_features(stock):

    stock = stock.copy()

    stock["Return_1D"] = stock["Close"].pct_change()

    stock["MA5"] = (
        stock["Close"].rolling(5).mean()
    )

    stock["MA20"] = (
        stock["Close"].rolling(20).mean()
    )

    stock["MA60"] = (
        stock["Close"].rolling(60).mean()
    )

    stock["Volatility_20D"] = (
        stock["Return_1D"].rolling(20).std()
    )

    stock["Volume_MA20"] = (
        stock["Volume"].rolling(20).mean()
    )

    stock["Volume_Change"] = (
        stock["Volume"].pct_change()
    )

    delta = stock["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    stock["RSI14"] = (
        100 - (100 / (1 + rs))
    )

    # 시장 지수

    market = yf.download(
        ["SPY","QQQ"],
        period="1y",
        auto_adjust=False,
        progress=False
    )

    spy = market["Close"]["SPY"].pct_change()
    qqq = market["Close"]["QQQ"].pct_change()

    spy = spy.reset_index()
    qqq = qqq.reset_index()

    spy.columns = ["Date","SPY_Return"]
    qqq.columns = ["Date","QQQ_Return"]

    stock = stock.merge(
        spy,
        on="Date",
        how="left"
    )

    stock = stock.merge(
        qqq,
        on="Date",
        how="left"
    )

    return stock


# =========================
# 5. 예측
# =========================

def predict():

    try:

        stock = get_stock_data()

        stock = make_features(stock)

        latest = stock.iloc[-1:].copy()

        # 현재 앱에서는
        # 학습 데이터에 없는 SEC/재무/공매도 변수는
        # 저장된 학습 데이터의 최근값을 사용

        training_data = pd.read_csv(
            "ATCH_AI_final_dataset.csv"
        )

        for feature in features:

            if feature not in latest.columns:

                latest[feature] = (
                    training_data[feature].dropna().iloc[-1]
                )

        # 필요한 값이 없는 경우
        # 학습 데이터의 최근값으로 보완

        for feature in features:

            if pd.isna(
                latest[feature].iloc[0]
            ):

                latest.loc[
                    latest.index[0],
                    feature
                ] = training_data[
                    feature
                ].dropna().iloc[-1]

        X = latest[features].astype(float)

        # 상승 확률

        X_direction = direction_scaler.transform(X)

        probability = direction_model.predict(
            X_direction,
            verbose=0
        )[0][0]

        # 예상 수익률

        X_return = return_scaler.transform(X)

        expected_return = return_model.predict(
            X_return,
            verbose=0
        )[0][0]

        probability_percent = probability * 100

        return (
            f"### ATCH AI 예측 결과\n\n"
            f"**다음 거래일 상승 확률:** "
            f"{probability_percent:.2f}%\n\n"
            f"**예상 수익률:** "
            f"{expected_return * 100:.2f}%"
        )

    except Exception as e:

        return f"오류가 발생했습니다:\n\n{str(e)}"


# =========================
# 6. Gradio 화면
# =========================

with gr.Blocks(
    title="ATCH AI Stock Predictor"
) as demo:

    gr.Markdown(
        """
        # 📈 ATCH AI Stock Predictor

        공개된 주가·시장·재무·공시·이벤트 데이터를
        이용하여 인공신경망(ANN)으로 ATCH의
        다음 거래일 움직임을 예측합니다.

        ⚠️ 본 프로그램의 결과는 학습된 모델의 예측값이며
        투자 권유가 아닙니다.
        """
    )

    predict_button = gr.Button(
        "🤖 AI 예측 실행"
    )

    result = gr.Markdown(
        "버튼을 눌러 예측을 시작하세요."
    )

    predict_button.click(
        fn=predict,
        outputs=result
    )


demo.launch()
