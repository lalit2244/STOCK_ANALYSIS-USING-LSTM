#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import ta
from ta.trend import MACD
from textblob import TextBlob
from newsapi import NewsApiClient
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("📈 Stock Price Prediction with Technical Indicators + Sentiment")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT):", "AAPL")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

data = load_data(ticker, start_date, end_date)

st.subheader(f"Raw Data for {ticker}")
st.write(data.tail())

# ---------------------- Technical Indicators ----------------------
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# ✅ Ensure 1D Series
close_series = data["Close"].squeeze()

data["RSI"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

macd_indicator = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
data["MACD"] = macd_indicator.macd()
data["MACD_Signal"] = macd_indicator.macd_signal()
data["MACD_Hist"] = macd_indicator.macd_diff()


# ---------------------- Sentiment Analysis ----------------------
newsapi = NewsApiClient(api_key="05795e3ad5904ccfa280f6804dcf7308")  # 🔑 Replace with your API key

@st.cache_data
def fetch_sentiment(ticker):
    today = datetime.today().strftime("%Y-%m-%d")
    try:
        articles = newsapi.get_everything(q=ticker, language="en", sort_by="relevancy", from_param=today)
        sentiments = []
        for article in articles["articles"]:
            analysis = TextBlob(article["title"])
            sentiments.append(analysis.sentiment.polarity)
        if len(sentiments) > 0:
            return np.mean(sentiments)
    except:
        return 0
    return 0

sentiment_today = fetch_sentiment(ticker)
last_row = data.iloc[-1].copy()
last_row["Date"] = datetime.today()
last_row["Sentiment"] = sentiment_today
data["Sentiment"] = 0
data = pd.concat([data, pd.DataFrame([last_row])], ignore_index=True)

st.write(f"📰 Today's sentiment score for {ticker}: **{sentiment_today:.3f}**")

# ---------------------- Visualization ----------------------
plot_data = data[["Date", "Close", "MA20", "MA50", "RSI", "MACD", "MACD_Signal", "MACD_Hist"]].dropna().set_index("Date")

# Price Chart
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=plot_data.index, y=plot_data["Close"], mode="lines", name="Close"))
fig_price.add_trace(go.Scatter(x=plot_data.index, y=plot_data["MA20"], mode="lines", name="MA20"))
fig_price.add_trace(go.Scatter(x=plot_data.index, y=plot_data["MA50"], mode="lines", name="MA50"))
fig_price.update_layout(title="Stock Price with Moving Averages", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_price, use_container_width=True)

# RSI Chart
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=plot_data.index, y=plot_data["RSI"], line=dict(color="blue", width=1.5), name="RSI"))
fig_rsi.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI", height=400)
st.plotly_chart(fig_rsi, use_container_width=True)

# MACD Chart
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=plot_data.index, y=plot_data["MACD"], line=dict(color="cyan", width=1.5), name="MACD"))
fig_macd.add_trace(go.Scatter(x=plot_data.index, y=plot_data["MACD_Signal"], line=dict(color="yellow", width=1.5), name="Signal Line"))
fig_macd.add_trace(go.Bar(x=plot_data.index, y=plot_data["MACD_Hist"], name="Histogram", marker_color="gray", opacity=0.5))
fig_macd.update_layout(title="MACD (Moving Average Convergence Divergence)", xaxis_title="Date", yaxis_title="MACD", height=400)
st.plotly_chart(fig_macd, use_container_width=True)

# ---------------------- LSTM Model for Prediction ----------------------
st.subheader("🔮 LSTM Stock Price Prediction")

# Prepare data
close_prices = data["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

time_step = 60
X_train, y_train, X_test, y_test = [], [], [], []

for i in range(time_step, len(train_data)):
    X_train.append(train_data[i - time_step:i, 0])
    y_train.append(train_data[i, 0])

for i in range(time_step, len(test_data)):
    X_test.append(test_data[i - time_step:i, 0])
    y_test.append(test_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot Predictions
valid = data[train_size + time_step:].copy()
valid["Predictions"] = predictions

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=valid["Date"], y=valid["Close"], mode="lines", name="Actual"))
fig_pred.add_trace(go.Scatter(x=valid["Date"], y=valid["Predictions"], mode="lines", name="Predicted"))
fig_pred.update_layout(title="LSTM Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_pred, use_container_width=True)

st.success("✅ Stock Prediction Completed Successfully!")


# In[ ]:




