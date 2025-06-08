# analyze_app.py
import streamlit as st
import pandas as pd
from trading_analysis.analyze import load_model_runs_df

df = load_model_runs_df()

st.title("📊 Анализ стратегии")

if df.empty:
    st.warning("Нет данных")
else:
    st.line_chart(df["pnl"].cumsum(), use_container_width=True)
    st.dataframe(df)

    st.subheader("📉 Убыточные дни")
    st.dataframe(df[(df["pnl"] < 0) & (df["winrate"] < 0.5)])
