import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AQI Predictor — India", page_icon="🌿", layout="wide")

# ── Helpers ──────────────────────────────────────────────────────────────────
def aqi_category(aqi):
    if aqi <= 50:    return "Good",      "#00e400"
    elif aqi <= 100: return "Satisfactory", "#92d050"
    elif aqi <= 200: return "Moderate",  "#f5d800"
    elif aqi <= 300: return "Poor",      "#ff7e00"
    elif aqi <= 400: return "Very Poor", "#ff0000"
    else:            return "Severe",    "#99004c"

feature_cols = [
    "PM2.5","PM10","NO2","CO","O3","SO2","Benzene",
    "month","day_of_week","season","quarter",
    "AQI_lag_1","AQI_lag_3","AQI_lag_7",
    "AQI_rolling_3","AQI_rolling_7","AQI_rolling_30"
]

# ── Load data & model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("city_day_clean.csv", parse_dates=["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("xgb_aqi_model.pkl")

df    = load_data()
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("AQI Predictor India")
st.sidebar.markdown("---")
cities = sorted(df["City"].unique())
city   = st.sidebar.selectbox("Select City", cities,
         index=cities.index("Delhi") if "Delhi" in cities else 0)

city_df = df[df["City"] == city].sort_values("Date").reset_index(drop=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"Air Quality Dashboard — {city}")
st.caption("Based on CPCB data 2015–2020 | Model: XGBoost")
st.markdown("---")

# ── Metric cards ──────────────────────────────────────────────────────────────
latest_aqi  = city_df["AQI"].iloc[-1]
prev_aqi    = city_df["AQI"].iloc[-2]
avg_7day    = city_df["AQI"].tail(7).mean()
avg_30day   = city_df["AQI"].tail(30).mean()
cat, color  = aqi_category(latest_aqi)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest AQI",   f"{latest_aqi:.0f}",  delta=f"{latest_aqi - prev_aqi:.0f} vs yesterday")
col2.metric("7-day Average", f"{avg_7day:.0f}")
col3.metric("30-day Average",f"{avg_30day:.0f}")
col4.metric("Category",     cat)

st.markdown("---")

# ── AQI Trend Chart ───────────────────────────────────────────────────────────
st.subheader("AQI Trend")
days = st.slider("Show last N days", 30, 365, 180, step=30)
trend = city_df.tail(days)

fig = go.Figure()
fig.add_trace(go.Scatter(x=trend["Date"], y=trend["AQI"],
    mode="lines", name="AQI", line=dict(color="#378ADD", width=2)))
fig.add_hline(y=100, line_dash="dot", line_color="orange",
    annotation_text="Moderate (100)", annotation_position="bottom right")
fig.add_hline(y=200, line_dash="dot", line_color="red",
    annotation_text="Poor (200)", annotation_position="bottom right")
fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
    template="plotly_white", xaxis_title="Date", yaxis_title="AQI")
st.plotly_chart(fig, use_container_width=True)

# ── Monthly Seasonality ───────────────────────────────────────────────────────
st.subheader("Monthly Pattern")
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = city_df.groupby("month")["AQI"].mean().reset_index()
monthly["month_name"] = [month_names[m-1] for m in monthly["month"]]

fig2 = px.bar(monthly, x="month_name", y="AQI", template="plotly_white",
              color="AQI", color_continuous_scale=["green","yellow","orange","red"],
              labels={"AQI": "Avg AQI", "month_name": "Month"})
fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
st.plotly_chart(fig2, use_container_width=True)

# ── Pollutant breakdown ───────────────────────────────────────────────────────
st.subheader("Pollutant Levels (latest available day)")
pollutants = ["PM2.5","PM10","NO2","CO","SO2","O3","Benzene"]
last_vals  = city_df[pollutants].dropna().iloc[-1]

fig3 = px.bar(x=pollutants, y=last_vals.values, template="plotly_white",
              labels={"x": "Pollutant", "y": "Value"},
              color=last_vals.values,
              color_continuous_scale=["green","yellow","red"])
fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
st.plotly_chart(fig3, use_container_width=True)

# ── Prediction Section ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Next-Day AQI Prediction")
st.caption("Adjust pollutant values to simulate tomorrow's air quality")

last_row = city_df[feature_cols].dropna().iloc[-1].to_dict()

c1, c2, c3 = st.columns(3)
last_row["PM2.5"] = c1.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(round(last_row["PM2.5"])), 1.0)
last_row["PM10"]  = c2.slider("PM10  (µg/m³)", 0.0, 500.0, float(round(last_row["PM10"])),  1.0)
last_row["NO2"]   = c3.slider("NO2   (µg/m³)", 0.0, 200.0, float(round(last_row["NO2"])),   1.0)
last_row["CO"]    = c1.slider("CO    (mg/m³)",  0.0, 50.0,  float(round(last_row["CO"],1)),  0.1)
last_row["O3"]    = c2.slider("O3    (µg/m³)", 0.0, 300.0,  float(round(last_row["O3"])),    1.0)
last_row["SO2"]   = c3.slider("SO2   (µg/m³)", 0.0, 200.0,  float(round(last_row["SO2"])),   1.0)

input_df   = pd.DataFrame([last_row])[feature_cols]
prediction = model.predict(input_df)[0]
cat_pred, col_pred = aqi_category(prediction)

st.markdown(f"""
<div style='padding:1.2rem 1.5rem; border-radius:12px;
     border-left: 5px solid {col_pred}; background:{col_pred}18; margin-top:1rem;'>
  <p style='margin:0; font-size:13px; color:gray;'>Predicted AQI for tomorrow</p>
  <p style='margin:0; font-size:36px; font-weight:600; color:{col_pred};'>
    {prediction:.0f} <span style='font-size:18px;'>— {cat_pred}</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ── City comparison ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Compare with other cities")
all_city_avg = df.groupby("City")["AQI"].mean().sort_values(ascending=False).reset_index()
all_city_avg.columns = ["City", "Avg AQI"]

fig4 = px.bar(all_city_avg, x="City", y="Avg AQI", template="plotly_white",
              color="Avg AQI", color_continuous_scale=["green","yellow","orange","red"])
city_index = list(all_city_avg["City"]).index(city)
fig4.add_vline(x=city_index, line_color="blue", line_width=2,
               annotation_text=f"← {city}", annotation_position="top")
fig4.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                   xaxis_tickangle=-45, coloraxis_showscale=False)
st.plotly_chart(fig4, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: CPCB via Kaggle | Model: XGBoost | Built with Streamlit")