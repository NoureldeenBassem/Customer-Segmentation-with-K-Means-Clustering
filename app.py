import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="centered")

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


# ---- Load the saved pipeline, cluster profiles, and training data ----
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("clustering_pipeline.pkl")
    with open("cluster_profiles.json") as f:
        profiles = {p["Cluster"]: p for p in json.load(f)}
    data = pd.read_csv("Mall_Customers.csv")
    data["Cluster"] = pipeline.predict(data[FEATURES])
    data["Segment"] = data["Cluster"].map(lambda c: profiles[c]["Segment_Name"])
    return pipeline, profiles, data


pipeline, profiles, data = load_artifacts()
segment_names = [profiles[c]["Segment_Name"] for c in sorted(profiles)]

st.title("🛍️ Mall Customer Segmentation")
st.write(
    "Enter a customer's details to predict which spending segment they belong to. "
    "The model was trained with K-Means clustering on the Mall Customers dataset."
)

st.subheader("Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=300, value=60, step=1)
spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

predicted_cluster = None
if st.button("Predict Segment"):
    # Same feature set and column names used during training
    new_customer = pd.DataFrame({
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending],
    })

    predicted_cluster = int(pipeline.predict(new_customer)[0])
    segment = profiles[predicted_cluster]

    st.success(f"Predicted Segment: **{segment['Segment_Name']}**")

    st.subheader("Segment Profile")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Age", segment["Age"])
    col2.metric("Avg. Income (k$)", segment["Annual Income (k$)"])
    col3.metric("Avg. Spending Score", segment["Spending Score (1-100)"])

    st.caption(
        f"This segment represents {segment['Count']} customers in the training data "
        f"with similar income and spending behavior."
    )

# ---- Cluster visualization ----
st.subheader("Customer Segments")

base_chart = (
    alt.Chart(data)
    .mark_circle(size=90, opacity=0.75)
    .encode(
        x=alt.X("Annual Income (k$):Q"),
        y=alt.Y("Spending Score (1-100):Q"),
        color=alt.Color(
            "Segment:N",
            scale=alt.Scale(domain=segment_names, range=PALETTE),
            legend=alt.Legend(title="Segment"),
        ),
        tooltip=["Segment:N", "Annual Income (k$):Q", "Spending Score (1-100):Q"],
    )
)

chart = base_chart

if predicted_cluster is not None:
    new_point_df = pd.DataFrame({
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending],
        "Segment": ["New Customer (Predicted)"],
    })
    new_point_chart = (
        alt.Chart(new_point_df)
        .mark_point(shape="diamond", size=500, filled=True, color="black", stroke="white", strokeWidth=2)
        .encode(
            x="Annual Income (k$):Q",
            y="Spending Score (1-100):Q",
            tooltip=["Segment:N", "Annual Income (k$):Q", "Spending Score (1-100):Q"],
        )
    )
    chart = base_chart + new_point_chart

st.altair_chart(
    chart.properties(width=600, height=450, title="Customer Segments"),
    use_container_width=True,
)
if predicted_cluster is not None:
    st.caption("The black diamond marker shows where the new customer falls among the existing segments.")

st.divider()
st.caption("Note: Age is collected for context, but the clustering model itself groups customers using Annual Income and Spending Score.")

st.markdown(
    "<p style='text-align: center; color: gray;'>Built by <b>Noureldin Bassem</b> — Computer & AI Engineer</p>",
    unsafe_allow_html=True,
)
