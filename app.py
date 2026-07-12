import json

import joblib
import matplotlib.pyplot as plt
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
    return pipeline, profiles, data


pipeline, profiles, data = load_artifacts()

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

fig, ax = plt.subplots(figsize=(7, 6))
for cluster_id, group in data.groupby("Cluster"):
    ax.scatter(
        group["Annual Income (k$)"],
        group["Spending Score (1-100)"],
        s=45,
        alpha=0.75,
        color=PALETTE[cluster_id % len(PALETTE)],
        label=profiles[cluster_id]["Segment_Name"],
    )

if predicted_cluster is not None:
    ax.scatter(
        income,
        spending,
        marker="*",
        s=500,
        color=PALETTE[predicted_cluster % len(PALETTE)],
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
        label="New Customer (Predicted)",
    )

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments" + (" — New Customer Highlighted" if predicted_cluster is not None else ""))
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
fig.tight_layout()

st.pyplot(fig)

st.divider()
st.caption("Note: Age is collected for context, but the clustering model itself groups customers using Annual Income and Spending Score.")
