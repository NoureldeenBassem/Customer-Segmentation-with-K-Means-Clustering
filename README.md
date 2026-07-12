# 🛍️ Mall Customer Segmentation — K-Means Clustering

Unsupervised learning project that segments mall customers into meaningful groups using K-Means 🎯, and deploys the trained model as an interactive Streamlit app 🚀.

## 🌐 Live App

👉 **[Mall Customer Segmentation · Streamlit](https://customer-segmentation-with-k-means-clustering-mnw3fqtkaffpckpw.streamlit.app/)**

## 📁 Project Structure

```
├── kmeans_customer_segmentation.ipynb   # 📓 Full notebook: EDA, elbow method, clustering, profiling
├── app.py                               # 🖥️ Streamlit deployment app
├── requirements.txt                     # 📦 Dependencies
├── data/
│   └── Mall_Customers.csv               # 📊 Dataset
└── models/
    ├── clustering_pipeline.pkl          # 🧠 Saved scaler + K-Means pipeline
    └── cluster_profiles.json            # 🏷️ Cluster averages + descriptive segment names
```

## ⚙️ How It Works

1. 🔍 **EDA** — inspect the dataset, check for missing values/duplicates, and visualize distributions.
2. 🎛️ **Feature Selection & Scaling** — cluster on Annual Income and Spending Score, scaled with `StandardScaler`.
3. 📉 **Elbow Method** — determine the optimal number of clusters (K = 5).
4. 🤖 **Train K-Means** — fit the final model and assign each customer to a cluster.
5. 🏷️ **Cluster Profiling & Labeling** — compute average Age/Income/Spending per cluster and convert cluster numbers into descriptive segment names (e.g. "High Income, High Spenders").
6. 💾 **Save Pipeline** — bundle scaling + K-Means into a single `Pipeline`, saved with `joblib`.
7. 🚀 **Deploy** — a Streamlit app loads the saved pipeline, takes new customer input, and predicts their labeled segment.

## ▶️ Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501` 🌐.

---

<p align="center">Built with ❤️ by <b>Noureldin Bassem</b> — Computer & AI Engineer 🤖</p>
