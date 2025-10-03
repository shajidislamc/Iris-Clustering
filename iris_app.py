import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ------------------------------
# Load Iris dataset
# ------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y_true = iris.target

# Scale the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Iris Flower Clustering", layout="centered")
st.title("🌸 Iris Flower Clustering App")
st.write("Choose a clustering method and explore results interactively!")

# --- User Inputs ---
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# --- Dropdown for Clustering Method ---
method = st.selectbox("Select Clustering Method", ["KMeans", "Hierarchical", "DBSCAN"])

# --- Run Clustering ---
if st.button("🔍 Run Clustering"):
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    if method == "KMeans":
        model = KMeans(n_clusters=3, random_state=42, n_init="auto")
        cluster_labels = model.fit_predict(X_scaled)
        df["cluster"] = cluster_labels
        cluster = model.predict(scaler.transform(input_data))[0]
        st.success(f"🌼 This flower belongs to **Cluster {cluster}** (using {method})")
    
    elif method == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
        cluster_labels = model.fit_predict(X_scaled)
        df["cluster"] = cluster_labels
        cluster = None  # Cannot assign cluster to a single new sample
        st.info("🌼 Cannot assign cluster to a single flower in Hierarchical Clustering. See plots instead.")
    
    elif method == "DBSCAN":
        model = DBSCAN(eps=1.0, min_samples=5)
        cluster_labels = model.fit_predict(X_scaled)
        df["cluster"] = cluster_labels
        cluster = None  # DBSCAN does not have predict for new points
        st.info("🌼 DBSCAN cannot predict a single new sample. See plots instead.")

    # --- Show ARI Scores ---
    st.subheader("📈 Clustering Performance (Adjusted Rand Index)")
    ari_kmeans = adjusted_rand_score(y_true, KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(X_scaled))
    ari_hier = adjusted_rand_score(y_true, AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled))
    ari_dbscan = adjusted_rand_score(y_true, DBSCAN(eps=1.0, min_samples=5).fit_predict(X_scaled))

    ari_df = pd.DataFrame({
        "Method": ["KMeans", "Hierarchical", "DBSCAN"],
        "ARI Score": [ari_kmeans, ari_hier, ari_dbscan]
    })
    st.table(ari_df)

    fig_ari, ax = plt.subplots()
    sns.barplot(x="Method", y="ARI Score", data=ari_df, palette="viridis", ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig_ari)

    # --- Tabs for Visualization ---
    tab1, tab2, tab3 = st.tabs(["📊 2D Scatter", "🌐 3D Scatter", "🌳 Dendrogram"])

    # 2D Scatter
    with tab1:
        st.subheader("2D Scatter Plot (Petal length vs Petal width)")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df, x="petal length (cm)", y="petal width (cm)",
            hue="cluster", palette="Set1", s=70, ax=ax
        )
        ax.scatter(petal_length, petal_width, color="black", s=120, marker="X", label="Your Flower")
        ax.legend()
        st.pyplot(fig)

    # 3D Scatter
    with tab2:
        st.subheader("3D Cluster Visualization")
        fig3d = px.scatter_3d(
            df,
            x="sepal length (cm)",
            y="sepal width (cm)",
            z="petal length (cm)",
            color="cluster",
            symbol="cluster",
            opacity=0.7
        )
        fig3d.add_scatter3d(
            x=[sepal_length],
            y=[sepal_width],
            z=[petal_length],
            mode="markers",
            marker=dict(size=8, color="black", symbol="x"),
            name="Your Flower"
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # Dendrogram (only for Hierarchical)
    with tab3:
        if method == "Hierarchical":
            st.subheader("Hierarchical Clustering Dendrogram")
            linked = linkage(X_scaled, method="ward")
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, truncate_mode="level", p=5, ax=ax)
            st.pyplot(fig)
        else:
            st.info("🌳 Dendrogram is only available for Hierarchical Clustering.")
