# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:13:51 2025

@author: LAB
"""

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set the page config
st.set_page_config(page_title="K-Means Clustering", layout="wide")

# Create two columns with custom widths
col1, col2 = st.columns([1, 2])

# Left Column: Controls with gray background
with col1:
    st.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h4>⚙️ Configure Clustering</h4>
        """,
        unsafe_allow_html=True
    )
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)
    st.markdown("</div>", unsafe_allow_html=True)  # close gray box

# Right Column: Title and Plot
with col2:
    st.title("K-Means Clustering App with Iris Dataset")

    # Load Iris dataset
    iris = load_iris()
    X = iris.data

    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.get_cmap('tab10', k)

    for cluster in range(k):
        cluster_points = X_pca[y_kmeans == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   label=f'Cluster {cluster}', s=50, alpha=0.7, color=colors(cluster))

    ax.set_title("Clusters (2D PCA Projection)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()

    st.pyplot(fig)
