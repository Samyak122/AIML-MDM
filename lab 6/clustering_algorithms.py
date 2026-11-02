
# IMPLEMENTATION OF CLUSTERING ALGORITHMS
# Dataset: Wine Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ---------------------------------------
# 1. LOAD DATASET
# ---------------------------------------

wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

print("Wine Dataset Loaded Successfully!")
print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])

print("\nFirst Five Records:")
print(X.head())

# ---------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData Preprocessing Completed!")
print("Features Standardized Successfully.")

# ---------------------------------------
# 3. APPLY CLUSTERING ALGORITHMS
# ---------------------------------------

results = {}

# K-MEANS CLUSTERING
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

results["K-Means"] = kmeans_labels

# HIERARCHICAL CLUSTERING
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

results["Hierarchical"] = hierarchical_labels

# DBSCAN CLUSTERING
dbscan = DBSCAN(
    eps=2.5,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_scaled)

results["DBSCAN"] = dbscan_labels

print("\nClustering Algorithms Executed Successfully!")

# ---------------------------------------
# 4. EVALUATE CLUSTERING PERFORMANCE
# ---------------------------------------

evaluation = []

for name, labels in results.items():

    unique_labels = set(labels)

    # Remove noise points for DBSCAN evaluation
    if -1 in unique_labels:
        mask = labels != -1
        eval_data = X_scaled[mask]
        eval_labels = labels[mask]
    else:
        eval_data = X_scaled
        eval_labels = labels

    n_clusters = len(set(eval_labels))

    if 1 < n_clusters < len(eval_labels):

        silhouette = silhouette_score(
            eval_data,
            eval_labels
        )

        db_index = davies_bouldin_score(
            eval_data,
            eval_labels
        )

    else:
        silhouette = np.nan
        db_index = np.nan

    evaluation.append({
        "Algorithm": name,
        "Number of Clusters": n_clusters,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index
    })

# Display results
results_df = pd.DataFrame(evaluation)

print("\n========== CLUSTERING EVALUATION ==========")
print(results_df.to_string(index=False))

# ---------------------------------------
# 5. PCA FOR VISUALIZATION
# ---------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "Principal Component 1": X_pca[:, 0],
    "Principal Component 2": X_pca[:, 1]
})

# ---------------------------------------
# 6. VISUALIZE CLUSTERS
# ---------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, labels) in zip(axes, results.items()):

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        palette="viridis",
        s=60,
        ax=ax,
        legend="full"
    )

    ax.set_title(name + " Clustering")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

# ---------------------------------------
# 7. COMPARE EVALUATION METRICS
# ---------------------------------------

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Algorithm",
    y="Silhouette Score"
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Clustering Algorithm")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Algorithm",
    y="Davies-Bouldin Index"
)

plt.title("Davies-Bouldin Index Comparison")
plt.xlabel("Clustering Algorithm")
plt.ylabel("Davies-Bouldin Index")

plt.tight_layout()
plt.show()

# ---------------------------------------
# 8. FINAL RESULT
# ---------------------------------------

valid_results = results_df.dropna(
    subset=["Silhouette Score"]
)

if not valid_results.empty:

    best_model = valid_results.loc[
        valid_results["Silhouette Score"].idxmax()
    ]

    print("\n========== FINAL RESULT ==========")

    print(
        "Best Clustering Algorithm:",
        best_model["Algorithm"]
    )

    print(
        "Best Silhouette Score:",
        round(best_model["Silhouette Score"], 4)
    )

    print(
        "Davies-Bouldin Index:",
        round(best_model["Davies-Bouldin Index"], 4)
    )

else:
    print("No valid clustering results found.")

print("\nProgram Executed Successfully!")
