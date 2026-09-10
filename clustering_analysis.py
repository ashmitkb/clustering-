"""
Clustering Assignment: PCA + DBSCAN vs KMeans on the Wine dataset
==================================================================

Dataset: UCI Wine Recognition dataset (via sklearn.datasets.load_wine)
- 178 samples, 13 numeric chemical-analysis features, 3 real cultivar classes.
- Features are on very different scales (e.g. "proline" ~ hundreds/thousands,
  "hue" ~ 0-2), and several features are highly correlated (e.g. flavanoids,
  total_phenols, OD280/OD315). This makes it a good candidate for
  dimensionality reduction before density/distance-based clustering.

Pipeline:
1. Load + standardize data (StandardScaler) -- required before PCA/DBSCAN
   because both are distance-based and sensitive to feature scale.
2. Apply PCA, decide how many components to keep using the explained
   variance ratio / scree plot, and project to 2D for clustering + plotting.
3. Use a k-distance plot to choose a reasonable eps for DBSCAN.
4. Run DBSCAN and KMeans (KMeans is the second clustering method; k chosen
   via the elbow method and silhouette score, cross-checked against the
   3 known cultivars).
5. Evaluate both clustering results using:
   - Silhouette Score        (higher is better, intrinsic)
   - Davies-Bouldin Index    (lower is better, intrinsic)
   - Calinski-Harabasz Index (higher is better, intrinsic)
   - Adjusted Rand Index & Normalized Mutual Information vs the true
     cultivar labels (extrinsic -- available for this dataset, used only
     as an extra sanity check, not as the primary criterion since true
     clustering evaluation is normally unsupervised).
6. Save all figures to ./outputs and print a results summary.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & standardize
# ---------------------------------------------------------------------------
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = data.target  # ground-truth cultivar, used only for extrinsic checks

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Classes (ground truth, not used for clustering): {np.unique(y_true)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 2. PCA - decide number of components via explained variance
# ---------------------------------------------------------------------------
pca_full = PCA(n_components=X.shape[1], random_state=RANDOM_STATE)
pca_full.fit(X_scaled)
evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(evr) + 1), evr, alpha=0.6, label="Individual")
plt.step(range(1, len(evr) + 1), cum_evr, where="mid", color="crimson", label="Cumulative")
plt.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot - Wine Dataset (13 features)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_pca_scree_plot.png", dpi=150)
plt.close()

n_components_90 = int(np.argmax(cum_evr >= 0.90) + 1)
print(f"\nComponents needed for >=90% variance: {n_components_90}")
print(f"Variance explained by first 2 PCs: {cum_evr[1]*100:.1f}%")

# We reduce to 2 components: enough for >55% variance individually captured
# by the first two axes, and it lets us cluster + visualize in 2D, which is
# what DBSCAN's eps/min_samples and cluster shapes are easiest to reason
# about in. (A 90%-variance solution needs ~8-9 PCs -- also computed above
# and reported in the write-up, but 2D is used for the main clustering run
# so that density-based DBSCAN behaves predictably and results are plottable.)
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca_2d.fit_transform(X_scaled)
print(f"PC1 variance ratio: {pca_2d.explained_variance_ratio_[0]:.3f}")
print(f"PC2 variance ratio: {pca_2d.explained_variance_ratio_[1]:.3f}")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="viridis", s=50, edgecolor="k", alpha=0.8)
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Wine Dataset Projected to 2D via PCA\n(colored by true cultivar, for reference only)")
plt.legend(*scatter.legend_elements(), title="Cultivar")
plt.tight_layout()
plt.savefig(f"{OUT}/02_pca_projection_true_labels.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Choose eps for DBSCAN via k-distance (k = min_samples) plot
# ---------------------------------------------------------------------------
min_samples = 5
nn = NearestNeighbors(n_neighbors=min_samples)
nn.fit(X_pca)
distances, _ = nn.kneighbors(X_pca)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 5))
plt.plot(k_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"Distance to {min_samples}th nearest neighbor")
plt.title("k-Distance Plot for DBSCAN eps Selection")
plt.tight_layout()
plt.savefig(f"{OUT}/03_dbscan_k_distance_plot.png", dpi=150)
plt.close()

# Elbow of the k-distance curve chosen by inspection of the plot / knee point.
# For this PCA-2D wine data this lands around eps ~ 0.6-0.7.
eps_value = 0.65
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
db_labels = dbscan.fit_predict(X_pca)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print(f"\nDBSCAN (eps={eps_value}, min_samples={min_samples}): "
      f"{n_clusters_db} clusters, {n_noise} noise points")

# ---------------------------------------------------------------------------
# 4. KMeans - choose k via elbow + silhouette
# ---------------------------------------------------------------------------
inertias, sil_scores, k_range = [], [], range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels_k))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(list(k_range), inertias, "o-", color="steelblue", label="Inertia (elbow)")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(list(k_range), sil_scores, "s-", color="darkorange", label="Silhouette")
ax2.set_ylabel("Silhouette Score", color="darkorange")
plt.title("KMeans Model Selection: Elbow + Silhouette")
fig.tight_layout()
plt.savefig(f"{OUT}/04_kmeans_elbow_silhouette.png", dpi=150)
plt.close()

best_k = 3  # elbow flattens at k=3 and matches the 3 known cultivars
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
km_labels = kmeans.fit_predict(X_pca)
print(f"KMeans (k={best_k}) fitted.")

# ---------------------------------------------------------------------------
# 5. Visual comparison of clustering results
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sc0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap="tab10", s=50, edgecolor="k", alpha=0.85)
axes[0].set_title(f"DBSCAN (eps={eps_value}, min_samples={min_samples})\n"
                   f"{n_clusters_db} clusters + {n_noise} noise points")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

sc1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="tab10", s=50, edgecolor="k", alpha=0.85)
axes[1].set_title(f"KMeans (k={best_k})")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

plt.suptitle("Clustering Results on PCA-Reduced Wine Data", y=1.03, fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUT}/05_clustering_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
def safe_eval(labels, name):
    mask = labels != -1  # exclude DBSCAN noise from intrinsic metrics
    n_clusters_found = len(set(labels[mask]))
    if n_clusters_found < 2:
        return {"method": name, "silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "ari": np.nan, "nmi": np.nan}
    sil = silhouette_score(X_pca[mask], labels[mask])
    db_idx = davies_bouldin_score(X_pca[mask], labels[mask])
    ch_idx = calinski_harabasz_score(X_pca[mask], labels[mask])
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return {"method": name, "silhouette": sil, "davies_bouldin": db_idx,
            "calinski_harabasz": ch_idx, "ari": ari, "nmi": nmi}

results = pd.DataFrame([
    safe_eval(db_labels, f"DBSCAN (eps={eps_value}, min_samples={min_samples})"),
    safe_eval(km_labels, f"KMeans (k={best_k})"),
]).set_index("method")

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print("\n===================== EVALUATION SUMMARY =====================")
print(results.to_string())
print("================================================================")
results.to_csv(f"{OUT}/evaluation_results.csv")

# Bar chart comparison of the two intrinsic metrics that matter most
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
results["silhouette"].plot(kind="bar", ax=axes[0], color=["#4c72b0", "#dd8452"])
axes[0].set_title("Silhouette Score (higher = better)")
axes[0].set_ylabel("Score")
axes[0].tick_params(axis="x", rotation=20)

results["davies_bouldin"].plot(kind="bar", ax=axes[1], color=["#4c72b0", "#dd8452"])
axes[1].set_title("Davies-Bouldin Index (lower = better)")
axes[1].set_ylabel("Score")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(f"{OUT}/06_metric_comparison.png", dpi=150)
plt.close()

print(f"\nAll figures and results saved to ./{OUT}/")
