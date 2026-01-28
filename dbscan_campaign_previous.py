# DBSCAN Campaign + Previous - JAVÍTOTT VERZIÓ
# Fix: Epsilon = 0.0 hiba kezelése

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ... [Előző kódok: adatok betöltése, feature selection] ...

# Példa feature selection:
# features = ['campaign', 'previous']
# X = df[features].copy()
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# MinPts értékek
minpts_values = [4, 10, 20]

# k-distance számítás
k_distances_dict = {}
for min_pts in minpts_values:
    neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    k_distances = distances[:, -1]
    k_distances_dict[min_pts] = k_distances

# Balance score függvények (ugyanaz mint előbb)
def calculate_gini_coefficient(cluster_sizes):
    if len(cluster_sizes) == 0:
        return 1.0
    cluster_sizes = np.array(sorted(cluster_sizes))
    n = len(cluster_sizes)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * cluster_sizes)) / (n * np.sum(cluster_sizes)) - (n + 1) / n
    return gini

def calculate_balance_score(labels):
    cluster_labels = labels[labels != -1]
    if len(cluster_labels) == 0:
        return 0.0
    
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    n_clusters = len(unique_labels)
    
    if n_clusters < 3:
        cluster_count_score = 0.3
    elif 3 <= n_clusters <= 8:
        cluster_count_score = 1.0
    elif 9 <= n_clusters <= 20:
        cluster_count_score = 0.6
    else:
        cluster_count_score = 0.1
    
    gini = calculate_gini_coefficient(counts)
    gini_score = 1 - gini
    
    largest_ratio = counts.max() / len(cluster_labels)
    
    if 0.3 <= largest_ratio <= 0.7:
        largest_score = 1.0
    elif 0.2 <= largest_ratio < 0.3:
        largest_score = 0.7
    elif largest_ratio > 0.8:
        largest_score = 0.1
    else:
        largest_score = 0.4
    
    balance_score = (0.5 * cluster_count_score + 
                    0.3 * largest_score + 
                    0.2 * gini_score)
    
    return balance_score

# ===== JAVÍTOTT PARAMÉTER TESZTELÉS =====
print("\n" + "="*100)
print("DBSCAN SZÉLES KÖRŰ PARAMÉTER TESZTELÉS (JAVÍTOTT)")
print("="*100)

all_results = []

for min_pts in minpts_values:
    k_distances = k_distances_dict[min_pts]
    
    # Epsilon értékek kiszámítása
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]
    epsilon_values = [np.percentile(k_distances, p) for p in percentiles]
    
    # ===== KRITIKUS FIX: Epsilon validálás =====
    # Ha az epsilon 0.0 vagy túl kicsi, használj minimum értéket
    MIN_EPSILON = 1e-6  # Nagyon kicsi pozitív szám
    
    epsilon_values_fixed = []
    percentiles_fixed = []
    
    for eps, pct in zip(epsilon_values, percentiles):
        if eps <= MIN_EPSILON:
            print(f"⚠ Kihagyva: MinPts={min_pts}, {pct}. percentilis, ε={eps:.6f} (túl kicsi)")
            continue  # Kihagyjuk ezt az epsilon értéket
        else:
            epsilon_values_fixed.append(eps)
            percentiles_fixed.append(pct)
    
    if len(epsilon_values_fixed) == 0:
        print(f"⚠ MinPts={min_pts}: NINCS ÉRVÉNYES EPSILON! Átugorva.")
        continue
    # ==========================================
    
    print(f"\n{'='*100}")
    print(f"MinPts = {min_pts}, Érvényes epsilon értékek: {len(epsilon_values_fixed)}")
    print(f"{'='*100}")
    
    for eps, pct in zip(epsilon_values_fixed, percentiles_fixed):
        # DBSCAN futtatása (most már eps > 0 garantált)
        dbscan = DBSCAN(eps=eps, min_samples=min_pts)
        labels = dbscan.fit_predict(X_scaled)
        
        # Statisztikák
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels) * 100
        n_core = len(dbscan.core_sample_indices_)
        
        # Klaszter-eloszlás elemzés
        cluster_labels = labels[labels != -1]
        if len(cluster_labels) > 0:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            largest_cluster_pct = counts.max() / len(cluster_labels) * 100 if len(counts) > 0 else 0
        else:
            largest_cluster_pct = 0
        
        # Validációs metrikák
        if n_clusters >= 2 and noise_ratio < 50:
            mask = labels != -1
            if mask.sum() > 1:
                try:
                    silhouette = silhouette_score(X_scaled[mask], labels[mask])
                    davies_bouldin = davies_bouldin_score(X_scaled[mask], labels[mask])
                    calinski = calinski_harabasz_score(X_scaled[mask], labels[mask])
                except:
                    silhouette = davies_bouldin = calinski = np.nan
            else:
                silhouette = davies_bouldin = calinski = np.nan
        else:
            silhouette = davies_bouldin = calinski = np.nan
        
        # Balance score
        balance_score = calculate_balance_score(labels)
        
        # Combined score
        if not np.isnan(silhouette) and balance_score > 0:
            silhouette_normalized = (silhouette + 1) / 2
            combined_score = 0.3 * silhouette_normalized + 0.7 * balance_score
        else:
            combined_score = 0
        
        all_results.append({
            'min_pts': min_pts,
            'epsilon': eps,
            'percentile': pct,
            'n_clusters': n_clusters,
            'n_core': n_core,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'largest_cluster_pct': largest_cluster_pct,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'balance_score': balance_score,
            'combined_score': combined_score,
            'labels': labels,
            'dbscan': dbscan
        })
        
        # Kimenet
        status = "✓" if noise_ratio <= 20 and 20 <= largest_cluster_pct < 85 and 2 <= n_clusters <= 20 else "⚠"
        silh_str = f"{silhouette:.3f}" if not np.isnan(silhouette) else "  N/A"
        print(f"ε={eps:.4f} ({pct:>2}%): Klaszter={n_clusters:3d}, Noise={noise_ratio:5.1f}%, "
              f"Max={largest_cluster_pct:5.1f}%, Balance={balance_score:.3f}, "
              f"Silh={silh_str:>5s}, Combined={combined_score:.3f} {status}")

print(f"\n✓ Összesen {len(all_results)} konfiguráció tesztelve!")

# ===== ALTERNATÍV MEGOLDÁS: Ha túl sok 0.0 epsilon van =====
# Használj log-transzformációt a változókon:
#
# X['campaign_log'] = np.log1p(X['campaign'])  # log1p(x) = log(1+x)
# X['previous_log'] = np.log1p(X['previous'])
# X_scaled = scaler.fit_transform(X[['campaign_log', 'previous_log']])
#
# Ez "szétnyújtja" a 0 körüli értékeket
# ===================================================
