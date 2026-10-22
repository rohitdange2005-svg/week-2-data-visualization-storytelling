"""Week 2: Advanced Data Visualization and Storytelling with Python
Install: pip install pandas numpy matplotlib scikit-learn
Run: python week2_advanced_visualization.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = load_wine(as_frame=True)
df = dataset.frame.copy().rename(columns={"target": "wine_class"})
class_names = {i: n for i, n in enumerate(dataset.target_names)}
df["wine_class_name"] = df["wine_class"].map(class_names)
feature_cols = list(dataset.feature_names)
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.drop_duplicates().dropna(subset=["wine_class"]).copy()
for col in feature_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

counts = df["wine_class_name"].value_counts().reindex(dataset.target_names)
counts.plot(kind="bar", title="Wine Class Distribution")
plt.xlabel("Wine class"); plt.ylabel("Samples"); plt.tight_layout(); plt.show()

df.boxplot(column="alcohol", by="wine_class_name")
plt.title("Alcohol by Wine Class"); plt.suptitle("")
plt.xlabel("Wine class"); plt.ylabel("Alcohol measurement"); plt.tight_layout(); plt.show()

for cl in dataset.target_names:
    s = df[df["wine_class_name"] == cl]
    plt.scatter(s["flavanoids"], s["color_intensity"], label=cl, alpha=0.8)
plt.title("Flavanoids vs Color Intensity")
plt.xlabel("Flavanoids"); plt.ylabel("Color intensity"); plt.legend(); plt.tight_layout(); plt.show()

corr = df[feature_cols].corr()
plt.figure(figsize=(10, 8))
im = plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, label="Correlation coefficient")
plt.xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=7)
plt.yticks(range(len(feature_cols)), feature_cols, fontsize=7)
plt.title("Correlation Heatmap"); plt.tight_layout(); plt.show()

X = StandardScaler().fit_transform(df[feature_cols])
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
pca_df["wine_class_name"] = df["wine_class_name"].values
for cl in dataset.target_names:
    s = pca_df[pca_df["wine_class_name"] == cl]
    plt.scatter(s["PC1"], s["PC2"], label=cl, alpha=0.8)
plt.title("PCA View of Wine Samples")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(); plt.tight_layout(); plt.show()

means = df.groupby("wine_class_name")[["alcohol", "flavanoids", "color_intensity", "proline"]].mean()
means_scaled = (means - means.min()) / (means.max() - means.min())
means_scaled.T.plot(kind="bar", title="Relative Feature Means by Wine Class")
plt.ylabel("Relative mean (0–1)"); plt.tight_layout(); plt.show()
