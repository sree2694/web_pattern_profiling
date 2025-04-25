from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters(df, title='User Clusters'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='x_pca', y='y_pca', hue='cluster', data=df, palette='deep', s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title='Cluster')
    plt.show()


def run_clustering(df, num_clusters=3, method="kmeans"):
    df['sequence_str'] = df['browsing_sequence'].apply(lambda x: ' '.join(x))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['sequence_str'])

    # Dimensionality Reduction for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    df['x_pca'] = X_pca[:, 0]
    df['y_pca'] = X_pca[:, 1]

    # Choose Clustering Method
    if method == "kmeans":
        model = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = model.fit_predict(X)
        print(f"[INFO] KMeans Inertia: {model.inertia_:.2f}")
    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        df['cluster'] = model.fit_predict(X)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=num_clusters)
        df['cluster'] = model.fit_predict(X.toarray())
    else:
        raise ValueError("Unsupported clustering method. Choose from: 'kmeans', 'dbscan', 'agglomerative'")

    return df
