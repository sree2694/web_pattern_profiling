from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='user_id', y='cluster', hue='cluster', data=df, palette='deep', s=100, alpha=0.7)
    plt.title('User Clusters')
    plt.xlabel('User ID')
    plt.ylabel('Cluster')
    plt.legend(title='Cluster')
    plt.show()

def run_clustering(df, num_clusters=3):
    df['sequence_str'] = df['browsing_sequence'].apply(lambda x: ' '.join(x))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['sequence_str'])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    return df

