import matplotlib.pyplot as plt
import seaborn as sns
from data_simulation import generate_fake_data
from pattern_mining import count_ngrams
from statistical_analysis import demographic_analysis
from clustering import run_clustering, visualize_clusters
import pandas as pd
import logging
from utils import save_cluster_results_to_csv, save_demographic_analysis_to_csv, save_patterns_to_csv


# Visualization for Most Common Patterns
def plot_most_common_patterns(pattern_counts):
    patterns, counts = zip(*pattern_counts.most_common(10))
    pattern_labels = [f"{p[0]} -> {p[1]}" for p in patterns]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=pattern_labels, palette='viridis')
    plt.title('Most Common Navigation Patterns')
    plt.xlabel('Frequency')
    plt.ylabel('Pattern')
    plt.show()

# Visualization for Pattern Correlation
def plot_correlation_heatmap(df):
    patterns = df['browsing_sequence'].apply(lambda seq: set(seq))
    all_patterns = [item for sublist in patterns for item in sublist]
    unique_patterns = list(set(all_patterns))

    pattern_matrix = pd.DataFrame(0, index=df.index, columns=unique_patterns)

    for i, row in df.iterrows():
        for pattern in row['browsing_sequence']:
            pattern_matrix.at[i, pattern] = 1

    corr = pattern_matrix.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Pattern Correlation Heatmap')
    plt.show()

# Visualization for Most Common Patterns Heatmap
def plot_pattern_heatmap(ngram_counts):
    patterns, counts = zip(*ngram_counts.most_common(10))  # Top 10 patterns
    pattern_labels = [f"{p[0]} -> {p[1]}" for p in patterns]
    pattern_matrix = pd.DataFrame(counts, index=pattern_labels, columns=['Frequency'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(pattern_matrix.T, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Most Common Navigation Patterns Heatmap')
    plt.xlabel('Pattern')
    plt.ylabel('Frequency')
    plt.show()

# Visualization for Cluster Features Heatmap
def plot_cluster_heatmap(df_clustered):
    df_features = df_clustered[['age_group', 'gender', 'cluster']]
    df_features['age_group'] = df_features['age_group'].astype('category').cat.codes
    df_features['gender'] = df_features['gender'].astype('category').cat.codes

    plt.figure(figsize=(10, 6))
    sns.clustermap(df_features.corr(), annot=True, cmap='coolwarm')
    plt.title('Cluster Feature Heatmap')
    plt.show()

    # Set up logging
    logging.basicConfig(filename='web_pattern_profiling.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example of logging messages
    logging.info("Starting the web pattern profiling script.")
    logging.error("Error occurred during the clustering step.", exc_info=True)

# Main function
def main():
    # Generate Data
    df = generate_fake_data(100)
    print("Sample user data:\n", df.head())

    # Mine Patterns
    ngram_counts = count_ngrams(df['browsing_sequence'], n=2)
    print("\nMost common patterns:")
    print(ngram_counts.most_common(5))

    # Save Patterns to CSV
    save_patterns_to_csv(ngram_counts)

    # Visualize Patterns
    plot_most_common_patterns(ngram_counts)
    plot_pattern_heatmap(ngram_counts)

    # Analyze Demographics
    pattern_to_check = ('google.com', 'youtube.com')
    analysis_results = demographic_analysis(df, pattern_to_check, n=2)
    print("\nDemographic pattern significance:")
    print(analysis_results)

    # Save Demographic Results to CSV
    save_demographic_analysis_to_csv(analysis_results)

    # Cluster Users
    df_clustered = run_clustering(df)
    print("\nClustered user profiles:")
    print(df_clustered[['user_id', 'age_group', 'gender', 'browsing_sequence', 'cluster']].head())
    
    # Save Cluster Results to CSV
    save_cluster_results_to_csv(df_clustered)

    # Visualize Clusters
    visualize_clusters(df_clustered)
    plot_cluster_heatmap(df_clustered)
    plot_correlation_heatmap(df)

if __name__ == "__main__":
    main()
