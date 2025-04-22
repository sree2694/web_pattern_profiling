import os
import pandas as pd
from datetime import datetime
from pathlib import Path

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_downloads_folder():
    """
    Return the path to the user's Downloads folder.
    """
    return str(Path.home() / "Downloads")

def save_patterns_to_csv(pattern_counts, filename="patterns"):
    timestamp = get_timestamp()
    downloads_folder = get_downloads_folder()
    full_path = os.path.join(downloads_folder, f"{filename}_{timestamp}.csv")

    patterns, counts = zip(*pattern_counts.most_common(10))
    pattern_df = pd.DataFrame({
        'Pattern': patterns,
        'Frequency': counts
    })
    pattern_df.to_csv(full_path, index=False)
    print(f"[✔] Patterns saved to: {full_path}")

def save_demographic_analysis_to_csv(df, filename="demographic_analysis"):
    timestamp = get_timestamp()
    downloads_folder = get_downloads_folder()
    full_path = os.path.join(downloads_folder, f"{filename}_{timestamp}.csv")
    
    df.to_csv(full_path, index=False)
    print(f"[✔] Demographic analysis saved to: {full_path}")

def save_cluster_results_to_csv(df, filename="cluster_results"):
    timestamp = get_timestamp()
    downloads_folder = get_downloads_folder()
    full_path = os.path.join(downloads_folder, f"{filename}_{timestamp}.csv")

    df.to_csv(full_path, index=False)
    print(f"[✔] Cluster results saved to: {full_path}")
