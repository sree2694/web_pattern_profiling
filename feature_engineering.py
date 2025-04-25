import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

# Define domain categories
DOMAIN_CATEGORIES = {
    'social': ['facebook.com', 'instagram.com', 'twitter.com'],
    'info': ['google.com', 'wikipedia.org', 'stackoverflow.com'],
    'shopping': ['amazon.com', 'reddit.com'],
    'entertainment': ['youtube.com', 'netflix.com']
}

def calculate_domain_category_counts(browsing_sequence):
    """
    Calculates counts for different domain categories.
    """
    category_counts = {'social': 0, 'info': 0, 'shopping': 0, 'entertainment': 0}
    
    for domain in browsing_sequence:
        for category, domains in DOMAIN_CATEGORIES.items():
            if domain in domains:
                category_counts[category] += 1
                break
    
    return category_counts

def calculate_sequence_entropy(browsing_sequence):
    """
    Calculate entropy for the browsing sequence to measure variety.
    Entropy will be higher if the user browses a wider range of domains.
    """
    domain_counts = {domain: browsing_sequence.count(domain) for domain in set(browsing_sequence)}
    freq_list = list(domain_counts.values())
    return entropy(freq_list)

def calculate_position_based_weights(browsing_sequence):
    """
    Calculate position-based weights for the sequence.
    Higher weight is assigned to the first and last clicks.
    """
    weights = np.zeros(len(browsing_sequence))
    weights[0] = 2  # First click gets higher weight
    weights[-1] = 2  # Last click gets higher weight
    return weights

def generate_features(dataframe):
    """
    Generate features from the browsing sequences for each user.
    """
    feature_data = []
    
    for index, row in dataframe.iterrows():
        browsing_sequence = row['browsing_sequence']
        
        # Calculate domain category counts
        category_counts = calculate_domain_category_counts(browsing_sequence)
        
        # Calculate sequence entropy
        seq_entropy = calculate_sequence_entropy(browsing_sequence)
        
        # Calculate position-based domain weights
        position_weights = calculate_position_based_weights(browsing_sequence)
        position_weighted_sum = sum(position_weights)
        
        # Create a feature vector for the user
        feature_vector = {
            'user_id': row['user_id'],
            'social_count': category_counts['social'],
            'info_count': category_counts['info'],
            'shopping_count': category_counts['shopping'],
            'entertainment_count': category_counts['entertainment'],
            'sequence_entropy': seq_entropy,
            'position_weighted_sum': position_weighted_sum
        }
        
        feature_data.append(feature_vector)
    
    feature_df = pd.DataFrame(feature_data)
    
    # Optionally, normalize/scale features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(feature_df.drop('user_id', axis=1))  # Exclude user_id for scaling
    scaled_feature_df = pd.DataFrame(scaled_features, columns=feature_df.columns[1:])
    scaled_feature_df['user_id'] = feature_df['user_id']
    
    return scaled_feature_df

# Example usage with a DataFrame containing browsing sequences
data = [
    {'user_id': 1, 'browsing_sequence': ['google.com', 'youtube.com', 'facebook.com', 'amazon.com']},
    {'user_id': 2, 'browsing_sequence': ['wikipedia.org', 'reddit.com', 'youtube.com']},
    {'user_id': 3, 'browsing_sequence': ['instagram.com', 'netflix.com', 'amazon.com']},
    {'user_id': 4, 'browsing_sequence': ['google.com', 'wikipedia.org', 'facebook.com', 'twitter.com']}
]

df = pd.DataFrame(data)

# Generate features
features_df = generate_features(df)
print(features_df)
