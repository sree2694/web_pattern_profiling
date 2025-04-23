import pandas as pd

def extract_features(df):
    """
    Extract advanced features from browsing sequences.
    """
    def domain_count(sequence):
        return len(sequence)

    def unique_domains(sequence):
        return len(set(sequence))

    def domain_repeats(sequence):
        return domain_count(sequence) - unique_domains(sequence)

    df['sequence_length'] = df['browsing_sequence'].apply(domain_count)
    df['unique_domains'] = df['browsing_sequence'].apply(unique_domains)
    df['repeats'] = df['browsing_sequence'].apply(domain_repeats)
    
    # More advanced features can be added here later
    return df
