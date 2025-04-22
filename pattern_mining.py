from collections import Counter

def extract_ngrams(sequence, n=2):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def count_ngrams(browsing_sequences, n=2):
    """
    Count the frequency of n-grams (sequences of n contiguous items) in the browsing sequences.
    """
    ngram_counter = Counter()
    for sequence in browsing_sequences:
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngram_counter[ngram] += 1
    return ngram_counter
