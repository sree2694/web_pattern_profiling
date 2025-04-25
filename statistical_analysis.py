import pandas as pd
import scipy.stats as stats
from pattern_mining import extract_ngrams
from statsmodels.stats.multitest import fdrcorrection


def demographic_analysis(df, pattern_to_check, n=2):
    grouped = df.groupby(['age_group', 'gender'])
    results = []

    for name, group in grouped:
        count = sum(pattern_to_check in extract_ngrams(seq, n) for seq in group['browsing_sequence'])
        p_value = stats.binomtest(count, len(group), p=0.1).pvalue
        results.append((name, count, len(group), p_value))

    results_df = pd.DataFrame(results, columns=['demographic', 'count', 'total', 'p_value'])
    results_df['bonferroni_p'] = results_df['p_value'] * len(results_df)
    _, fdr_corrected = fdrcorrection(results_df['p_value'])
    results_df['fdr_p'] = fdr_corrected

    return results_df
