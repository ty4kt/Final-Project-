# import pandas as pd

# def sample_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Returns a new DataFrame containing only the first 1000 rows of the input DataFrame.
#     If the input DataFrame has fewer than 1000 rows, it returns the DataFrame as-is.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         The original DataFrame to reduce.

#     Returns:
#     --------
#     pd.DataFrame
#         A DataFrame containing at most the first 1000 rows.
#     """
#     return df.head(1000)

import pandas as pd

def sample_df(df: pd.DataFrame, n_samples=1000, random_state=42) -> pd.DataFrame:
    """
    Returns a DataFrame with a stratified sample of n_samples rows, ensuring both classes of 'is_fraud' are included.
    If the input DataFrame has fewer than n_samples rows, it returns the DataFrame as-is.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The original DataFrame to sample.
    n_samples : int, optional
        Number of rows to sample (default: 1000).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns:
    --------
    pd.DataFrame
        A sampled DataFrame with at most n_samples rows, preserving class distribution.
    """
    if 'is_fraud' not in df.columns:
        raise ValueError("DataFrame must contain an 'is_fraud' column for stratified sampling.")
    
    if len(df) <= n_samples:
        return df.copy()
    
    # Perform stratified sampling to preserve class distribution
    try:
        sampled_df = df.groupby('is_fraud', group_keys=False).apply(
            lambda x: x.sample(frac=n_samples/len(df), random_state=random_state)
        )
        # Ensure at least one sample from each class if possible
        if len(sampled_df['is_fraud'].unique()) < 2 and len(df['is_fraud'].unique()) == 2:
            # Add one fraud case if missing
            fraud_cases = df[df['is_fraud'] == 1]
            if not fraud_cases.empty:
                sampled_df = pd.concat([sampled_df, fraud_cases.sample(1, random_state=random_state)])
        return sampled_df.reset_index(drop=True)
    except ValueError as e:
        # Fallback to random sampling if stratified fails (e.g., too few fraud cases)
        print(f"Stratified sampling failed: {e}. Using random sampling.")
        return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)