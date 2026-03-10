import numpy as np
from scipy.special import logit
import pandas as pd

def get_percentage_fields(csv_path = 'metab_dictionary.csv'):
    metab_data_dict = pd.read_csv(csv_path)

    mask = metab_data_dict['Type'].str.contains('percentage', case=False, na=False)
    perc_df = metab_data_dict[mask].copy()

    percentage_fields = set(
        perc_df['UKB Field ID']
        .dropna()
        .astype(int)
        .map(lambda x: f"p{x}")
    )

    return percentage_fields

PERCENTAGE_FIELDS = get_percentage_fields()


def find_percentage_indices(metab_df, col_dict=PERCENTAGE_FIELDS):
    """
    Heuristic to identify percentage columns based on column names.
    This is a simple approach and may need adjustments based on actual data.
    
    Args:
        df: Pandas DataFrame of raw data.
        col_dict: Dictionary of UKB percentage fields.
    
    Returns:
        List of column indices that are percentages.
    """
    percentage_cols = [c for c in metab_df.columns if c in col_dict]
    other_cols = [c for c in metab_df.columns if c not in percentage_cols]
   
    # Get the integer indices of the percentage columns relative to the 
    # metabolomics block passed to the ColumnTransformer
    percentage_indices = [metab_df.columns.get_loc(c) for c in percentage_cols]

    return percentage_indices

def metabolomic_transforms(X, percentage_indices=None, epsilon=1e-6):
    """
    Apply the transformations to the metabolomics data with a FunctionTransformer in the pipeline.
    Replicates the Nature paper preprocessing: https://www.nature.com/articles/s41597-023-01949-y#Sec25
    1. Percentages: Handle 0/100 bounds -> Logit Transform.
    2. Others: Handle 0 bounds -> Log Transform.
    
    Args:
        X: numpy array
        percentage_indices: List of column indices that are percentages .
        epsilon: Small offset to handle 0 and 100% values.
    
    Returns:
        Numpy array of transformed data ready for next step of pipeline.
    """
    # Step 1: work on copy
    X = np.array(X).copy()

    # Step 2: Logit for percentages
    if percentage_indices is not None:
        for idx in percentage_indices:
            # If 0-100, convert to 0-1
            if np.nanmax(X[:, idx]) > 1.0:
                X[:, idx] = X[:, idx] / 100.0
            # Clip to [epsilon, 1-epsilon]
            X[:, idx] = np.clip(X[:, idx], epsilon, 1.0 - epsilon)
            # Logit transform
            X[:, idx] = logit(X[:, idx])

    # Step 3: Log for others
    other_indices = [i for i in range(X.shape[1]) if i not in percentage_indices]
    for idx in other_indices:
        # Replace exact 0s with epsilon
        X[:, idx] = np.where(X[:, idx] == 0, epsilon, X[:, idx])
        # Natural Log Transform
        X[:, idx] = np.log(X[:, idx])

    return X

def metab_prepro_func(X, percentage_indices=None):
    return metabolomic_transforms(X, percentage_indices=percentage_indices)

def clipper_func(x):
    return np.clip(x, -10, 10)

def binarize_cci(X):
    return (X >= 2).astype(int)