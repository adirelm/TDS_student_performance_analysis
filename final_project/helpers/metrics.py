import pandas as pd

from .utils import *

def compute_cf_metrics(all_cf_examples, X_test, model_results, numeric_columns, scaler, target_feature):
    """
    Computes counterfactual explanation quality metrics for each model.

    For each model in `all_cf_examples`, this function calculates:
      - **Validity (%)**: The percentage of counterfactuals that yield the desired prediction.
      - **Average Sparsity**: The average number of features changed in each counterfactual.
      - **Average Proximity (L1)**: The average L1 (Manhattan) distance, in original units, between 
        the original instance and its counterfactual(s).
      - **Average Proximity (L2)**: The average L2 (Euclidean) distance, in original units, between 
        the original instance and its counterfactual(s).

    The function processes each counterfactual example by:
      1. Constructing a combined DataFrame of the original instance and its counterfactuals.
      2. Dropping the target column (if present) for metric computation.
      3. Computing the number of features changed (sparsity) and converting the differences back 
         to their original scale using the provided StandardScaler.
      4. Accumulating the metrics over all counterfactual examples per model.

    Parameters
    ----------
    all_cf_examples : dict
        Dictionary mapping model names to lists of counterfactual examples.
        Each example is a tuple: (idx, orig_pred, desired_class, cf_object), where:
          - idx: The index of the original instance in X_test.
          - orig_pred: The model's original prediction.
          - desired_class: The desired (target) prediction for the counterfactual.
          - cf_object: The DiCE counterfactual object containing the generated counterfactuals.
    X_test : pd.DataFrame
        The test dataset containing the original instances.
    model_results : dict
        Dictionary containing model evaluation results, including the best model instance for each model.
    numeric_columns : list of str
        List of column names corresponding to numeric features (used for reverting scaled differences).
    scaler : StandardScaler
        The fitted StandardScaler used during preprocessing; needed to convert scaled differences back 
        to their original units.
    target_feature : str
        The name of the target feature column (e.g., 'income'); this column is dropped from the DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame (indexed by model name) with the following columns:
            - validity_pct: Percentage of counterfactuals that achieve the desired prediction.
            - avg_sparsity: Average number of features changed per counterfactual.
            - avg_l1: Average L1 distance (in original units) between the original and counterfactual instances.
            - avg_l2: Average L2 distance (in original units) between the original and counterfactual instances.
    """
    from collections import defaultdict

    # Initialize a dictionary to accumulate metric sums for each model.
    metrics_dict = defaultdict(lambda: {
        "valid_cf_count": 0,
        "total_cf_count": 0,
        "sum_sparsity": 0.0,
        "sum_proximity_l1": 0.0,
        "sum_proximity_l2": 0.0
    })
    
    # Iterate over each model and its associated counterfactual examples.
    for model_name, cf_list in all_cf_examples.items():
        model = model_results[model_name]["best_model"]
        
        for (idx, orig_pred, desired_class, cf_object) in cf_list:
            # 1) Build the combined DataFrame of the original instance and its counterfactuals.
            combined_df = build_original_and_cfs_df(idx, cf_object, X_test)

            # 2) Drop the target feature column (if present) to prevent it from affecting the metric calculations.
            if target_feature in combined_df.columns:
                combined_df = combined_df.drop(columns=target_feature)

            # Get the original instance in scaled form.
            original_scaled = combined_df.loc["Original"].copy()

            # Identify the counterfactual rows.
            cf_rows = [row for row in combined_df.index if row.startswith("CF_")]

            for cf_idx in cf_rows:
                cf_scaled = combined_df.loc[cf_idx]

                # VALIDITY: Check if the counterfactual's prediction matches the desired class.
                pred = model.predict([cf_scaled.values])[0]
                metrics_dict[model_name]["total_cf_count"] += 1
                if pred == desired_class:
                    metrics_dict[model_name]["valid_cf_count"] += 1

                # SPARSITY: Count the number of features that were changed.
                diff_scaled = (cf_scaled - original_scaled).abs()
                sparsity = (diff_scaled > 1e-6).sum()
                metrics_dict[model_name]["sum_sparsity"] += sparsity

                # Proximity: Convert differences back to original scale for L1 and L2 calculations.
                diff_unscaled = unscale_diff(diff_scaled, scaler, numeric_columns)
                l1_distance = diff_unscaled.abs().sum()
                l2_distance = (diff_unscaled ** 2).sum() ** 0.5

                metrics_dict[model_name]["sum_proximity_l1"] += l1_distance
                metrics_dict[model_name]["sum_proximity_l2"] += l2_distance
    
    # Construct the final DataFrame of metrics.
    rows = []
    for model_name, stats in metrics_dict.items():
        total_cfs = stats["total_cf_count"]
        if total_cfs == 0:
            validity = 0
            avg_sparsity = 0
            avg_l1 = 0
            avg_l2 = 0
        else:
            validity = 100.0 * stats["valid_cf_count"] / total_cfs
            avg_sparsity = stats["sum_sparsity"] / total_cfs
            avg_l1 = stats["sum_proximity_l1"] / total_cfs
            avg_l2 = stats["sum_proximity_l2"] / total_cfs
        
        rows.append({
            "model": model_name,
            "validity_pct": validity,
            "avg_sparsity": avg_sparsity,
            "avg_l1": avg_l1,
            "avg_l2": avg_l2
        })

    return pd.DataFrame(rows).set_index("model")