import dice_ml
import numpy as np 
from dice_ml import Dice

def generate_counterfactuals_for_models(model_results, X_test, y_test, dice_data, 
                                        constraints={}, num_samples_to_explain=10, 
                                        total_CFs=2, max_display=2):
    """
    Generates counterfactual explanations for each trained model using DiCE.
    
    For each model in the provided model_results dictionary, this function:
      1. Creates a DiCE Model object wrapping the trained model.
      2. Instantiates a DiCE explainer (using the "random" method by default).
      3. Obtains model predictions on the test set and identifies misclassified instances.
      4. Processes up to num_samples_to_explain misclassified instances to generate counterfactuals.
         (All processed counterfactuals are stored, but detailed console output is limited.)
      5. For each processed instance, generates counterfactual explanations (total_CFs per instance)
         to flip the prediction to the desired (opposite) class.
      6. Displays enhanced print output (for up to max_display instances) including:
         - Model name and total misclassifications.
         - Instance index, original prediction, and desired class.
         - Number of counterfactuals generated compared to the requested number.
         - Detailed feature-level changes for each counterfactual.
         - A notification when the display limit is reached.
    
    Parameters
    ----------
    model_results : dict
        Dictionary of model evaluation results, with model names as keys and dicts
        containing at least the key "best_model" with the trained model instance.
    X_test : pd.DataFrame
        The test features DataFrame.
    y_test : array-like or pd.Series
        The true target labels for the test set.
    dice_data : dice_ml.Data
        The DiCE Data object built from the training data, defining the search space.
    constraints : dict, optional
        Dictionary of feature constraints (e.g., immutable features) for counterfactual generation.
    num_samples_to_explain : int, optional (default=10)
        Maximum number of misclassified instances per model for which counterfactuals are generated.
    total_CFs : int, optional (default=2)
        Number of counterfactual examples to generate for each selected instance.
    max_display : int, optional (default=2)
        Maximum number of instances for which detailed visual information is printed.
        (Counterfactuals for additional instances are still processed and stored, but not
         printed to avoid excessive output.)
    
    Returns
    -------
    all_cf_examples : dict
        Dictionary mapping each model name to a list of tuples. Each tuple is:
            (instance_index, original_prediction, desired_class, cf_object)
        containing the results for a given misclassified instance.
    """
    features_to_vary = get_features_to_vary(X_test, constraints)
    permitted_range = get_permitted_range(X_test, features_to_vary)

    all_cf_examples = {}

    # Iterate through each model in model_results
    for model_name, result_dict in model_results.items():
        model = result_dict["best_model"]
        print(f"\n{'='*60}")
        print(f"Counterfactual Generation for Model: {model_name}")
        
        # Predict on test set and compute misclassifications.
        y_pred = model.predict(X_test)
        misclassified_indices = np.where(y_test != y_pred)[0]
        print(f"Total misclassified instances for {model_name}: {len(misclassified_indices)}")
        
        # Randomly select up to num_samples_to_explain misclassified instances
        if len(misclassified_indices) >= num_samples_to_explain:
            selected_indices = np.random.choice(misclassified_indices, size=num_samples_to_explain, replace=False)
        else:
            selected_indices = misclassified_indices
        
        model_cf_examples = []  # To store counterfactuals for this model
        display_count = 0
        
        # Instantiate the DiCE explainer once for the model.
        m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
        exp = dice_ml.Dice(dice_data, m, method="random")
        
        for idx in selected_indices:
            # Always process the instance, but print details for only up to max_display instances
            instance = X_test.iloc[[idx]].copy()  # Extract instance as DataFrame
            current_prediction = y_pred[idx]
            # For binary classification, set desired_class to the opposite label.
            desired_class = 1 if current_prediction == 0 else 0
            
            # Generate counterfactuals for this instance.
            cf = exp.generate_counterfactuals(
                instance,
                total_CFs=total_CFs,
                desired_class=desired_class,
                verbose=False,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range
            )
            
            # Append counterfactual result.
            model_cf_examples.append((idx, current_prediction, desired_class, cf))
            
            # Enhanced printing only for a limited number of instances
            if display_count < max_display:
                print(f"\n{'-'*40}")
                print(f"Processing Instance Index: {idx}")
                print(f"Original Prediction: {current_prediction} | Desired Class: {desired_class}")
                
                # Enhanced printing of counterfactual details.
                if cf.cf_examples_list:
                    num_cfs_generated = len(cf.cf_examples_list[0].final_cfs_df)
                    print(f"Counterfactuals generated: {num_cfs_generated} (Requested: {total_CFs})")
                    # Output feature-level changes for each generated counterfactual.
                    for cf_idx in range(min(total_CFs, num_cfs_generated)):
                        cf_instance = cf.cf_examples_list[0].final_cfs_df.iloc[cf_idx]
                        changes = get_feature_changes(instance.iloc[0], cf_instance, features_to_vary)
                        print(f"\nCounterfactual {cf_idx + 1}:")
                        for feature, change in changes.items():
                            print(f"  {feature}: {change[0]} -> {change[1]}")
                else:
                    print("No counterfactual generated or an error occurred.")
                display_count += 1
            else:
                # Do not print detailed outputs for instances beyond max_display.
                pass
        
        all_cf_examples[model_name] = model_cf_examples
        print(f"\nFinished processing {model_name}.")
        print(f"{'='*60}\n")

    return all_cf_examples

def get_features_to_vary(X_test, constraints):
    """
    Helper function to calculate which features should be varied (all features minus immutable ones).
    
    Parameters
    ----------
    X_test : pd.DataFrame
        The test feature DataFrame.
    constraints : dict
        A dictionary of feature constraints that includes "immutable_features".
    
    Returns
    -------
    features_to_vary : list
        A list of features that should be varied (excluding immutable ones).
    """
    # Process immutable features by expanding categoricals to one-hot columns
    immutable_features = []
    for feat in constraints.get("immutable_features", []):
        # Find all matching one-hot encoded columns
        encoded_cols = [col for col in X_test.columns if col.startswith(feat + "_")]
        if encoded_cols:
            immutable_features.extend(encoded_cols)
        else:  # For continuous features
            immutable_features.append(feat)

    # Determine features to vary by subtracting immutable features from all features
    features_to_vary = list(set(X_test.columns) - set(immutable_features))
    return features_to_vary

def get_feature_changes(original_instance, cf_instance, features_to_vary):
    """
    Helper function to extract specific feature changes between the original and counterfactual instances.

    Parameters
    ----------
    original_instance : pd.Series
        The original instance.
    cf_instance : pd.Series
        The counterfactual instance.
    features_to_vary : list
        List of features that are allowed to change.

    Returns
    -------
    changes : dict
        A dictionary where keys are the changed features and values are tuples (original_value, cf_value).
    """
    changes = {}
    for feature in features_to_vary:
        if original_instance[feature] != cf_instance[feature]:
            changes[feature] = (original_instance[feature], cf_instance[feature])
    return changes

def get_permitted_range(X_test, features_to_vary):
    """
    Helper function to determine the permitted range for varying each feature.
    
    Parameters
    ----------
    X_test : pd.DataFrame
        The test feature DataFrame.
    features_to_vary : list
        List of features that are allowed to change.
    
    Returns
    -------
    permitted_range : dict
        A dictionary where keys are feature names and values are tuples (min_value, max_value).
    """
    permitted_range = {}
    for feature in features_to_vary:
        if feature in X_test.select_dtypes(include=[np.number]).columns:
            min_val = X_test[feature].min()
            max_val = X_test[feature].max()
            permitted_range[feature] = (min_val, max_val)
    return permitted_range