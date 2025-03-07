from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config.general import config as general_config

def train_tune_evaluate_advanced(models, param_grids, X_train, y_train, X_test, y_test, scoring='accuracy', default_cv=5, n_jobs=-1):
    """
    Trains, tunes, and evaluates multiple models.

    Args:
        models (dict): Dictionary of model instances.
        param_grids (dict): Dictionary of hyperparameter grids.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        scoring (str or callable): Scoring metric for evaluation (default is 'accuracy').
        default_cv (int): Default number of cross-validation folds.
        n_jobs (int): Number of CPU cores to use (-1 for all).

    Returns:
        dict: Dictionary containing best models, best parameters, and evaluation results.
    """
    results = {}

    for name, model in models.items():
        print(f"Training and tuning {name}...")

        search = RandomizedSearchCV(
            model,
            param_grids[name],
            scoring=scoring,
            cv=default_cv,
            n_jobs=n_jobs,
            random_state=general_config['random_seed'],
            n_iter=10 
        )
        search.fit(X_train, y_train)

        # Store best model and parameters
        best_model = search.best_estimator_
        best_params = search.best_params_

        # Evaluate on the test set
        y_pred = best_model.predict(X_test)
        
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)[:, 1]
        else:  # Use decision_function for models without predict_proba
            y_prob = best_model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "best_model": best_model,
            "best_params": best_params,
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        print(f"{name} - Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}")

    return results

def train_evaluate_baseline(models, X_train, y_train, X_test, y_test, scoring='accuracy'):
    """
    Trains and evaluates multiple models using their default hyperparameters (no hyperparameter tuning).
    
    Args:
        models (dict): Dictionary of model instances.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        scoring (str or callable): Scoring metric for evaluation (default is 'accuracy').
    
    Returns:
        dict: Dictionary containing evaluation results for each model, including accuracy, f1_score, and roc_auc.
    """
    results = {}

    for name, model in models.items():
        print(f"Training {name} with default hyperparameters...")

        # Train the model using default hyperparameters
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]  # For AUC-ROC
        else:  # Use decision_function for models that don't have predict_proba
            y_prob = model.decision_function(X_test)

        # Evaluate on the test set
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        print(f"{name} - Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}")

    return results