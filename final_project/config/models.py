import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .general import config as general_config

config = {
    "basic_classification": {
      "models": {
        "DecisionTree": DecisionTreeClassifier(random_state=general_config["random_seed"])
      },
    },
    "advanced_classification": {
      "models": {
        "LogisticRegression": LogisticRegression(random_state=general_config["random_seed"]),
        "DecisionTree": DecisionTreeClassifier(random_state=general_config["random_seed"]),
        "RandomForest": RandomForestClassifier(random_state=general_config["random_seed"]),
        "SVM": SVC(random_state=general_config["random_seed"], probability=True),  # Enable probability estimates for AUC-ROC
        "XGBoost": xgb.XGBClassifier(random_state=general_config["random_seed"], eval_metric='logloss'),  # Use logloss for early stopping
        "LightGBM": lgb.LGBMClassifier(random_state=general_config["random_seed"], verbose=-1)  # Suppress LightGBM warnings
      },
      "param_grids": {
        "LogisticRegression": {
            "penalty": ["l1", "l2"],
            "C": [0.001, 0.01, 0.1, 1.0, 10, 100],
            "solver": ["liblinear", "saga"],  # solvers that support l1/l2
            "max_iter": [100, 500, 1000, 2000]
        },
        "DecisionTree": {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8]
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.1, 1, 10]  
            # 'gamma' only matters if kernel='rbf'. 
            # 'linear' will ignore gamma, but it's fine for RandomizedSearchCV.
        },
        "XGBoost": {
            "learning_rate": [0.01, 0.05, 0.1, 0.3],
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 6, 7, 9],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.3, 0.5],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1, 1.5, 2]
        },
        "LightGBM": {
            "learning_rate": [0.01, 0.05, 0.1, 0.3],
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [-1, 5, 7, 10],  
            "num_leaves": [31, 40, 50, 80, 100],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [10, 20, 50]  
        }
      }
    }
}