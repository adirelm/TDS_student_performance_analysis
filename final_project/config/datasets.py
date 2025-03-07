"""
Configuration for datasets used in the pipeline.

This configuration dictionary defines settings for each dataset that the pipeline will use.
Each dataset entry contains the following keys:
  - train_file: Path to the training data file.
  - test_file: Path to the testing data file (if available). For datasets provided as a single file, 
               this can be the same as train_file.
  - columns: A manually defined list of column names to use for loading the dataset.
  - target_column: The name of the target variable (i.e., the column to be predicted).
  - missing_values: A string representing how missing values are denoted in the dataset.
  - skip_first_row_train: Boolean indicating whether to skip the first row in the train file (e.g., if it contains an extra header).
  - skip_first_row_test: Boolean indicating whether to skip the first row in the test file (e.g., if it contains an extra header).
"""

config = {
    "adult_dataset": {
        "train_file": "data/adult/adult.data",
        "test_file": "data/adult/adult.test",
        "columns": [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ],
        "target_column": "income",
        "missing_values": "?",
        "skip_first_row_test": True,
        "header": None,
        "delim_whitespace": False
    },
    "wdbc_dataset": {
        "train_file": "data/wdbc/wdbc.data",
        "test_file": "data/wdbc/wdbc.data",
        "columns": [
            "ID", "Diagnosis",
            "radius1", "texture1", "perimeter1", "area1", "smoothness1",
            "compactness1", "concavity1", "concave_points1", "symmetry1",
            "fractal_dimension1",
            "radius2", "texture2", "perimeter2", "area2", "smoothness2",
            "compactness2", "concavity2", "concave_points2", "symmetry2",
            "fractal_dimension2",
            "radius3", "texture3", "perimeter3", "area3", "smoothness3",
            "compactness3", "concavity3", "concave_points3", "symmetry3",
            "fractal_dimension3"
        ],
        "target_column": "Diagnosis",
        "missing_values": "",
        "skip_first_row_test": False,
        "header": None,
        "delim_whitespace": False
    },
        "credit_dataset": {
            "train_file": "data/statlog/german.data",
            "test_file": "data/statlog/german.data",
            "columns": [
                "status",            # Attribute 1: Checking account status (categorical: A11, A12, A13, A14)
                "duration",          # Attribute 2: Duration in months (numeric)
                "credit_history",    # Attribute 3: Credit history (categorical: A30, A31, A32, A33, A34)
                "purpose",           # Attribute 4: Purpose (categorical: A40, A41, …, A410)
                "credit_amount",     # Attribute 5: Credit amount (numeric)
                "savings",           # Attribute 6: Savings account/bonds (categorical: A61, A62, A63, A64, A65)
                "employment",        # Attribute 7: Present employment since (categorical: A71, A72, A73, A74, A75)
                "installment_rate",  # Attribute 8: Installment rate in % of disposable income (numeric)
                "personal_status_sex",  # Attribute 9: Personal status and sex (categorical: A91, A92, A93, A94, A95)
                "other_debtors",     # Attribute 10: Other debtors/guarantors (categorical: A101, A102, A103)
                "residence",         # Attribute 11: Present residence since (numeric)
                "property",          # Attribute 12: Property (categorical: A121, A122, A123, A124)
                "age",               # Attribute 13: Age in years (numeric)
                "other_installment_plans",  # Attribute 14: Other installment plans (categorical: A141, A142, A143)
                "housing",           # Attribute 15: Housing (categorical: A151, A152, A153)
                "credits",           # Attribute 16: Number of existing credits at this bank (numeric)
                "job",               # Attribute 17: Job (categorical: A171, A172, A173, A174)
                "people_liable",     # Attribute 18: Number of people being liable to provide maintenance for (numeric)
                "telephone",         # Attribute 19: Telephone (categorical: A191, A192)
                "foreign_worker",    # Attribute 20: Foreign worker (categorical: A201, A202)
                "class"              # Target: 1 = Good, 2 = Bad
            ],
            "target_column": "class",
            "missing_values": "",
            "skip_first_row_test": False,
            "header": None,
            "delim_whitespace": True
    },
    "red_wine_dataset": {
        "train_file": "data/wine/winequality-red.csv",
        "test_file": "data/wine/winequality-red.csv",
        "columns": [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "quality"
        ],
        "target_column": "quality_class",
        "missing_values": "",
        "skip_first_row_test": False,
        "skip_first_row_train": True,
        "header": None,
        "delim_whitespace": False,
        "sep": ";"
    },
    "white_wine_dataset": {
        "train_file": "data/wine/winequality-white.csv",
        "test_file": "data/wine/winequality-white.csv",
        "columns": [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "quality"
        ],
        "target_column": "quality_class",
        "missing_values": "",
        "skip_first_row_test": False,
        "skip_first_row_train": True,
        "header": None,
        "delim_whitespace": False,
        "sep": ";"
    }
}
