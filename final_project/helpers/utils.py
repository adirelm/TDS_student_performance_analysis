import pandas as pd

def unscale_diff(diff_series, scaler, numeric_columns):
    """
    Reverts scaled differences for numeric features back to their original units.

    This function takes a pandas Series representing the differences between a counterfactual
    instance and the original instance (computed in the scaled feature space) and converts the
    differences for numeric features back to their original scale using the corresponding standard
    deviations stored in the provided StandardScaler. Differences for non-numeric features (such as
    one-hot encoded columns) are left unchanged.

    Parameters
    ----------
    diff_series : pd.Series
        A Series containing the scaled differences between a counterfactual and an original instance.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted StandardScaler object used to scale the data. It holds the standard deviations for
        the numeric features.
    numeric_columns : list of str
        A list of column names that represent numeric features (and were scaled).

    Returns
    -------
    pd.Series
        A Series with the numeric feature differences converted back to their original units.
        Non-numeric feature differences remain unchanged.
    """
    # Create a copy of the diff_series to avoid modifying the original data
    unscaled_diff = diff_series.copy()

    # Create a dictionary mapping numeric column names to their corresponding index in scaler.scale_
    col_to_idx = {col: i for i, col in enumerate(numeric_columns)}

    # Iterate through the columns in the diff_series
    for col in diff_series.index:
        # If the column is numeric (present in col_to_idx)
        if col in col_to_idx:
            idx = col_to_idx[col]
            # Unscale the difference by multiplying it with the column's original standard deviation
            unscaled_diff[col] = diff_series[col] * scaler.scale_[idx]
        # Otherwise, it's a one-hot encoded or categorical feature; leave it unchanged.
    return unscaled_diff

def build_original_and_cfs_df(idx, cf_object, X_test):
    """
    Constructs a combined DataFrame for comparing an original instance with its counterfactual examples.

    This function extracts the original instance from the test dataset using the provided index,
    and retrieves the generated counterfactual instances from the DiCE explainer object. It then
    concatenates these into a single DataFrame, where the original instance is labeled as "Original"
    and the counterfactuals are labeled as "CF_1", "CF_2", etc.

    Parameters
    ----------
    idx : int
        The index of the original instance in the X_test DataFrame.
    cf_object : dice_ml.explainer_interfaces.explainer_base.ExplainerBase
        The DiCE explainer object containing the counterfactual examples.
    X_test : pd.DataFrame
        The test dataset DataFrame from which the original instance is extracted.

    Returns
    -------
    pd.DataFrame
        A DataFrame with:
          - A row labeled "Original" representing the original instance.
          - Rows labeled "CF_1", "CF_2", ... representing each counterfactual instance.
    """
    # Extract the original instance from X_test using the provided index.
    original_instance = X_test.iloc[idx].copy()
    # Create a DataFrame for the original instance, labeling the index as "Original".
    original_df = pd.DataFrame([original_instance], index=["Original"], columns=X_test.columns)

    # Extract the counterfactual instances from the DiCE cf_object.
    cf_rows = cf_object.cf_examples_list[0].final_cfs_df
    # Relabel the counterfactual instance indices as "CF_1", "CF_2", etc.
    cf_rows.index = [f"CF_{i+1}" for i in range(cf_rows.shape[0])]

    # Concatenate the original instance DataFrame and the counterfactuals DataFrame vertically.
    combined_df = pd.concat([original_df, cf_rows], axis=0)
    return combined_df

def load_dataset(train_file, columns, test_columns=None, test_file=None, target_column=None,
                 missing_values=" ?", skip_first_row_test=False, skip_first_row_train=False, header=None,
                 delim_whitespace=False, sep=None):  
    """
    Loads and prepares a dataset for analysis.

    This function reads the training data file (and optionally a separate test file) and uses
    the manually provided list of column names (via columns_override). If no separate test file is
    provided (or if test_file == train_file), only one file is loaded.

    Parameters
    ----------
    train_file : str
        Path to the training data file.
    test_file : str or None, optional
        Path to the testing data file. If None or if train_file equals test_file, only the training file is loaded.
    columns: list
        A list of column names to use for the dataset.
    test_columns: list or None, optional
        A list of column names to use for the test dataset. If None, the training columns are used.
    target_column : str or None, optional
        The name of the target variable.
    missing_values : str, optional
        A string indicating how missing values are represented. Default is " ?".
    skip_first_row_test : bool, optional
        If True, skips the first row in the test file (useful if the test file contains an extra header row).
    skip_first_row_train : bool, optional
        If True, skips the first row in the train file (useful if the train file contains an extra header row).
    header : int or None, optional
        The row number to use as the header (0-indexed). If None, the default behavior is used.
    delim_whitespace : bool, optional
        If True, use whitespace as the delimiter. Default is False.
    sep : str, optional
        Delimiter to use. If sep is None, the default is ',' for CSV files and whitespace if delim_whitespace=True.

    Returns
    -------
    train_data : pd.DataFrame
        The training data (or the entire dataset if no test file is provided).
    test_data : pd.DataFrame or None
        The testing data if a separate file is provided; otherwise, None.
    columns : list or None
        The list of column names used.
    """

    # Check if a separate test file is provided.
    if test_file is None or test_file == train_file:
        try:
            read_csv_params = {
                "filepath_or_buffer": train_file,
                "header": header,
                "names": columns,
                "na_values": missing_values,
                "skipinitialspace": True,
                "skiprows": 1 if skip_first_row_train else 0
            }
            if delim_whitespace:
                read_csv_params["delim_whitespace"] = True
            elif sep is not None:
                read_csv_params["sep"] = sep

            data = pd.read_csv(**read_csv_params)

        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            return None, None, None
        return data, None, columns
    else:
        try:
            train_read_csv_params = {
                "filepath_or_buffer": train_file,
                "header": header,
                "names": columns,
                "na_values": missing_values,
                "skipinitialspace": True,
                "skiprows": 1 if skip_first_row_train else 0
            }
            if delim_whitespace:
                train_read_csv_params["delim_whitespace"] = True
            elif sep is not None:
                train_read_csv_params["sep"] = sep

            test_read_csv_params = {
                "filepath_or_buffer": test_file,
                "header": header,
                "names": test_columns if test_columns else columns,
                "na_values": missing_values,
                "skipinitialspace": True,
                "skiprows": 1 if skip_first_row_test else 0
            }
            if delim_whitespace:
                test_read_csv_params["delim_whitespace"] = True
            elif sep is not None:
                test_read_csv_params["sep"] = sep

            train_data = pd.read_csv(**train_read_csv_params)
            test_data = pd.read_csv(**test_read_csv_params)

        except FileNotFoundError as e:
            print(f"Error loading data files: {e}")
            return None, None, None
        return train_data, test_data, columns


