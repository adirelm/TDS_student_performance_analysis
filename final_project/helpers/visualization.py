import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from IPython.display import HTML, display

from .utils import * 

def interactive_counterfactual_visualization(combined_df, scaler, numeric_columns, top_n=5, section_title="Counterfactual Visualizations"):
    """
    Generates interactive counterfactual visualizations using Plotly for the top 'top_n'
    features with the largest scaled differences, and displays an accompanying table
    of the unscaled differences.
    
    Parameters:
      combined_df (pd.DataFrame): DataFrame with rows labeled "Original" and "CF_*" for 
                                  the original and counterfactual instances.
      scaler (StandardScaler): Fitted scaler used to revert scaled differences.
      numeric_columns (pd.Index): Numeric feature names (pandas Index) in the same order as used by scaler.
      top_n (int, optional): Number of top features to display. Defaults to 5.
      section_title (str, optional): Title for the visualization section.
    
    Returns:
      None. Displays interactive visuals.
    """
    original = combined_df.loc["Original"]
    html_sections = []
    cf_rows = [row for row in combined_df.index if row.startswith("CF_")]
    
    for cf_idx in cf_rows:
        cf_row = combined_df.loc[cf_idx]
        # Compute differences in scaled space
        diff_scaled = cf_row - original
        abs_diff = diff_scaled.abs()
        top_features = abs_diff.nlargest(top_n).index
        
        # Create interactive horizontal bar chart for scaled differences
        top_diff = diff_scaled[top_features]
        fig = go.Figure(go.Bar(
            x=top_diff.values,
            y=top_features,
            orientation='h',
            marker_color='skyblue'
        ))
        fig.update_layout(
            title=f"{cf_idx} vs. Original (Scaled Differences)",
            xaxis_title="Difference (Scaled Units)",
            yaxis_title="Feature",
            margin=dict(l=120, r=20, t=60, b=40)
        )
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Convert scaled differences back to original units for numeric features.
        unscaled_changes = {}
        for feature in top_features:
            if feature in numeric_columns:
                feat_idx = numeric_columns.get_loc(feature)
                orig_unscaled = original[feature] * scaler.scale_[feat_idx] + scaler.mean_[feat_idx]
                cf_unscaled = cf_row[feature] * scaler.scale_[feat_idx] + scaler.mean_[feat_idx]
            else:
                # For non-numeric or one-hot encoded features, just use the current value.
                orig_unscaled = original[feature]
                cf_unscaled = cf_row[feature]
            unscaled_changes[feature] = (orig_unscaled, cf_unscaled)
        
        # Build HTML table of unscaled changes
        changes_data = []
        for feature, (orig_val, cf_val) in unscaled_changes.items():
            changes_data.append({
                "Feature": feature,
                "Original": np.round(orig_val, 3) if isinstance(orig_val, (int, float)) else orig_val,
                "Counterfactual": np.round(cf_val, 3) if isinstance(cf_val, (int, float)) else cf_val,
                "Difference": np.round(cf_val - orig_val, 3) if isinstance(cf_val, (int, float)) and isinstance(orig_val, (int, float)) else "N/A"
            })
        df_changes = pd.DataFrame(changes_data)
        table_html = df_changes.to_html(index=False, border=1, justify="center")
        
        html_sections.append(f"<h4>{cf_idx}</h4>{fig_html}{table_html}<hr>")
    
    full_html = f"<h3>{section_title}</h3>" + "".join(html_sections)
    display(HTML(full_html))

def plot_cf_bars_scaled_and_table_unscaled_collapsible(
    combined_df,  
    scaler,       
    numeric_columns,  
    top_n=5,          
    section_title="Counterfactual Visualizations"  
):
    """
    Generates horizontal bar charts and tables of unscaled differences for each counterfactual instance,
    and displays them in a collapsible HTML <details> section.

    For each counterfactual instance in the provided DataFrame, this function performs the following steps:
    
    1. **Bar Chart (Scaled Differences):**
       - Calculates the difference between the counterfactual and the original instance in the scaled space.
       - Identifies the top `top_n` features with the largest absolute differences.
       - Generates a horizontal bar chart showing these differences (in scaled units).
       - Converts the chart to a base64-encoded PNG image for embedding into HTML.
    
    2. **Table (Unscaled Differences):**
       - Converts the top `top_n` scaled differences back to their original units using the provided scaler.
       - Constructs an HTML table displaying both the scaled and unscaled differences for these features.
    
    3. **Collapsible HTML Output:**
       - Combines the bar chart image and the table into an HTML snippet.
       - Wraps all snippets within a `<details>` element, making the visualizations collapsible.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        A DataFrame with rows labeled "Original", "CF_1", "CF_2", ... representing the original instance 
        and its counterfactual instances.
    scaler : StandardScaler
        The fitted StandardScaler object used to scale the data; required to revert scaled differences to original units.
    numeric_columns : list of str
        A list of column names that correspond to numeric features.
    top_n : int, default 5
        The number of top features (with the largest absolute differences) to display.
    section_title : str, default "Counterfactual Visualizations"
        The title to be displayed above the collapsible HTML section.

    Returns
    -------
    None
        The function displays the generated HTML output directly in the Jupyter Notebook.

    Notes
    -----
    This visualization bridges the gap between the model's scaled feature space and the original, interpretable units.
    It offers a combined view (chart and table) for each counterfactual instance, thereby enhancing the interpretability
    of the counterfactual explanations.
    """
    # Extract the original instance row.
    original = combined_df.loc["Original"]
    # Identify the indices for counterfactual rows.
    cf_rows = [r for r in combined_df.index if r.startswith("CF_")]

    html_blocks = []  # Initialize an empty list to accumulate HTML snippets for each counterfactual instance.

    for cf_idx in cf_rows:  # Iterate through each counterfactual instance.
        cf_row = combined_df.loc[cf_idx]  # Extract the current counterfactual row.
        diff_scaled = cf_row - original  # Calculate the scaled difference between the counterfactual and the original instance.
        abs_diff_scaled = diff_scaled.abs()  # Take the absolute value of the scaled differences.

        # Identify the top N features with the largest absolute scaled differences.
        top_features = abs_diff_scaled.nlargest(top_n).index
        top_diff_scaled = diff_scaled[top_features]

        # ====== 1) Generate the bar chart in memory ======
        fig, ax = plt.subplots(figsize=(7, 5))  # Create a new figure and axes for the bar chart.
        top_diff_scaled.sort_values().plot(kind='barh', color='skyblue', ax=ax)  # Plot the top N scaled differences.
        ax.set_xlabel("Change in Scaled Units (CF - Original)")  # Set the x-axis label.
        ax.set_title(f"{cf_idx} vs. Original: Top {top_n} (Scaled)")  # Set the chart title.
        ax.invert_yaxis()  # Invert the y-axis to show the largest changes at the top.
        plt.tight_layout()  # Adjust the layout to prevent labels from being cut off.

        # Convert the Matplotlib figure to a base64 encoded PNG image.
        buf = io.BytesIO()  # Create an in-memory bytes buffer.
        plt.savefig(buf, format='png', dpi=100, bbox_inches="tight")  # Save the figure as a PNG image.
        buf.seek(0)  # Rewind the buffer's file pointer to the beginning.
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')  # Encode the image data.
        plt.close(fig)  # Close the figure to free up memory.

        # ====== 2) Create the unscaled differences table ======
        # Convert the top N scaled differences back to their original units.
        diff_unscaled = unscale_diff(top_diff_scaled, scaler, numeric_columns)
        # Create a DataFrame to display the scaled and unscaled differences.
        changes_df = pd.DataFrame({
            "Scaled Diff": top_diff_scaled,
            "Unscaled Diff": diff_unscaled
        })

        table_html = changes_df.to_html(border=1, justify="center")  # Convert the DataFrame to an HTML table.

        # Build an HTML snippet combining the base64 encoded image and the HTML table.
        snippet = f"""
        <h4>{cf_idx}</h4>
        <p><img src="data:image/png;base64,{img_b64}" /></p>
        <p>{table_html}</p>
        <hr/>
        """
        html_blocks.append(snippet)  # Append the snippet to the list.

    full_content = "".join(html_blocks)  # Concatenate all HTML snippets.
    # Wrap the content in a collapsible <details> tag.
    details_html = f"""
    <h3>{section_title}</h3>
    <details>
      <summary>Click to expand/collapse</summary>
      {full_content}
    </details>
    """
    display(HTML(details_html))  # Display the final HTML output in the notebook.


def visualize_counterfactual_examples(all_cf_examples, X_test, scaler, numeric_columns, target_feature, max_instances_to_show=2):
    """
    Iterates over counterfactual examples for each model and displays visualizations.

    This function performs the following steps for each model in the provided dictionary:
    
    1. **Model Header:**  
       - Prints the model name along with a header indicating that both scaled and unscaled differences will be shown.
       
    2. **Example Iteration:**  
       - Iterates through the counterfactual examples (up to a maximum specified by `max_instances_to_show`) for that model.
       - For each example, it prints details about the misclassified instance, including its index, the original prediction, 
         and the desired class.
       
    3. **Visualization:**  
       - Calls `build_original_and_cfs_df` to create a combined DataFrame that contains the original instance along with its counterfactuals.
       - Invokes `plot_cf_bars_scaled_and_table_unscaled_collapsible` to generate and display a collapsible visualization, which 
         includes a horizontal bar chart of the top `top_n` scaled differences and an HTML table of the corresponding unscaled differences.

    Parameters
    ----------
    all_cf_examples : dict
        A dictionary containing counterfactual examples for each model. Each key is a model name, and the corresponding value is a list 
        of tuples in the form (idx, orig_pred, desired_class, cf_object), where:
            - idx is the index of the original instance in X_test.
            - orig_pred is the original prediction for that instance.
            - desired_class is the class we aim to achieve through counterfactual alteration.
            - cf_object is the DiCE counterfactual object for that instance.
    X_test : pd.DataFrame
        The test dataset DataFrame, used to extract the original instance for comparison.
    scaler : StandardScaler
        The fitted StandardScaler object used to scale the data; necessary for reverting scaled differences to their original units.
    numeric_columns : list of str
        A list of column names that represent numeric features in the dataset.
    target_feature : str
        The name of the target feature column (e.g., 'income'); this column is excluded from the visualizations.
    max_instances_to_show : int, optional (default=2)
        The maximum number of counterfactual examples to visualize per model, to prevent an excessive number of plots.
        Defaults to 2.

    Returns
    -------
    None
        The function prints detailed information and displays the generated visualizations directly in the notebook.
    """
    # Iterate over each model and its associated counterfactual examples.
    for model_name, examples in all_cf_examples.items():
        print(f"\n=== {model_name} (Scaled vs. Unscaled Differences) ===")
        
        # Iterate through the counterfactual examples for the current model,
        # up to the specified maximum number of instances.
        for i, (idx, orig_pred, desired_class, cf_object) in enumerate(examples):
            if i >= max_instances_to_show:
                break  # Stop visualizing after reaching the limit

            # Print key information about the misclassified instance.
            print(f"\nMisclassified Instance Index: {idx}")
            print(f"Original Prediction: {orig_pred}, Desired Class: {desired_class}")

            # Build a combined DataFrame containing the original instance and its counterfactuals.
            combined_df = build_original_and_cfs_df(idx, cf_object, X_test)

            # Generate and display the visualization (bar chart and table) for the current counterfactual.
            plot_cf_bars_scaled_and_table_unscaled_collapsible(
                combined_df=combined_df,
                scaler=scaler,
                numeric_columns=numeric_columns,
                top_n=5,
                section_title="CF Visuals for This Instance"
            )
            
            interactive_counterfactual_visualization(combined_df, scaler, numeric_columns, top_n=5, section_title=f"Interactive CF Visualizations for Instance {idx}")

def sanity_check_and_print_diffs(
    idx,  
    cf_object,  
    X_test,  
    model,  
    scaler,  
    numeric_columns, 
    target_feature,  
    top_n=5,  
):
    """
    Performs a sanity check on generated counterfactuals by comparing predictions and feature differences.

    This function executes the following steps:
      1. Constructs a combined DataFrame containing the original instance and its counterfactual(s)
         using `build_original_and_cfs_df`.
      2. Drops the target feature column (if present) to prevent it from affecting the metric calculations.
      3. Retrieves and prints the model's prediction for the original instance.
      4. For each counterfactual instance, it:
         - Prints the model's prediction.
         - Computes the differences (both scaled and unscaled) between the counterfactual and the original instance.
         - Identifies and displays the top `top_n` features with the largest absolute differences.
    
    Parameters
    ----------
    idx : int
        The index of the original instance in X_test.
    cf_object : dice_ml.explainer_interfaces.explainer_base.ExplainerBase
        The DiCE explainer object containing the generated counterfactuals.
    X_test : pd.DataFrame
        The test dataset DataFrame from which the original instance is extracted.
    model : sklearn model
        The trained machine learning model used to generate predictions.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted StandardScaler object used for scaling, required to revert differences to original units.
    numeric_columns : list of str
        A list of column names corresponding to numeric features.
    target_feature : str
        The name of the target feature column (e.g., 'income'); this column is dropped from the DataFrame.
    top_n : int, optional (default=5)
        The number of top feature changes (by absolute difference) to display for each counterfactual.

    Returns
    -------
    None
        The function prints prediction details and displays DataFrames of the top feature differences.
    """
    # Build a combined DataFrame that includes the original instance and its counterfactual examples.
    combined_df = build_original_and_cfs_df(idx, cf_object, X_test)

    # Remove the target feature column if it is present.
    if target_feature in combined_df.columns:
        combined_df.drop(columns=[target_feature], inplace=True)

    # Retrieve the original (scaled) instance and obtain its model prediction.
    original_scaled = combined_df.loc["Original"].copy()
    original_pred = model.predict([original_scaled.values])[0]

    print(f"Sanity check for instance {idx}:")
    print(f"  Original => Predicted: {original_pred}")

    # Process each counterfactual instance in the combined DataFrame.
    cf_rows = [row for row in combined_df.index if row.startswith("CF_")]
    for cf_idx in cf_rows:
        cf_scaled = combined_df.loc[cf_idx]

        # Get the model's prediction for the counterfactual instance.
        pred = model.predict([cf_scaled.values])[0]
        print(f"  {cf_idx} => Predicted: {pred}")

        # Compute the difference (in scaled units) between the counterfactual and the original instance.
        diff_scaled = cf_scaled - original_scaled
        abs_diff_scaled = diff_scaled.abs()

        # Identify the top N features with the largest absolute differences.
        top_features = abs_diff_scaled.nlargest(top_n).index

        # Convert the differences for these top features back to their original (unscaled) units.
        diff_unscaled = unscale_diff(diff_scaled, scaler, numeric_columns)

        # Create a DataFrame displaying both the scaled and unscaled differences.
        top_diff_df = pd.DataFrame({
            'Δ_scaled': diff_scaled[top_features],
            'Δ_unscaled': diff_unscaled[top_features]
        })

        print(f"\n  Top {top_n} changes for {cf_idx}: (scaled vs. unscaled)")
        display(top_diff_df)
        print("-" * 60)

def plot_feature_distributions(data, features, title="Feature Distributions"):
    """
    Plot distributions for each feature in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset containing the features.
        features (list): List of feature names to plot.
        title (str, optional): Title for the overall figure. Defaults to "Feature Distributions".

    Returns:
        None. Displays the generated plots.
    """
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(features):
        plt.subplot(len(features) // 3 + 1, 3, i + 1)
        sns.histplot(data[feature], kde=True)
        plt.title(f"Distribution of {feature}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data, title="Correlation Heatmap"):
    """
    Display a heatmap of the correlation matrix for the dataset.

    Parameters:
        data (pd.DataFrame): The dataset to compute correlations from.
        title (str, optional): Title for the heatmap. Defaults to "Correlation Heatmap".

    Returns:
        None. Displays the heatmap.
    """
    plt.figure(figsize=(15, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.show()

def plot_pca(data, target, title="PCA Plot"):
    """
    Plots a PCA scatter plot using only numeric features.
    
    Parameters:
    - data: pd.DataFrame containing the dataset (may include non-numeric columns).
    - target: str, name of the target column.
    - title: str, title for the plot.
    """
    # Filter only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Fit PCA on numeric features only
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(numeric_data)
    
    # Create a dataframe with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    # If target is non-numeric, try to map it from the original data
    pca_df[target] = data[target].values
    
    # Plot PCA scatter plot
    plt.figure(figsize=(10, 7))
    categories = pca_df[target].unique()
    for cat in categories:
        indices = pca_df[target] == cat
        plt.scatter(pca_df.loc[indices, 'PC1'], pca_df.loc[indices, 'PC2'], label=str(cat))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_count_all(data, title_prefix="Count Plot for "):
    """
    Plot count plots for all categorical features in the DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        title_prefix (str): A prefix for each plot title.

    This function automatically selects columns of type 'object' or 'category'
    and displays their count plots using subplots.
    """
    # Get categorical columns (including object and category types)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        print("No categorical features found in the data.")
        return

    n_cols = 2  # Number of plots per row
    n_rows = (len(categorical_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
    
    # Flatten axes array for easy iteration when there are multiple subplots
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, col in enumerate(categorical_cols):
        sns.countplot(x=col, data=data, ax=axes[i])
        axes[i].set_title(f"{title_prefix}{col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
    
    # Hide any extra subplots if categorical_cols does not fill them all
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
