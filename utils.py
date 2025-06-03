import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_data_t1t2(path: str):
    """
    Reads a CSV file from the given path and splits the data into training and testing datasets.
    Rows containing 't2' in the 'repo' column are considered training data,
    while rows containing 't1' are considered testing data.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        tuple: A tuple containing two DataFrames: (train_data, test_data).
    """

    data = pd.read_csv(path)
    train_data = data[data['Repo'].str.contains('t2')]
    test_data = data[data['Repo'].str.contains('t1')]

    return train_data, test_data

def get_data_fps(path: str, fps: str):

    """
    Reads a CSV file from the given path and splits the data into training and testing datasets.
    Rows containing the specified fps in the 'Repo' column are considered testing data,
    while all other rows are considered training data.

    Args:
        path (str): The file path to the CSV file.
        fps (str): The fps value to filter the testing data.

    Returns:
        tuple: A tuple containing two DataFrames: (train_data, test_data).
    """

    data = pd.read_csv(path)
    test_data = data[data['Repo'].str.contains(fps)]
    train_data = data[~data['Repo'].str.contains(fps)]

    return train_data, test_data

def get_random_data(path: str):
    """
    Reads a CSV file from the given path and splits the data into training and testing datasets.
    The training data is a random sample of 80% of the data, while the remaining 20% is used for testing.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        tuple: A tuple containing two DataFrames: (train_data, test_data).
    """

    data = pd.read_csv(path)

    # Remove the instance where Repo = 'MVAU_hls_0' and NodeName = 't1w8_50000fps_u'
    data = data[~((data['NodeName'] == 'MVAU_hls_0') & (data['Repo'] == 't1w8_50000fps_u'))]

    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    return train_data, test_data


def split_data(data: pd.DataFrame, targets):

    """
    Splits the data into features and target variable.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target (str): The name of the target variable column.

    Returns:
        tuple: A tuple containing two DataFrames: (X, y).
    """
    if isinstance(targets, str):
        targets = [targets]

    X = data.drop(columns=targets)
    y = data[targets]
    return X, y

def plot_correlation_matrix(df, target_columns, threshold=0.2, save_path=None):
    """
    Plots a correlation matrix heatmap for a DataFrame.

    Args:
        df (pd.DataFrame): The data.
        target_columns (list): List of target columns to highlight.
        save_path (str): If provided, saves the plot to this path.
    """
    correlation = df.corr(numeric_only=True)  # numeric_only avoids issues with object-type columns

    # Filter correlations with the target(s) only
    relevant_features = set()
    for target in target_columns:
        if target not in correlation:
            print(f"Warning: Target '{target}' not found in correlation matrix. Skipping.")
            continue

        valid_corr = pd.to_numeric(correlation[target], errors='coerce').dropna().astype(float)        
        strong_corr = valid_corr[(valid_corr.abs() >= threshold) & (valid_corr.abs() < 1.0)]
        relevant_features.update(strong_corr.index)

    relevant_features.update(target_columns)
    relevant_features_list = list(relevant_features)
    filtered_corr = correlation.loc[relevant_features_list, relevant_features_list]

    plt.figure(figsize=(14, 10))
    sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar=True, linewidths=0.5)

    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def new_corr_matrix(df):
    """
    Computes the correlation matrix for a DataFrame and includes feature names in the plot.

    Args:
        df (pd.DataFrame): The data.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    correlation_matrix = df.corr(numeric_only=True)  # numeric_only avoids issues with object-type columns
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                linewidths=0.5, xticklabels=correlation_matrix.columns, 
                yticklabels=correlation_matrix.columns)
    plt.title("Full Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig("full_correlation_matrix_new.png")
    plt.close()

def plot_cumulative_feature_importance(model, feature_names, target_names):
    """
    Plots the cumulative feature importances from multiple target variables in a single plot.
    
    Args:
        model (MultiOutputRegressor): The multi-output regressor model.
        feature_names (list): List of feature names.
        target_names (list): List of target variable names.
    """
    # Initialize an array to accumulate the importances for each feature
    cumulative_importances = np.zeros(len(feature_names))
    
    # Loop through each target and add the importances
    for i, target in enumerate(target_names):
        importances = model.estimators_[i].feature_importances_
        cumulative_importances += importances
        
    # Normalize by the number of targets
    #cumulative_importances /= len(target_names)
    
    # Get the indices to sort the features by importance
    indices = cumulative_importances.argsort()[::-1]

    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=cumulative_importances[indices], y=[feature_names[i] for i in indices])
    plt.title('Cumulative Feature Importances')
    plt.xlabel('Cumulative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('cumulative_feature_importances_no_norm.png')
    plt.close()

def get_instance(X_test):
    """
    Selects a random instance from the test set.

    Args:
        X_test (pd.DataFrame): The test set features.

    Returns:
        pd.Series: A random instance from the test set.
    """
    
    if X_test.empty:
        raise ValueError("X_test is empty. Cannot select a random instance.")
    
    random_index = np.random.choice(X_test.index)
    print(f"Selected random instance index: {random_index}")
    return X_test.loc[random_index]
    
