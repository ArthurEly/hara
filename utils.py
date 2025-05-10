import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
