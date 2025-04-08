import pandas as pd

def get_data(path: str):
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


