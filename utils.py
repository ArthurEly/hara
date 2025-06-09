import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

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

    data = fill_nan_with_zero(data)
    data = remove_columns(data, ['Hardware config', 'Submodule Instance', 'op_type'])

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

def plot_cumulative_feature_importance(model, feature_names, output_path): # Added output_path
    """
    Plots the cumulative feature importances from multiple target variables in a single plot.
    
    Args:
        model (MultiOutputRegressor): The multi-output regressor model.
        feature_names (list): List of feature names.
        target_names (list): List of target variable names. (Mainly for context if needed, importances come from model)
        output_path (str): Directory to save the plot.
    """
    num_features = model.estimators_[0].n_features_in_ 
    cumulative_importances = np.zeros(num_features)
    
    # Loop through each target's estimator and add the importances
    for i, estimator in enumerate(model.estimators_):
        importances = estimator.feature_importances_
        if len(importances) == num_features:
            cumulative_importances += importances
        else:
            print(f"Warning: Mismatch in feature count for estimator {i}. Expected {num_features}, got {len(importances)}")
            continue 
            
    # cumulative_importances /= len(model.estimators_) 
    
    indices = cumulative_importances.argsort()[::-1]

    # Create the plot
    plt.figure(figsize=(12, 8))
    y_labels = [feature_names[i] for i in indices[:len(feature_names)]] 
    x_values = cumulative_importances[indices[:len(feature_names)]]

    sns.barplot(x=x_values, y=y_labels)
    plt.title('Cumulative Feature Importances (Sum)')
    plt.xlabel('Cumulative Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    filename = 'cumulative_feature_importances_sum_no_norm.png'
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def plot_individual_feature_importance(importances, feature_names, target_name, output_path):
    """
    Plots feature importances for a single target, scaled from 0 to 100.
    """
    importances_percent = importances * 100
    
    indices = np.argsort(importances_percent)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances_percent[indices], y=[feature_names[i] for i in indices])
    
    plt.title(f'Feature Importances for {target_name}')
    plt.xlabel('Importance (%)') # Update label
    plt.ylabel('Feature')
    
    plt.xlim(0, 100)
    
    plt.tight_layout()
    # Sanitize filename
    safe_target_name = "".join(c for c in target_name if c.isalnum() or c in (' ', '_')).rstrip()
    plt.savefig(os.path.join(output_path, f'importance_{safe_target_name.replace(" ", "_")}.png'))
    plt.close()

def fill_nan_with_zero(df):
    """
    Fills NaN values in a DataFrame with zero.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with NaN values filled with zero.
    """
    return df.fillna(0)

def remove_columns(df, columns_to_remove):
    """
    Removes string columns from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with string columns removed.
    """
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df

def remove_split_columns(train, test, columns_to_remove):
    """
    Removes specified columns from both training and testing DataFrames.

    Args:
        X_train (pd.DataFrame): The training set features.
        X_test (pd.DataFrame): The test set features.
        columns_to_remove (list): List of column names to remove.

    Returns:
        tuple: A tuple containing the modified training and testing DataFrames.
    """
    train = train.drop(columns=columns_to_remove, errors='ignore')
    test = test.drop(columns=columns_to_remove, errors='ignore')
    return train, test

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

def perform_feature_importance_analysis(X_train, y_train, model_type, output_path, importance_threshold=0.015):
    """
    Performs feature importance analysis.

    Returns:
        tuple: (list_of_unimportant_features, series_of_all_importances)
    """
    # ... (function logic is the same until the end)
    if model_type != 'random_forest':
        print(f"Feature importance is not implemented for model type: {model_type}")
        return [], pd.Series() # <--- Return empty list and empty Series

    print("Calculating feature importances using Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cumulative_importances = None

    # Case 1: Single Target
    if isinstance(y_train, pd.Series) or (isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1):
        # ... (logic remains the same)
        y_single = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
        target_name_str = y_single.name
        #print(f"Training single-target Random Forest for: {target_name_str}")
        rf_model.fit(X_train, y_single)
        importances = rf_model.feature_importances_
        plot_individual_feature_importance(importances, X_train.columns.tolist(), target_name_str, output_path)
        print(f"Feature importance plot saved for '{target_name_str}'")
        cumulative_importances = importances

    # Case 2: Multiple Targets
    elif isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
        # ... (logic remains the same)
        #print(f"Training multi-target Random Forest for targets: {y_train.columns.tolist()}")
        multioutput_model = MultiOutputRegressor(rf_model)
        multioutput_model.fit(X_train, y_train)
        # ... (plotting logic)
        num_features = multioutput_model.estimators_[0].n_features_in_
        calculated_importances = np.zeros(num_features)
        for estimator in multioutput_model.estimators_:
            calculated_importances += estimator.feature_importances_
        cumulative_importances = calculated_importances
        # ... (your existing plotting loops)
    
    else:
        print("Warning: y_train is in an unrecognized format for feature importance analysis.")
        return [], pd.Series() # <--- Return empty list and empty Series

    # --- Identify and return unimportant features ---
    unimportant_features_list = []
    # Initialize an empty series for the case where calculation fails
    feature_imp_series = pd.Series() 
    
    if cumulative_importances is not None:
        # Normalize the cumulative importances so they sum to 1.0
        normalized_importances = cumulative_importances / np.sum(cumulative_importances)
        feature_imp_series = pd.Series(normalized_importances, index=X_train.columns)
        
        # Filter features below the threshold
        unimportant_features = feature_imp_series[feature_imp_series < importance_threshold]
        
        if not unimportant_features.empty:
            print(f"\n--- Features below {importance_threshold:.2%} normalized importance threshold ---")
            for feature, score in unimportant_features.items():
                print(f"- {feature} (Importance: {score:.2%})")
            unimportant_features_list = unimportant_features.index.tolist()
        else:
            print(f"\nNo features found below the {importance_threshold:.2%} importance threshold.")
    
    # --- CHANGED: Return both the list and the full Series of importances ---
    return unimportant_features_list, feature_imp_series
    
def perform_correlation_analysis(X_train, y_train, output_path, feature_importances, threshold=0.9):
    """
    Performs correlation analysis and identifies features to remove based on
    multicollinearity, using feature importance scores to decide which feature
    from a correlated pair to drop.

    Args:
        X_train (pd.DataFrame): DataFrame with training features.
        y_train (pd.DataFrame or pd.Series): Series or DataFrame with training targets.
        output_path (str): The directory path to save the correlation plot.
        feature_importances (pd.Series): Series with feature names as index and
                                         their importance score as values.
        threshold (float): The correlation threshold to identify highly correlated features.

    Returns:
        list: A list of feature names recommended for removal.
    """
    print("\nPerforming correlation analysis with importance scores...")
    
    
    if feature_importances.empty:
        print("Warning: Feature importances are not available. Cannot intelligently select features to remove.")
        # Fallback to default behavior or simply return empty list
        return []

    # --- Part 1: Heatmap (remains the same) ---
    # ... (code for creating and saving the heatmap)
    full_df = pd.concat([X_train, y_train], axis=1)
    corr_matrix = full_df.corr(numeric_only=True)
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title('Feature-Target Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(output_path, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    # --- Part 2: Filter and Report with Importance Scores ---
    print(f"Correlation heatmap saved to: {heatmap_path}")
    print(f"Checking for multicollinearity (correlation > {threshold}) and deciding with feature importance...")
    
    corr_features = X_train.corr().abs()
    upper_tri = corr_features.where(np.triu(np.ones(corr_features.shape), k=1).astype(bool))
    
    features_to_remove = set()
    
    # Find pairs of columns with correlation greater than the threshold
    for column in upper_tri.columns:
        correlated_with = upper_tri.index[upper_tri[column] > threshold].tolist()
        for feature in correlated_with:
            # For each pair (feature, column), decide which one to drop
            importance1 = feature_importances.get(feature, 0)
            importance2 = feature_importances.get(column, 0)
            
            # Keep the one with higher importance
            if importance1 >= importance2:
                feature_to_drop = column
                feature_to_keep = feature
            else:
                feature_to_drop = feature
                feature_to_keep = column
            
            # Announce the decision and add the feature to the removal set
            if feature_to_drop not in features_to_remove:
                print(f"- Pair: ('{feature}', '{column}') | Correlation: {upper_tri.loc[feature, column]:.3f}")
                print(f"  - Importance '{feature_to_keep}': {max(importance1, importance2):.2%}")
                print(f"  - Importance '{feature_to_drop}': {min(importance1, importance2):.2%}")
                print(f"  - Decision: Removing '{feature_to_drop}'")
                features_to_remove.add(feature_to_drop)

    if not features_to_remove:
        print("No pairs of features found with a correlation above the threshold.")
    
    # --- CHANGED: Return the list of features to remove ---
    return sorted(list(features_to_remove))



def tune_model(X_train, y_train, model_type):
    """
    Tunes a model's hyperparameters using GridSearchCV with 5-fold cross-validation.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or pd.DataFrame): Training target(s).
        model_type (str): Type of model to tune ('random_forest', 'xgboost', 'neural_network').

    Returns:
        model: The best-tuned model found by GridSearchCV.
    """
    if model_type == 'random_forest':
        print(f"🚀 Tuning Random Forest model...")

        # 1. Define the base model and the multi-output wrapper
        rf = RandomForestRegressor(random_state=42)
        multi_output_rf = MultiOutputRegressor(rf)

        # 2. Define the hyperparameter grid to search
        param_grid = {
            'estimator__n_estimators': [100, 200],         # Number of trees
            'estimator__max_depth': [10, 30, None],        # Max depth of trees
            'estimator__min_samples_split': [2, 5],        # Min samples to split a node
            'estimator__min_samples_leaf': [1, 2]          # Min samples in a leaf node
        }

        # 3. Set up GridSearchCV
        # Uses 5-fold cross-validation, all CPU cores (n_jobs=-1), and minimizes MSE
        grid_search = GridSearchCV(
            estimator=multi_output_rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )

        # 4. Run the tuning process
        grid_search.fit(X_train, y_train)

        # 5. Print the results and return the best model
        print("\nTuning complete!")
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score (Negative MSE): {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    elif model_type == 'xgboost':
        print(f"Tuning for {model_type} is not yet implemented.")
        # Placeholder for XGBoost tuning logic. 
        return None

    elif model_type == 'neural_network':
        print(f"Tuning for {model_type} is not yet implemented.")
        # Placeholder for Neural Network tuning. 
        return None

    else:
        print(f"Unknown model type '{model_type}' for tuning.")
        return None
    
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def run_random_forest_experiment(X_train, y_train, X_test, y_test, output_path):
    """
    Trains a multi-output Random Forest model, evaluates it on the test set,
    and saves predictions, metrics, and diagnostic plots.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training targets.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.DataFrame): Testing targets.
        output_path (str): Path to the directory where results will be saved.
    """
    print("🚀 Starting Random Forest experiment...")
    
    # --- 1. Train the Model ---
    # Use MultiOutputRegressor to handle multiple target variables
    print("Training the model...")
    base_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_rf)
    model.fit(X_train, y_train)

    # --- 2. Make Predictions on the Test Set ---
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Convert predictions array to a DataFrame for easier handling
    y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

    # --- 3. Save Predictions ---
    # Concatenate actual and predicted values for easy comparison
    results_df = pd.concat([y_test.add_suffix('_actual'), y_pred_df.add_suffix('_pred')], axis=1)
    predictions_path = os.path.join(output_path, 'predictions.csv')
    results_df.to_csv(predictions_path)
    print(f"Predictions saved to: {predictions_path}")

    # --- 4. Calculate and Save Metrics ---
    metrics_path = os.path.join(output_path, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        print("\nCalculating and saving metrics...")
        f.write("--- Regression Metrics ---\n\n")
        
        # Calculate metrics for each target individually
        for target in y_test.columns:
            r2 = r2_score(y_test[target], y_pred_df[target])
            mse = mean_squared_error(y_test[target], y_pred_df[target])
            mae = mean_absolute_error(y_test[target], y_pred_df[target])
            
            f.write(f"Metrics for target: '{target}'\n")
            f.write(f"  R-squared (R²): {r2:.4f}\n")
            f.write(f"  Mean Squared Error (MSE): {mse:.4f}\n")
            f.write(f"  Mean Absolute Error (MAE): {mae:.4f}\n\n")
        
        # Optional: Calculate average metrics across all outputs
        avg_r2 = r2_score(y_test, y_pred_df, multioutput='uniform_average')
        f.write("--- Average Metrics Across All Targets ---\n")
        f.write(f"  Average R-squared (R²): {avg_r2:.4f}\n")
        
    print(f"Metrics saved to: {metrics_path}")

    # --- 5. Generate and Save Plots for Each Target ---
    print("Generating and saving plots...")

    # Define and create subdirectories for the plots
    predictions_plot_path = os.path.join(output_path, 'predictions_plots')
    residuals_plot_path = os.path.join(output_path, 'residuals_plots')
    os.makedirs(predictions_plot_path, exist_ok=True)
    os.makedirs(residuals_plot_path, exist_ok=True)

    for target in y_test.columns:
        # Sanitize target name for filenames
        safe_target_name = "".join(c for c in target if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')

        # a) Prediction vs. Actual Plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_test[target], y=y_pred_df[target], alpha=0.6)
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        plt.title(f'Prediction vs. Actual for {target}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot in the 'predictions_plots' subfolder
        prediction_filename = f'plot_prediction_vs_actual_{safe_target_name}.png'
        plt.savefig(os.path.join(predictions_plot_path, prediction_filename))
        plt.close()

        # b) Residuals Plot
        residuals = y_test[target] - y_pred_df[target]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred_df[target], y=residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Residuals for {target}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot in the 'residuals_plots' subfolder
        residual_filename = f'plot_residuals_{safe_target_name}.png'
        plt.savefig(os.path.join(residuals_plot_path, residual_filename))
        plt.close()

    print(f"Experiment complete! All outputs saved in: {output_path}")