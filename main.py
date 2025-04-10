from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

from utils import get_data_fps, get_data_t1t2, split_data

conv_input_path = 'data/results_cleaned/last_run/ConvolutionInputGenerator_hls_merged_cleaned.csv'
mvau_input_path = 'data/results_cleaned/last_run/MVAU_hls_merged_cleaned.csv'
area_summary_path = 'data/results_onnx/last_run/area_summary.csv'

def random_forest_wrapper(X_train, y_train, X_test, y_test,
                            output_path, model=None,):
    """
    Wrapper function for Random Forest regression.
    """
    if model is None:
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # eval
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # pred vs actual
    df_results = pd.DataFrame({
        'Actual_Total_LUTs': y_test.values.flatten(),
        'Pred_Total_LUTs': y_pred.flatten()
    })
    
    # save evaluation metrics
    metrics_file = os.path.join(output_path, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"R^2 Score: {r2}\n")
    print(f"Metrics saved to {metrics_file}")

    # save results
    output_file = os.path.join(output_path, 'output.csv')
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_results['Actual_Total_LUTs'], df_results['Pred_Total_LUTs'], alpha=0.6)
    plt.plot([df_results['Actual_Total_LUTs'].min(), df_results['Actual_Total_LUTs'].max()],
             [df_results['Actual_Total_LUTs'].min(), df_results['Actual_Total_LUTs'].max()],
             color='red', linestyle='--')
    plt.title('Actual vs Predicted Total LUTs')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    plot_file = os.path.join(output_path, 'prediction_plot.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")

    residuals = df_results['Actual_Total_LUTs'] - df_results['Pred_Total_LUTs']
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plot_file = os.path.join(output_path, 'residuals_plot.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")

def tuning_rf(x_train, y_train, output_path):
    """
    Perform hyperparameter tuning for RandomForestRegressor using GridSearchCV.
    """
    print("Tuning Random Forest with GridSearchCV...")
    param_grid = {
    'estimator__n_estimators': [100, 300, 500],
    'estimator__max_depth': [10, 20, 30, None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__max_features': ['sqrt', 'log2', None]
    }

    base_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    grid_search = GridSearchCV(estimator=base_model,
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               scoring='neg_mean_squared_error',
                               verbose=1)

    grid_search.fit(x_train, y_train)
    best_params_file = os.path.join(output_path, 'best_params.json')
    with open(best_params_file, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_file}")
    return grid_search.best_estimator_


def main():
    parser = argparse.ArgumentParser(description='HARA')
    
    parser.add_argument('--input', type=str, help='Input CSV file path', default=area_summary_path)
    parser.add_argument('--output', type=str, help='Output CSV file path', default='results/')
    parser.add_argument('--model', type=str, help='Model type', default='random_forest')
    parser.add_argument('--split', type=str, help='Split type', default='500fps')

    args = parser.parse_args()

    args.output = args.output + args.model + '/' + args.split + '/' 
    os.makedirs(args.output, exist_ok=True)

    train, test = get_data_fps(area_summary_path, args.split)
    train = train.drop(train.columns[:2], axis=1)
    test = test.drop(test.columns[:2], axis=1)
    X_train, y_train = split_data(train, ['Total LUTs'])
    X_test, y_test = split_data(test, ['Total LUTs'])

    if args.model == 'random_forest':
        best_model = tuning_rf(X_train, y_train, args.output)
        random_forest_wrapper(X_train, y_train, X_test, y_test, args.output, model=best_model)
    else:
        print(f"Model {args.model} not recognized. Please use 'random_forest'")
    
if __name__ == "__main__":
    main()