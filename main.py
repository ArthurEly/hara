from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
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


def model_wrapper(X_train, y_train, X_test, y_test,
                            output_path, targets, model=None, flag=None):
    """
    Wrapper function for ML models.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # eval
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # pred vs actual
    df_results = pd.DataFrame(y_test.values, columns=[f'Actual_{t}' for t in targets])
    pred_df = pd.DataFrame(y_pred, columns=[f'Pred_{t}' for t in targets])
    df_results = pd.concat([df_results, pred_df], axis=1)

    # save results
    output_file = os.path.join(output_path, 'output.csv')
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # save evaluation metrics
    metrics_file = os.path.join(output_path, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"R^2 Score: {r2}\n")
    print(f"Metrics saved to {metrics_file}")

    # plot
    for target in targets:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_results[f'Actual_{target}'], df_results[f'Pred_{target}'], alpha=0.6)
        plt.plot([df_results[f'Actual_{target}'].min(), df_results[f'Actual_{target}'].max()],
                 [df_results[f'Actual_{target}'].min(), df_results[f'Actual_{target}'].max()],
                 color='red', linestyle='--')
        plt.title(f'Actual vs Predicted - {target}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        plt.tight_layout()
        plot_file = os.path.join(output_path, f'prediction_plot_{target}.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved to {plot_file}")

        # Residuals
        residuals = df_results[f'Actual_{target}'] - df_results[f'Pred_{target}']
        plt.figure(figsize=(8, 4))
        plt.hist(residuals, bins=30)
        plt.title(f"Residuals - {target}")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        res_file = os.path.join(output_path, f'residuals_plot_{target}.png')
        plt.savefig(res_file)
        plt.close()
        print(f"Residual plot saved to {res_file}")

def tuning_model(x_train, y_train, output_path, model):
    """
    Perform hyperparameter tuning for ML models using GridSearchCV.
    """

    print(f"Tuning {model} with GridSearchCV...")

    if(model == 'random_forest'):
        param_grid = {
        'estimator__n_estimators': [100, 300, 500],
        'estimator__max_depth': [10, 20, 30, None],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['sqrt', 'log2', None]
        }

        base_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))

    if(model == 'xgboost'):
        param_grid = {
        'estimator__n_estimators': [100, 200, 300],
        'estimator__max_depth': [3, 6, 10],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__subsample': [0.8, 1.0],
        'estimator__colsample_bytree': [0.8, 1.0],
        'estimator__reg_alpha': [0, 0.1, 1],
        'estimator__reg_lambda': [1, 1.5, 2]
        }
        base_model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42))

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

def create_nn_model(X_train):
    """
    Create a simple feedforward neural network model.
    """

    model = MLPRegressor(hidden_layer_sizes=(100, 50),
                        activation='relu',
                        solver='adam',
                        max_iter=500,
                        random_state=42)
    return model


def main():
    parser = argparse.ArgumentParser(description='HARA')
    
    parser.add_argument('--input', type=str, help='Input CSV file path', default=mvau_input_path)
    parser.add_argument('--output', type=str, help='Output CSV file path', default='results/')
    parser.add_argument('--model', type=str, help='Model type', default='neural_network')
    parser.add_argument('--split', type=str, help='Split type', default='500fps')
    parser.add_argument('--target', type=str, help='Target variable', default='all')

    args = parser.parse_args()

    args.output = args.output + args.model + '/' + args.split + '/' 
    os.makedirs(args.output, exist_ok=True)

    if(args.target == 'luts'):
        args.target = ['Total LUTs']
    if(args.target == 'all'):
        args.target = ['Total LUTs', 'Logic LUTs','LUTRAMs','SRLs','FFs','RAMB36','RAMB18','DSP Blocks']

    train, test = get_data_fps(args.input, args.split)
    train = train.drop(columns=['Repo', 'NodeName'])
    test = test.drop(columns=['Repo', 'NodeName'])
    X_train, y_train = split_data(train, args.target)
    X_test, y_test = split_data(test, args.target)

    if args.model == 'random_forest':
        best_model = tuning_model(X_train, y_train, args.output, args.model)
        model_wrapper(X_train, y_train, X_test, y_test, args.output, args.target, model=best_model)

    if args.model == 'xgboost':
        best_model = tuning_model(X_train, y_train, args.output, args.model)
        model_wrapper(X_train, y_train, X_test, y_test, args.output, args.target, model=best_model)

    if args.model == 'neural_network':
        # y_train = y_train.values.ravel()
        # y_test = y_test.values.ravel()
        network = MultiOutputRegressor(create_nn_model(X_train))
        model_wrapper(X_train, y_train, X_test, y_test, args.output, args.target, model=network)
    else:
        print(f"Model {args.model} not recognized.")
    
if __name__ == "__main__":
    main()