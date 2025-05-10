import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from scipy.stats import randint, uniform

from sklearn.preprocessing import MinMaxScaler
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse

from utils import get_data_fps, get_data_t1t2, get_random_data, split_data, plot_correlation_matrix

conv_input_path = 'data/results_cleaned/last_run/ConvolutionInputGenerator_hls_merged_cleaned.csv'
mvau_input_path = 'data/results_cleaned/last_run/MVAU_hls_merged_cleaned.csv'
area_summary_path = 'data/results_onnx/last_run/area_summary.csv'


def model_wrapper(X_train, y_train, X_test, y_test,
                            output_path, targets, model=None, target_scaler=None, args=None):
    """
    Wrapper function for ML models.
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred)
        y_test = target_scaler.inverse_transform(y_test)

    # eval
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # pred vs actual
    df_results = pd.DataFrame(y_test, columns=[f'Actual_{t}' for t in targets])
    pred_df = pd.DataFrame(y_pred, columns=[f'Pred_{t}' for t in targets])
    df_results = pd.concat([df_results, pred_df], axis=1)

    # save results
    output_file = os.path.join(output_path, 'output.csv')
    df_results.to_csv(output_file, index=False)
    # print(f"Results saved to {output_file}")
    
    # save evaluation metrics
    metrics_file = os.path.join(output_path, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"R^2 Score: {r2}\n")
    # print(f"Metrics saved to {metrics_file}")

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
        # print(f"Plot saved to {plot_file}")

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
        # print(f"Residual plot saved to {res_file}")

    if(args.tuning == 'no'):
        return {"mse": mse, "r2": r2}

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

def tuning_keras_model(x_train, y_train, output_path, input_dim):
    """
    Perform hyperparameter tuning for a Keras model using RandomizedSearchCV.
    """

    print("Tuning keras_seq with RandomizedSearchCV...")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    keras_reg = KerasRegressor(
        model=create_keras_nn_model,
        model__input_dim=input_dim,
        verbose=0,
        callbacks=[early_stop]
    )

    param_dist = {
        'estimator__model__units_1': randint(64, 128),  
        'estimator__model__units_2': randint(64, 128),
        'estimator__model__units_3': randint(64, 128),
        'estimator__model__units_4': randint(64, 128),
        # 'estimator__model__learning_rate': uniform(0.0001, 0.01),
        'estimator__model__learning_rate': uniform(0.001, 0.005),
        # 'estimator__epochs': [50, 100, 150],
        # 'estimator__batch_size': [16, 32, 64]
        'estimator__epochs': [100, 200],  # fewer choices
        'estimator__batch_size': [32, 64]
    }

    base_model = MultiOutputRegressor(keras_reg)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(x_train, y_train)

    # Save best parameters
    best_params_file = os.path.join(output_path, 'best_params_keras.json')
    with open(best_params_file, 'w') as f:
        json.dump(random_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_file}")

    return random_search.best_estimator_

def create_nn_model(X_train=None):
    """
    Create a feedforward neural network model using MultiLayerPerceptron.
    """

    model = MLPRegressor(hidden_layer_sizes=(100, 50),
                        activation='relu',
                        solver='adam',
                        max_iter=500,
                        random_state=42)

    return model

def create_keras_nn_model(input_dim, units_1=64, units_2=64, units_3=64, units_4=64, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(units_1, activation='relu'))
    model.add(Dense(units_2, activation='relu'))
    model.add(Dense(units_3, activation='relu'))
    model.add(Dense(units_4, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

def load_best_params(input_dim, param_path):
    with open(param_path, 'r') as f:
        best_params = json.load(f)

    units_1 = best_params['estimator__model__units_1']
    units_2 = best_params['estimator__model__units_2']
    units_3 = best_params['estimator__model__units_3']
    units_4 = best_params['estimator__model__units_4']
    learning_rate = best_params['estimator__model__learning_rate']
    epochs = best_params['estimator__epochs']
    batch_size = best_params['estimator__batch_size']

    model = create_keras_nn_model(input_dim=input_dim,
                                   units_1=units_1,
                                   units_2=units_2,
                                   units_3=units_3,
                                   units_4=units_4,
                                   learning_rate=learning_rate)

    return model, epochs, batch_size

def main():
    parser = argparse.ArgumentParser(description='HARA')
    
    parser.add_argument('--input', type=str, help='Input CSV file path', default=mvau_input_path)
    parser.add_argument('--output', type=str, help='Output CSV file path', default='results/')
    parser.add_argument('--model', type=str, help='Model type', default='random_forest')
    parser.add_argument('--split', type=str, help='Split type', default='random')
    parser.add_argument('--target', type=str, help='Target variable', default='all')
    parser.add_argument('--plot', type=str, help='Plot type', default='none')
    parser.add_argument('--scaled', type=str, default='yes')
    parser.add_argument('--tuning', type=str, default='no', help='Tuning model or not')

    args = parser.parse_args()

    args.output = args.output + args.model + '/' + args.split + '/' 
    os.makedirs(args.output, exist_ok=True)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print("Available GPUs:", gpus)
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    if(args.target == 'luts'):
        args.target = ['Total LUTs']
    if(args.target == 'all'):
        args.target = ['Total LUTs', 'Logic LUTs','LUTRAMs','FFs','RAMB36','RAMB18','DSP Blocks']

    if(args.split == 'random'):
        train, test = get_random_data(args.input)
    else:
        train, test = get_data_fps(args.input, args.split)
    train = train.drop(columns=['Repo', 'NodeName', 'SRLs'])
    test = test.drop(columns=['Repo', 'NodeName','SRLs'])
    X_train, y_train = split_data(train, args.target)
    X_test, y_test = split_data(test, args.target)

    if(args.scaled == 'yes'):
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)  
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)

    if args.plot == 'corr':
        print("Plotting correlation matrices...")
        for target in args.target:
            output_path = f'plots/{target}_correlation_matrix.png'
            print(f"Plotting correlation matrix for {target}...")
            plot_correlation_matrix(train, [target], 0.5, output_path)
        return
    
    if args.tuning == 'no':
        print("Tuning is disabled. Using default parameters for all models.")

        output_base_path = "spot_checking_results/"
        os.makedirs(output_base_path, exist_ok=True)

        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        models = {
            'random_forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
            'mlp': MultiOutputRegressor(create_nn_model()),
            'knn': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
            'xgboost': MultiOutputRegressor(XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42))
        }

        X = pd.concat([X_train, X_test]).values
        y = pd.concat([y_train, y_test]).values 

        for name, model in models.items():
            print(f"\n=== Cross-validating {name} ===")
            model_output_path = os.path.join(output_base_path, name)
            os.makedirs(model_output_path, exist_ok=True)

            metrics_list = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
                # Split data
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]

                # Scale inside CV loop
                feature_scaler = MinMaxScaler()
                X_train_scaled = feature_scaler.fit_transform(X_train_fold)
                X_test_scaled = feature_scaler.transform(X_test_fold)

                target_scaler = MinMaxScaler()
                y_train_scaled = target_scaler.fit_transform(y_train_fold)
                y_test_scaled = target_scaler.transform(y_test_fold)

                fold_output_path = os.path.join(model_output_path, f"fold_{fold+1}")
                if os.path.exists(fold_output_path):
                    shutil.rmtree(fold_output_path)
                os.makedirs(fold_output_path, exist_ok=True)

                # Modify model_wrapper to return metrics
                metrics = model_wrapper(
                    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                    output_path=fold_output_path,
                    targets=args.target,
                    model=model,
                    target_scaler=target_scaler,
                    args=args
                )

                metrics_list.append(metrics)

            # Aggregate metrics
            mse_vals = [m['mse'] for m in metrics_list]
            r2_vals = [m['r2'] for m in metrics_list]
            print(f"\n{name} - Avg MSE: {np.mean(mse_vals):.4f} (+/- {np.std(mse_vals):.4f})")
            print(f"{name} - Avg R^2: {np.mean(r2_vals):.4f} (+/- {np.std(r2_vals):.4f})")

    exit()

    if args.model == 'random_forest':
        best_model = tuning_model(X_train, y_train, args.output, args.model)
        model_wrapper(X_train, y_train, X_test, y_test, args.output, args.target, model=best_model)

    if args.model == 'xgboost':
        best_model = tuning_model(X_train, y_train, args.output, args.model)
        model_wrapper(X_train, y_train, X_test, y_test, args.output, args.target, model=best_model)

    if args.model == 'mlp':
        model = MultiOutputRegressor(create_nn_model(X_train))
        model_wrapper(X_train, y_train, X_test, y_test, args.output, args.target, model=model)

    if args.model == 'keras_seq':
        if(args.scaled == 'yes'):
            
            input_dim = X_train_scaled.shape[1]
            best_params_path = os.path.join(args.output, 'best_params_keras.json')

            if os.path.exists(best_params_path):
                print("Loading best parameters from previous run...")
                model_fn, epochs, batch_size = load_best_params(input_dim, best_params_path)
                keras_reg = KerasRegressor(model=model_fn, epochs=epochs, batch_size=batch_size, verbose=0)
                model = MultiOutputRegressor(keras_reg)

                model_wrapper(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, args.output,
                            args.target, model=model, target_scaler=target_scaler)
                return

            print("No previous run found, tuning keras model...")
            input_dim = X_train_scaled.shape[1]
            best_model = tuning_keras_model(X_train_scaled, y_train_scaled, args.output, input_dim)
            model_wrapper(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, args.output, args.target, model=best_model, target_scaler=target_scaler)
            return

        input_dim = X_train.shape[1]
        best_model = tuning_keras_model(X_train, y_train, args.output, input_dim)
        model_wrapper(X_train, y_train, X_test, y_test, args.output, args.target, model=best_model)

    else:
        print(f"Model {args.model} not recognized.")
    
if __name__ == "__main__":
    main()