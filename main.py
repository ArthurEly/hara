import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings

import argparse
from sklearn.preprocessing import StandardScaler
from xmtr import *
from utils import get_data_fps, get_random_data, split_data, remove_split_columns, get_instance
from utils import perform_feature_importance_analysis, perform_correlation_analysis
from utils import tune_model, evaluate_model

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATASET_PATHS = {
    'label_select': 'retrieval/results/splitted/preprocessed/vivado_LabelSelect_area_attrs_cleaned.csv',
    'convolution': 'retrieval/results/splitted/preprocessed/vivado_ConvolutionInputGenerator_area_attrs_cleaned.csv',
    'padding': 'retrieval/results/splitted/preprocessed/vivado_FMPadding_area_attrs_cleaned.csv',
    'mvau': 'retrieval/results/splitted/preprocessed/vivado_MVAU_area_attrs_cleaned.csv',
    'data_width_converter': 'retrieval/results/splitted/preprocessed/vivado_StreamingDataWidthConverter_area_attrs_cleaned.csv',
    'fifo': 'retrieval/results/splitted/preprocessed/vivado_StreamingFIFO_area_attrs_cleaned.csv'
}

def main():
    parser = argparse.ArgumentParser(description='HARA')
    
    parser.add_argument('--input', type=str, help='Name of the dataset to use.',default='mvau', choices=DATASET_PATHS.keys())
    parser.add_argument('--output', type=str, help='Output CSV file path', default='results/')
    parser.add_argument('--model', type=str, help='Model type', default='neural_network', choices=['xgboost', 'random_forest', 'neural_network'])
    parser.add_argument('--split', type=str, help='Split type', default='random')
    parser.add_argument('--target', type=str, help='Target variable', default='all')
    parser.add_argument('--plot', type=str, help='Plot type', default='corr')

    parser.add_argument('--tuning', action='store_true', help='Tuning model or not', default=True)
    parser.add_argument('--importance', action='store_true', help='Feature importance or not', default=True)
    parser.add_argument('--correlation', action='store_true', help='Correlation analysis or not', default=True)
    parser.add_argument('--interpret', action='store_true', help='Interpretation or not', default=True)

    args = parser.parse_args()

    output_directory = os.path.join(args.output, args.input, args.model)
    os.makedirs(output_directory, exist_ok=True)
    input_filepath = DATASET_PATHS[args.input]

    train, test = get_random_data(input_filepath)

    if(args.target == 'luts'):
        args.target = ['Total LUTs']
    if(args.target == 'all'):
        args.target = ['Total LUT','Total FFs','BRAM (36k eq.)','DSP Blocks']
        # args.target = ['Total LUT','Total FFs','DSP Blocks']

    X_train, y_train = split_data(train, args.target)
    X_test, y_test = split_data(test, args.target)

    if args.importance:
        importance_output_path = os.path.join(output_directory, 'feature_importance')
        os.makedirs(importance_output_path, exist_ok=True)
        unimportant_features, all_feature_importances = perform_feature_importance_analysis(X_train=X_train, y_train=y_train, 
                                                                                            output_path=importance_output_path, importance_threshold=0.015)
        X_train, X_test = remove_split_columns(X_train, X_test, unimportant_features)
        y_train, y_test = remove_split_columns(y_train, y_test, unimportant_features)
        print(f"Removed {len(unimportant_features)} features based on importance threshold.")

    if args.correlation:
        correlation_output_path = os.path.join(output_directory, 'correlation')
        os.makedirs(correlation_output_path, exist_ok=True)
        correlated_features_to_remove = perform_correlation_analysis(X_train=X_train, y_train=y_train, output_path=correlation_output_path, feature_importances=all_feature_importances, threshold=0.9)
        X_train, X_test = remove_split_columns(X_train, X_test, correlated_features_to_remove)
        y_train, y_test = remove_split_columns(y_train, y_test, correlated_features_to_remove)
        print(f"Removed {len(correlated_features_to_remove)} features based on correlation and importance.")

    if args.model == 'neural_network':
        print("Scaling data for neural network...")
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
    
    if args.tuning:
        print("Tuning model...")
        model = tune_model(X_train, y_train, model_type=args.model)
    
    print(f"Evaluating {args.model} model...")
    evaluate_model(model, X_test=X_test, y_test=y_test, output_path=output_directory, y_scaler=y_scaler if args.model == 'neural_network' else None)
    
if __name__ == "__main__":
    main()