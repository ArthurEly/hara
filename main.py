from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np

from utils import get_data, split_data

conv_input_path = 'data/results_cleaned/ConvolutionInputGenerator_hls_merged_cleaned.csv'
mvau_input_path = 'data/results_cleaned/MVAU_hls_merged_cleaned.csv'
area_summary_path = 'data/results_onnx/area_summary.csv'

def main():

    train, test = get_data(conv_input_path)

    # remove Repo and NodeName columns from train and test datasets
    train = train.drop(train.columns[:2], axis=1)
    test = test.drop(test.columns[:2], axis=1)
    
    # split data into features and target variable
    X_train, y_train = split_data(train, ['Total LUTs', 'FFs'])
    X_test, y_test = split_data(test, ['Total LUTs', 'FFs'])

    # training and predicting
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # eval
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    # pred vs actual
    df_results = pd.DataFrame(
    np.hstack([y_test.values, y_pred]),
                columns=['Actual_Total_LUTs', 'Actual_FFs', 'Pred_Total_LUTs', 'Pred_FFs']
    )
    print(df_results.head(10))
    
if __name__ == "__main__":
    main()