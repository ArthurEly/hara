import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from xmtr import *
from utils import get_data_fps, get_data_t1t2, get_random_data, split_data, plot_correlation_matrix, new_corr_matrix, plot_cumulative_feature_importance
from utils import get_instance

label_area_path = 'retrieval/results/vivado_LabelSelect_area_attrs.csv'
conv_area_path = 'retrieval/results/vivado_ConvolutionInputGenerator_area_attrs.csv'
padding_area_path = 'retrieval/results/vivado_FMPadding_area_attrs.csv'
mvau_area_path = 'retrieval/results/vivado_MVAU_area_attrs.csv'
data_w_area_path = 'retrieval/results/vivado_StreamingDataWidthConverter_area_attrs.csv'
fifo_area_path = 'retrieval/results/vivado_StreamingFIFO_area_attrs.csv'


def main():
    parser = argparse.ArgumentParser(description='HARA')
    
    parser.add_argument('--input', type=str, help='Input CSV file path', default=label_area_path)
    parser.add_argument('--output', type=str, help='Output CSV file path', default='results/')
    parser.add_argument('--model', type=str, help='Model type', default='random_forest')
    parser.add_argument('--split', type=str, help='Split type', default='random')
    parser.add_argument('--target', type=str, help='Target variable', default='all')
    parser.add_argument('--plot', type=str, help='Plot type', default='corr')
    parser.add_argument('--tuning', type=bool, help='Tuning model or not', default=False)
    parser.add_argument('--importance', type=bool, help='Feature importance or not', default=False)
    parser.add_argument('--interpret', type=bool, help='Interpretation or not', default=False)

    args = parser.parse_args()

    args.output = args.output + args.model + '/' + args.split + '/' 
    os.makedirs(args.output, exist_ok=True)

    # print all the columns names in the input file
    print("Input file columns:", pd.read_csv(args.input).columns.tolist())

    # print one instance of the the files:
    for file in [label_area_path, conv_area_path, padding_area_path, mvau_area_path, data_w_area_path, fifo_area_path]:
        print("Input file path:", file)
        print("One instance of the input file:")
        print(pd.read_csv(file).iloc[0])

    # if(args.split == 'random'):
    #     train, test = get_random_data(args.input)
    # else:
    #     train, test = get_data_fps(args.input, args.split)
    
    # train = train.drop(columns=['Repo', 'NodeName', 'SRLs'])
    # test = test.drop(columns=['Repo', 'NodeName', 'SRLs'])

    # if(args.target == 'luts'):
    #     args.target = ['Total LUTs']
    # if(args.target == 'all'):
    #     # args.target = ['Total LUTs','FFs','RAMBs','DSP Blocks']
    #     args.target = ['Total LUTs','FFs','RAMB36','RAMB18','DSP Blocks']

    # X_train, y_train = split_data(train, args.target)
    # X_test, y_test = split_data(test, args.target)

    # if (args.interpret):
    #     if(args.model == 'random_forest')
    #         xmtr_interpreter = MTR(model, X_train, X_test, y_train, y_test, X_train.columns, y_train.columns)
    #         instance = get_instance(X_test)
    #         print("Prediction and interpretation rule:", xmtr_interpreter.explain(instance, 1)) 
    #     else:
    #         raise ValueError("Currently only random forest model is supported for interpretation.")

    

    

if __name__ == "__main__":
    main()