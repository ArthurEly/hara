MODELS=("neural_network" "xgboost" "random_forest")
INPUTS=("label_select" "convolution" "padding" "mvau" "data_width_converter" "fifo")

for model in "${MODELS[@]}"; do
    for input in "${INPUTS[@]}"; do
        echo "Running model=$model on input=$input"
        python3 main.py --model "$model" --input "$input"
    done
done