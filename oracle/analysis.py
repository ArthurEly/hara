import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
MODEL_OPTIONS = ['random_forest', 'xgboost', 'neural_network']
DATASET_KEYS = ['label_select', 'convolution', 'padding', 'mvau', 'data_width_converter', 'fifo']

def parse_metrics_file(filepath):
    """
    Reads a single metrics.txt file and extracts the metrics into a structured format.
    (This is the corrected version)
    """
    metrics_data = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            current_target = None
            for line in lines:
                line = line.strip()
                if not line or "---" in line:
                    continue

                target_match = re.match(r"Metrics for target: '(.+)'", line)
                if target_match:
                    current_target = target_match.group(1)
                    continue

                if ':' in line:
                    parts = line.split(':', 1) 
                    metric_name_raw = parts[0].strip()
                    value_str = parts[1].strip()

                    metric_name = re.sub(r'\s*\([^)]*\)', '', metric_name_raw).strip()

                    try:
                        value = float(value_str)
                        target_for_record = current_target if current_target else 'Average'
                        
                        metrics_data.append({
                            'target': target_for_record,
                            'metric': metric_name,
                            'value': value
                        })
                    except ValueError:
                        continue

    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
        
    return metrics_data

def analyze_all_results():
    """
    Main function to parse all metrics files, create comparison plots,
    and analyze the results.
    """
    all_results = []
    print("--- Parsing all metric files ---")
    
    for dataset in DATASET_KEYS:
        for model in MODEL_OPTIONS:
            model_dir = os.path.join(RESULTS_DIR, dataset, model)
            metrics_file = os.path.join(model_dir, 'metrics.txt')

            if os.path.exists(metrics_file):
                print(f"Parsing: {metrics_file}")
                parsed_data = parse_metrics_file(metrics_file)
                for record in parsed_data:
                    record['dataset'] = dataset
                    record['model'] = model
                all_results.extend(parsed_data)
            else:
                 print(f"SKIPPING - Not found: {metrics_file}")


    if not all_results:
        print("\nNo metrics files found. Exiting analysis.")
        return

    df = pd.DataFrame(all_results)
    
    analysis_output_dir = 'analysis_results'
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    print("\n--- Analyzing MSE vs. MAE (Higher ratio = more large errors) ---")
    df_pivot = df.pivot_table(
        index=['dataset', 'model', 'target'],
        columns='metric',
        values='value'
    ).reset_index()

    df_pivot['MSE_MAE_Ratio'] = (df_pivot['Mean Squared Error'] / df_pivot['Mean Absolute Error']).fillna(0)
    
    big_mistakes = df_pivot.sort_values(by='MSE_MAE_Ratio', ascending=False)
    print("Top 5 model/target combinations most prone to large outlier errors:")
    print(big_mistakes[['dataset', 'model', 'target', 'MSE_MAE_Ratio']].head(5).to_string(index=False))

    metrics_to_plot = ['R-squared', 'Mean Squared Error', 'Mean Absolute Error']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(15, 8))
        
        plot_data = df[(df['metric'] == metric) & (df['target'] != 'Average')]
        
        if plot_data.empty:
            continue

        sns.barplot(data=plot_data, x='target', y='value', hue='model')
        
        plt.title(f'Model Comparison: {metric}', fontsize=16, pad=20)
        plt.xlabel('Target Variable', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"comparison_{metric.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(analysis_output_dir, plot_filename))
        plt.close()
        print(f"\nGenerated plot: {plot_filename}")

    print("\n--- Best Model per Target (based on lowest Mean Absolute Error) ---")
    mae_df = df[df['metric'] == 'Mean Absolute Error']
    
    if not mae_df.empty:
        # Find the index of the minimum MAE for each target within each dataset
        best_indices = mae_df.loc[mae_df.groupby(['dataset', 'target'])['value'].idxmin()]
        
        # Select and reorder columns for a clean printout
        best_models_summary = best_indices[['dataset', 'target', 'model', 'value']].rename(columns={'value': 'Lowest MAE'})
        print(best_models_summary.to_string(index=False))

if __name__ == "__main__":
    analyze_all_results()
