import pandas as pd
from scipy.stats import ttest_ind

# Load the CSV files
masked_metrics = pd.read_csv('/Users/rushil/Downloads/agg_masked_images_metrics.csv')
bounding_box_metrics = pd.read_csv('/Users/rushil/Downloads/agg_bounding_box_images_metrics.csv')
rectilinear_metrics = pd.read_csv('/Users/rushil/Downloads/agg_rectilinear_images_metrics.csv')

def perform_ttest(df, metric, dataset_type, results_df):
    labels = df['Label_aggregate_label'].unique()
    group1 = df[df['Label_aggregate_label'] == labels[0]][metric]
    group2 = df[df['Label_aggregate_label'] == labels[1]][metric]
    t_stat, p_val = ttest_ind(group1, group2)
    new_row = pd.DataFrame({
        'Metric': [metric],
        'Dataset Type': [dataset_type],
        't-statistic': [t_stat],
        'p-value': [p_val]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Metric', 'Dataset Type', 't-statistic', 'p-value'])

print("Masked Images Metrics:")
results_df = perform_ttest(masked_metrics, 'mean_mean', 'Masked', results_df)
results_df = perform_ttest(masked_metrics, 'mean_max', 'Masked', results_df)
results_df = perform_ttest(masked_metrics, 'mean_min', 'Masked', results_df)
results_df = perform_ttest(masked_metrics, 'variance_mean', 'Masked', results_df)
results_df = perform_ttest(masked_metrics, 'variance_max', 'Masked', results_df)
results_df = perform_ttest(masked_metrics, 'variance_min', 'Masked', results_df)

print("Bounding Box Images Metrics:")
results_df = perform_ttest(bounding_box_metrics, 'mean_mean', 'Bounding Box', results_df)
results_df = perform_ttest(bounding_box_metrics, 'mean_max', 'Bounding Box', results_df)
results_df = perform_ttest(bounding_box_metrics, 'mean_min', 'Bounding Box', results_df)
results_df = perform_ttest(bounding_box_metrics, 'variance_mean', 'Bounding Box', results_df)
results_df = perform_ttest(bounding_box_metrics, 'variance_max', 'Bounding Box', results_df)
results_df = perform_ttest(bounding_box_metrics, 'variance_min', 'Bounding Box', results_df)

print("Rectilinear Images Metrics:")
results_df = perform_ttest(rectilinear_metrics, 'mean_mean', 'Rectilinear', results_df)
results_df = perform_ttest(rectilinear_metrics, 'mean_max', 'Rectilinear', results_df)
results_df = perform_ttest(rectilinear_metrics, 'mean_min', 'Rectilinear', results_df)
results_df = perform_ttest(rectilinear_metrics, 'variance_mean', 'Rectilinear', results_df)
results_df = perform_ttest(rectilinear_metrics, 'variance_max', 'Rectilinear', results_df)
results_df = perform_ttest(rectilinear_metrics, 'variance_min', 'Rectilinear', results_df)

# Display the results
print(results_df)

# Optionally, save the results to a CSV file
results_df.to_csv('/Users/rushil/Downloads/clip_ttest_results.csv', index=False)