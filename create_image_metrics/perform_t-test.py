import pandas as pd
from scipy.stats import ttest_ind

masked_metrics = pd.read_csv('/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/masked_images_metrics.csv')
bounding_box_metrics = pd.read_csv('/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/bounding_box_images_metrics.csv')
rectilinear_metrics = pd.read_csv('/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/rectilinear_images_metrics.csv')

def perform_ttest(df, metric):
    labels = df['Label'].unique()
    group1 = df[df['Label'] == labels[0]][metric]
    group2 = df[df['Label'] == labels[1]][metric]
    t_stat, p_val = ttest_ind(group1, group2)
    print(f"t-test for {metric} between {labels[0]} and {labels[1]}:")
    print(f"t-statistic: {t_stat}, p-value: {p_val}")

print("Masked Images Metrics:")
perform_ttest(masked_metrics, 'mean')
perform_ttest(masked_metrics, 'variance')

print("Bounding Box Images Metrics:")
perform_ttest(bounding_box_metrics, 'mean')
perform_ttest(bounding_box_metrics, 'variance')

print("Rectilinear Images Metrics:")
perform_ttest(rectilinear_metrics, 'mean')
perform_ttest(rectilinear_metrics, 'variance')
