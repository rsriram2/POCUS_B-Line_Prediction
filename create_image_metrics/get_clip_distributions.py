import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files
masked_metrics = pd.read_csv('/Users/rushil/Downloads/agg_masked_images_metrics.csv')
bounding_box_metrics = pd.read_csv('/Users/rushil/Downloads/agg_bounding_box_images_metrics.csv')
rectilinear_metrics = pd.read_csv('/Users/rushil/Downloads/agg_rectilinear_images_metrics.csv')

def save_Plots(data, x, y, title, output_path):
    sns.violinplot(data=data, x=x, y=y)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


save_Plots(masked_metrics, 'Label_aggregate_label', 'mean_mean', 'Clip-Level Distribution of Mean_Mean', '/Users/rushil/Downloads/masked_clip_mean_mean_distribution.png')
save_Plots(masked_metrics, 'Label_aggregate_label', 'mean_max', 'Clip-Level Distribution of Mean_Max', '/Users/rushil/Downloads/masked_clip_mean_max_distribution.png')
save_Plots(masked_metrics, 'Label_aggregate_label', 'mean_min', 'Clip-Level Distribution of Mean_Min', '/Users/rushil/Downloads/masked_clip_mean_min_distribution.png')
save_Plots(masked_metrics, 'Label_aggregate_label', 'variance_mean', 'Clip-Level Distribution of Variance_mean', '/Users/rushil/Downloads/masked_clip_variance_mean_distribution.png')
save_Plots(masked_metrics, 'Label_aggregate_label', 'variance_max', 'Clip-Level Distribution of Variance_max', '/Users/rushil/Downloads/masked_clip_variance_max_distribution.png')
save_Plots(masked_metrics, 'Label_aggregate_label', 'variance_min', 'Clip-Level Distribution of Variance_min', '/Users/rushil/Downloads/masked_clip_variance_min_distribution.png')

save_Plots(bounding_box_metrics, 'Label_aggregate_label', 'mean_mean', 'Clip-Level Distribution of Mean_Mean', '/Users/rushil/Downloads/bounding_box_clip_mean_mean_distribution.png')
save_Plots(bounding_box_metrics, 'Label_aggregate_label', 'mean_max', 'Clip-Level Distribution of Mean_Max', '/Users/rushil/Downloads/bounding_box_clip_mean_max_distribution.png')
save_Plots(bounding_box_metrics, 'Label_aggregate_label', 'mean_min', 'Clip-Level Distribution of Mean_Min', '/Users/rushil/Downloads/bounding_box_clip_mean_min_distribution.png')
save_Plots(bounding_box_metrics, 'Label_aggregate_label', 'variance_mean', 'Clip-Level Distribution of Variance_mean', '/Users/rushil/Downloads/bounding_box_clip_variance_mean_distribution.png')
save_Plots(bounding_box_metrics, 'Label_aggregate_label', 'variance_max', 'Clip-Level Distribution of Variance_max', '/Users/rushil/Downloads/bounding_box_clip_variance_max_distribution.png')
save_Plots(bounding_box_metrics, 'Label_aggregate_label', 'variance_min', 'Clip-Level Distribution of Variance_min', '/Users/rushil/Downloads/bounding_box_clip_variance_min_distribution.png')

save_Plots(rectilinear_metrics, 'Label_aggregate_label', 'mean_mean', 'Clip-Level Distribution of Mean_Mean', '/Users/rushil/Downloads/rectilinear_clip_mean_mean_distribution.png')
save_Plots(rectilinear_metrics, 'Label_aggregate_label', 'mean_max', 'Clip-Level Distribution of Mean_Max', '/Users/rushil/Downloads/rectilinear_clip_mean_max_distribution.png')
save_Plots(rectilinear_metrics, 'Label_aggregate_label', 'mean_min', 'Clip-Level Distribution of Mean_Min', '/Users/rushil/Downloads/rectilinear_clip_mean_min_distribution.png')
save_Plots(rectilinear_metrics, 'Label_aggregate_label', 'variance_mean', 'Clip-Level Distribution of Variance_mean', '/Users/rushil/Downloads/rectilinear_clip_variance_mean_distribution.png')
save_Plots(rectilinear_metrics, 'Label_aggregate_label', 'variance_max', 'Clip-Level Distribution of Variance_max', '/Users/rushil/Downloads/rectilinear_clip_variance_max_distribution.png')
save_Plots(rectilinear_metrics, 'Label_aggregate_label', 'variance_min', 'Clip-Level Distribution of Variance_min', '/Users/rushil/Downloads/rectilinear_clip_variance_min_distribution.png')
