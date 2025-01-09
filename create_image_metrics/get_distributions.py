import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files
masked_metrics = pd.read_csv('/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/masked_images_metrics.csv')
bounding_box_metrics = pd.read_csv('/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/bounding_box_images_metrics.csv')
rectilinear_metrics = pd.read_csv('/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/rectilinear_images_metrics.csv')

# Visualize the distribution of mean intensity by label
sns.violinplot(data=rectilinear_metrics, x='Label', y='mean')
plt.title('Distribution of Mean Intensity by Label')
plt.show()

# Visualize the distribution of variance by label
sns.violinplot(data=rectilinear_metrics, x='Label', y='variance')
plt.title('Distribution of Variance by Label')
plt.show()
