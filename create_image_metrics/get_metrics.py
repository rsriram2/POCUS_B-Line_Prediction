from PIL import Image
import numpy as np
import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

def is_image_file(filename):
    return filename.lower().endswith('.png')

def load_images_with_glob(folder):
    images = []
    filenames = []
    labels = []
    # Use glob to search recursively for image files
    for filepath in glob.glob(os.path.join(folder, '**/*.png'), recursive=True):
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        images.append(np.array(img))
        filenames.append(filepath)
        labels.append(os.path.basename(os.path.dirname(filepath)))
    return images, filenames, labels


masked_images, masked_filenames, masked_labels = load_images_with_glob("/dcs05/ciprian/smart/pocus/rushil/2D_data/masked")
bounding_box_images, bounding_box_filenames, bounding_box_labels = load_images_with_glob("/dcs05/ciprian/smart/pocus/rushil/2D_data/bounding_box")
rectilinear_images, rectilinear_filenames, rectilinear_labels = load_images_with_glob("/dcs05/ciprian/smart/pocus/rushil/2D_data/rectilinear")

def get_metrics(images, filenames, labels):
    stats = []
    for img, filename, label in zip(images, filenames, labels):
        mean = np.mean(img)
        var = np.var(img)
        stats.append({'mean': mean, 'variance': var, 'filename': filename, 'Label': label})
    return stats

def save_metrics(images, filenames, labels, metrics_csv_path, descriptive_csv_path):
    metrics = get_metrics(images, filenames, labels)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_csv_path, index=False)
    descriptive_stats = metrics_df.groupby('Label').describe()
    descriptive_stats.to_csv(descriptive_csv_path)

save_metrics(masked_images, masked_filenames, masked_labels, "/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/masked_images_metrics.csv", "/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/masked_images_descriptive_stats.csv")
save_metrics(bounding_box_images, bounding_box_filenames, bounding_box_labels, "/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/bounding_box_images_metrics.csv", "/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/bounding_box_images_descriptive_stats.csv")
save_metrics(rectilinear_images, rectilinear_filenames, rectilinear_labels, "/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/rectilinear_images_metrics.csv", "/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/rectilinear_images_descriptive_stats.csv")
