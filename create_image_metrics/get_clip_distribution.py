import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

# Load the CSV file
masked_metrics = pd.read_csv('/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/masked_images_metrics.csv')

# Function to extract Patient ID, Clip No., and Slice No. from file name
def extract_info(file_name):
    match = re.search(r'/([^/]+)_IM(\d+)_([^/]+)\.png$', file_name)
    if match:
        patient_id = match.group(1)
        clip_no = match.group(2)
        slice_no = match.group(3)
        return patient_id, clip_no, slice_no
    return None, None, None

# Apply the function to extract information
masked_metrics['Patient ID'], masked_metrics['Clip No.'], masked_metrics['Slice No.'] = zip(*masked_metrics['filename'].apply(extract_info))

# Drop rows where extraction failed
masked_metrics.dropna(subset=['Patient ID', 'Clip No.', 'Slice No.'], inplace=True)

# Convert Slice No. to numeric
masked_metrics['Slice No.'] = pd.to_numeric(masked_metrics['Slice No.'], errors='coerce')

# Drop rows where Slice No. conversion failed
masked_metrics.dropna(subset=['Slice No.'], inplace=True)

# Select only numeric columns for grouping and calculation
numeric_cols = masked_metrics.select_dtypes(include='number').columns

# Ensure 'Slice No.' is included in the numeric columns
if 'Slice No.' not in numeric_cols:
    numeric_cols = numeric_cols.append(pd.Index(['Slice No.']))

# Group by Patient ID, Clip No., and Slice No., and calculate mean for each numeric column
grouped = masked_metrics.groupby(['Patient ID', 'Clip No.', 'Slice No.'])[numeric_cols].mean()

# Drop the original 'Slice No.' column to avoid conflict
grouped = grouped.drop(columns=['Slice No.'])

# Reset the index to make 'Patient ID', 'Clip No.', and 'Slice No.' columns
grouped = grouped.reset_index()

# Create a base directory for saving plots
base_dir = '/Users/rushil/POCUS_B-Line_Prediction/patient_plots'
os.makedirs(base_dir, exist_ok=True)

# Plot the mean values for each clip for every patient and save the plots
patients = grouped['Patient ID'].unique()
for patient in patients:
    patient_data = grouped[grouped['Patient ID'] == patient]
    clips = patient_data['Clip No.'].unique()
    patient_dir = os.path.join(base_dir, patient)
    os.makedirs(patient_dir, exist_ok=True)
    for clip in clips:
        clip_data = patient_data[patient_data['Clip No.'] == clip]
        plt.figure()
        sns.lineplot(data=clip_data, x='Slice No.', y='mean')
        plt.title(f'Mean Intensity for Patient {patient} - Clip {clip}')
        plt.xlabel('Slice No.')
        plt.ylabel('Mean Intensity')
        plot_path = os.path.join(patient_dir, f'{patient}_Clip_{clip}.png')
        plt.savefig(plot_path)
        plt.close()