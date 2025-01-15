import pandas as pd
import os

def extract_details_from_filename(filepath):
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    pid, clip, slice_ = name.split('_')[0], name.split('_')[1], name.split('_')[2]
    return pid, clip, slice_

def aggregate_label(Label):
    if 'b-line' in Label.values:
        return 'b-line'
    else:
        return 'control'

def aggregate_2D_metrics(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    df['PID'], df['Clip'], df['Slice'] = zip(*df['filename'].map(extract_details_from_filename))
    
    grouped_df = df.groupby(['PID', 'Clip']).agg({
        'mean': ['mean', 'max', 'min'],
        'variance': ['mean', 'max', 'min'],
        'Label': aggregate_label
    }).reset_index()
    
    grouped_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_df.columns.values]
    grouped_df.to_csv(output_csv_path, index=False)
    return grouped_df

# Define input and output file paths
input_masked_csv_path = '/Users/rushil/Downloads/masked_images_metrics.csv'
output_masked_csv_path = '/Users/rushil/Downloads/agg_masked_images_metrics.csv'
masked_grouped_df = aggregate_2D_metrics(input_masked_csv_path, output_masked_csv_path)

input_bounding_box_csv_path = '/Users/rushil/Downloads/bounding_box_images_metrics.csv'
output_bounding_box_csv_path = '/Users/rushil/Downloads/agg_bounding_box_images_metrics.csv'
bounding_box_grouped_df = aggregate_2D_metrics(input_bounding_box_csv_path, output_bounding_box_csv_path)

input_rectilinear_csv_path = '/Users/rushil/Downloads/rectilinear_images_metrics.csv'
output_rectilinear_csv_path = '/Users/rushil/Downloads/agg_rectilinear_images_metrics.csv'
rectilinear_grouped_df = aggregate_2D_metrics(input_rectilinear_csv_path, output_rectilinear_csv_path)
