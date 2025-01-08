import os
import cv2
import numpy as np
from skimage import io
from skimage.transform import resize
from scipy.ndimage import map_coordinates

# Define directories
source_dir = '/dcs05/ciprian/smart/pocus/rushil/full_png'
mask_dir = '/dcs05/ciprian/smart/pocus/data/mask/'
output_rescaled_dir = '/dcs05/ciprian/smart/pocus/rushil/bounding_box_rescaled/'
output_rectilinear_dir = '/dcs05/ciprian/smart/pocus/rushil/rectilinear_rescaled/'
output_masked_dir = '/dcs05/ciprian/smart/pocus/rushil/masked/'

# Get a list of all .png files in source_dir
sep_png_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_dir) for f in filenames if f.endswith('.png')]

# Function to apply mask
def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

# Iterate through each .png file in source_dir
for file_path in sep_png_files:
    slice_file = os.path.basename(file_path)
    slice_folder = os.path.dirname(file_path)

    # Extract patient_id and DICOM name
    parts = os.path.splitext(slice_file)[0].split('_')
    patient_id = parts[0]
    dicom_filename = parts[1]

    # Find the corresponding mask file
    mask_file = os.path.join(mask_dir, patient_id, dicom_filename + '.png')

    # Extract the subfolder structure (training/testing/validation, b-line/control)
    split_parts = slice_folder.split(os.sep)
    data_set_split = split_parts[-2]
    data_label_category = split_parts[-1]

    # Define the new output names and paths for rescaled and rectilinear images
    output_rescaled_file = os.path.join(output_rescaled_dir, data_set_split, data_label_category, slice_file)
    rect_output_file = os.path.join(output_rectilinear_dir, data_set_split, data_label_category, slice_file)
    masked_output_file = os.path.join(output_masked_dir, data_set_split, data_label_category, slice_file)

    output_rescaled_subdir = os.path.join(output_rescaled_dir, data_set_split, data_label_category)
    output_rectilinear_subdir = os.path.join(output_rectilinear_dir, data_set_split, data_label_category)
    output_masked_subdir = os.path.join(output_masked_dir, data_set_split, data_label_category)

    # Make directories if they do not exist
    os.makedirs(output_rescaled_subdir, exist_ok=True)
    os.makedirs(output_rectilinear_subdir, exist_ok=True)
    os.makedirs(output_masked_subdir, exist_ok=True)

    # Read the image and mask
    image = io.imread(file_path)
    mask = io.imread(mask_file)

    # Apply clip level mask
    image = apply_mask(image, mask)
    target_size = (224, 224)
    masked_resize = resize(image, target_size, anti_aliasing=True)
    io.imsave(masked_output_file, (masked_resize * 255).astype(np.uint8))

    # Bounding box and cropping logic
    binary_mask = mask > 1
    props = cv2.boundingRect(binary_mask.astype(np.uint8))
    x, y, width, height = props

    # Crop the image using the bounding box
    cropped_image = image[y:y+height, x:x+width]

    # Resize and save the cropped image
    resized_image = resize(cropped_image, target_size, anti_aliasing=True)
    io.imsave(output_rescaled_file, (resized_image * 255).astype(np.uint8))

    # Polar conversion and rectilinear image creation
    apex = [x + width / 2, y]
    theta, rho = np.meshgrid(np.linspace(0, np.pi, width), np.linspace(0, height, height))
    X, Y = rho * np.cos(theta) + apex[0], rho * np.sin(theta) + apex[1]
    X = np.clip(X, 0, image.shape[1] - 1).astype(np.int32)
    Y = np.clip(Y, 0, image.shape[0] - 1).astype(np.int32)

    rectilinear_image = map_coordinates(image, [Y, X], order=1, mode='constant', cval=0)
    edges = cv2.Canny(rectilinear_image.astype(np.uint8), 100, 200)
    y_coords, x_coords = np.where(edges)
    rectilinear_image = rectilinear_image[min(y_coords):max(y_coords), min(x_coords):max(x_coords)]
    rectilinear_image = resize(rectilinear_image, target_size, anti_aliasing=True)
    rectilinear_image = np.fliplr(rectilinear_image)
    io.imsave(rect_output_file, (rectilinear_image * 255).astype(np.uint8))