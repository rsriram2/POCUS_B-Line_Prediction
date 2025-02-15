import numpy as np
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
from skimage import io

def segmentation_algorithm2(x):
    # Find connected components
    labeled_array, num_features = label(x > 0, return_num=True)
    region_props = regionprops(labeled_array)
    
    # Find the largest connected component
    largest_region = max(region_props, key=lambda region: region.area)
    mask = np.zeros_like(x)
    mask[labeled_array == largest_region.label] = 1
    
    # Flip the mask horizontally
    fmask = np.flip(mask, axis=1)
    
    # Create a symmetric mask
    symmask = (mask + fmask) > 0
    
    # Create a structural element
    se = disk(3)
    
    # Erode the symmetric mask
    dmask = erosion(symmask, se)
    
    # Find connected components in the eroded mask
    labeled_array, num_features = label(dmask, return_num=True)
    region_props = regionprops(labeled_array)
    
    # Find the largest connected component in the eroded mask
    largest_region = max(region_props, key=lambda region: region.area)
    new_mask = np.zeros_like(dmask)
    new_mask[labeled_array == largest_region.label] = 1
    
    # Create a larger structural element for dilation
    se5 = disk(5)
    
    # Dilate the mask
    new_mask = dilation(new_mask, se5)
    
    # Fill holes in the mask
    new_mask = clear_border(new_mask)
    
    return new_mask
