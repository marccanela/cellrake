"""
Created on Thu Jul  4 15:43:26 2024
@author: mcanela
"""

from pathlib import Path
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops, label
from shapely.geometry import Polygon
import pickle as pkl

from utils import get_cell_mask, crop_cell, create_stats_dict, get_layer, compress, crop, fix_polygon


def filter_roi(layer, roi_dict, cell_background_threshold=None):
    '''
    Filter Regions of Interest (ROIs) based on their characteristics and
    background ratios. This function evaluates and filters ROIs based on
    their size and the ratio of cell pixel values to background pixel values.
    ROIs are initially pre-selected based on their area, and then further
    filtered based on a ratio threshold if specified.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which ROIs are
        extracted. The array should be of shape (height, width).

    rois : dict
        A dictionary where keys are ROI names and values are dictionaries with, 
        at least, the following keys:
        - "x": A list or array of x-coordinates of the ROI vertices.
        - "y": A list or array of y-coordinates of the ROI vertices.

    cell_background_threshold : float, optional
        A threshold ratio for the cell-to-background pixel mean value. Only 
        ROIs with a mean ratio above this threshold are kept. If None, all 
        pre-selected ROIs are returned.

    Returns:
    -------
    dict
        A dictionary where keys are ROI names and values are dictionaries 
        containing ROI information.
    '''
    first_keeped = {}
    cell_background_ratios = {}
    final_keeped = {}
    
    for roi_name, roi_info in roi_dict.items():
        x_coords, y_coords = roi_info["x"], roi_info["y"]
        coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
        cell_mask = get_cell_mask(layer, coordinates)
        
        # Pre-select ROIs based on area
        prop = regionprops(label(cell_mask))[0]
        my_area = prop.area
        if my_area < 1000:

            # Crop to the bounding box of the ROI
            cell_pixels = layer[cell_mask == 1]
            layer_cropped = crop_cell(layer, x_coords, y_coords)
            background_mask_cropped = crop_cell(1 - cell_mask, x_coords, y_coords)
            background_pixels = layer_cropped[background_mask_cropped == 1]
            
            # Calculate the mean ratio
            cell_pixels_mean = np.mean(cell_pixels)
            background_pixels_mean = np.mean(background_pixels)
            mean_ratio = cell_pixels_mean / (background_pixels_mean if background_pixels_mean != 0 else 1)
            if mean_ratio > 0:
                first_keeped[roi_name] = roi_info
                cell_background_ratios[roi_name] = mean_ratio
                
    if cell_background_threshold == None:
        return first_keeped
    else:
        for roi_name, mean_ratio in cell_background_ratios.items():
            if mean_ratio >= cell_background_threshold:
                final_keeped[roi_name] = first_keeped[roi_name]
        return final_keeped


def analyze_roi(roi_name, roi_stats, best_model):
    '''
    This function uses features extracted from a ROI as imput of a pretrained model
    to determine if the ROI meets certain criteria.

    Parameters:
    ----------
    roi_name : str
        The name or identifier of the ROI being analyzed.

    roi_stats : dict
        A dictionary containing the stats and textures of the ROI.

    best_model : object
        A trained model with a `predict` method that takes a feature array as 
        input and returns a prediction.

    Returns:
    -------
    str or None
        The ROI name if the `best_model` predicts that the ROI meets the
        criteria (label == 1); otherwise, `None`.
    '''
      
    # Exclude large ROIs
    # if roi_stats['area'] > 1000:
    #     return None    
    
    # Pre-select the ROIs based on mean difference
    # if roi_stats['mean_difference'] < 0:
    #     return False
    
    # Create input matrix
    X = np.array(list(roi_stats.values())).reshape(1,-1)

    return True if best_model.predict(X) == 1 else False


def analyze_image(tag, layers, rois, cmap, project_folder, best_model):
    '''
    Analyze an image, extract and process ROIs, and visualize results..

    Parameters:
    ----------
    tiff_path : Path
        The name of the image file Path to be analyzed.

    cmap : matplotlib.colors.Colormap
        The colormap to be used for visualization of the image and ROIs.

    best_model : object or None
        A trained model with a `predict` method, used to classify ROIs. 
        If `None`, the function uses a default filtering approach.

    output_folder : str
        The path to the folder where the ROIs are stored and images will be saved.
    
    root : str
        Exclusive identifier of the image.
        
    Returns:
    -------
    dict
        A dictionary of ROIs that are considered positive according to the
        model or filtering criteria. The dictionary keys are ROI names, and
        the values are dictionaries containing the ROI information.
    '''
    # Load roi_info and layer
    roi_dict = rois[tag]
    layer = layers[tag]
                
    # Process ROIs
    if best_model:

        roi_props = create_stats_dict(roi_dict, layer)
        results = {roi_name: analyze_roi(roi_name, roi_stats, best_model) 
                   for roi_name, roi_stats in roi_props.items()}
        
        keeped = {roi_name: roi_dict[roi_name]
                  for roi_name, result in results.items() if result == True}

    else:
        keeped = filter_roi(layer, roi_dict, 1.1)
        
    #Export the processed ROIs
    pkl_path = project_folder / 'rois_processed' / f'{tag}.pkl'
    with open(str(pkl_path), 'wb') as file:
        pkl.dump(keeped, file)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(layer, cmap=cmap, vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # ROIs
    axes[1].imshow(layer, cmap=cmap, vmin=0, vmax=255)
    for roi in keeped.values():
        axes[1].plot(roi["x"], roi["y"], 'b-', linewidth=1)
    axes[1].set_title("Identified Cells")
    axes[1].axis("off")
    
    # canvas = np.zeros(layer.shape, dtype=np.uint8)
    # for roi in keeped.values():
    #     coordinates = np.array([list(zip(roi["x"], roi["y"]))], dtype=np.int32)
    #     cv2.fillPoly(canvas, coordinates, 255)
    # axes[1].imshow(canvas, cmap=cmap)
    # axes[1].set_title("Identified Cells")
    # axes[1].axis("off")
    
    plt.tight_layout()
    png_path = project_folder / 'labelled_images' / f'{tag}.png'
    plt.savefig(png_path)
    plt.close()

    return keeped


def iterate_predicting(layers, rois, cmap, project_folder, best_model):
    '''
    Processes each TIFF image in the specified folder by:
    1. Identifying positive Regions of Interest (ROIs) using `analyze_image`.
    2. Calculating the number of ROIs, the total area, and the density of
    cells per square millimeter.
    3. Aggregating and saving the results into CSV.

    Parameters:
    ----------
    image_folder : str
        The path to the folder containing TIFF images and ROI area files.

    cmap : matplotlib.colors.Colormap
        The colormap to be used for visualization of images.

    best_model : object or None
        A trained model with a `predict` method, used to classify ROIs. If
        `None`, the function uses a default filtering approach.

    output_folder : str
        The path to the folder where there are the ROIs and results will be saved.

    areas_folder : str
        The path to the folder containing area files in `.txt` format.
        
    Returns:
    -------
    None
        This function does not return a value but saves the results to CSV.

    Notes:
    -----
    - The function assumes that each TIFF image file has a corresponding area
    file with a `.txt` extension.
    - The area file should be located in `areas_folder` and should have a
    tab-delimited format with a column named "Area".
    - The results are saved as a CSV file named "results.csv" in the `output_folder`.

    '''
    results = []
    
    for tag in tqdm(rois.keys(), desc="Applying prediction model", unit="image"):
        try:
            # Get the keeped ROIs
            keeped = analyze_image(tag, layers, rois, cmap, project_folder, best_model)
            
            # Count ROIs
            final_count = len(keeped)
            results.append((tag, final_count))
        
        except Exception as e:
            print(f"Error processing {tag}: {e}")
    
    df = pd.DataFrame(results, columns=["file_name", "num_cells"])
    df.to_csv(project_folder / "counts.csv", index=False)
    df.to_excel(project_folder / "counts.xlsx", index=False)


def colocalize(processed_rois_path_1, images_path_1, processed_rois_path_2, images_path_2):
    """
    This function processes TIFF images from two sets of identified ROIs, 
    compares them to find overlaps, and exports the results as images and
    CSV file. The overlap is determined based on an 80% area overlap
    criterion.

    Parameters:
    ----------
    rois1 : str
        Path to the first folder containing subfolders with identified ROIs.
    
    rois1_images : str
        Path to the folder containing images corresponding to the first set of ROIs.
    
    rois2 : str
        Path to the second folder containing subfolders with identified ROIs.
    
    rois2_images : str
        Path to the folder containing images corresponding to the second set of ROIs.
    
    areas_folder : str
        Path to the folder containing area information files for the images.

    Returns:
    -------
    None
        This function does not return a value but saves the overlapping ROIs 
        results as images and data files.

    Notes:
    -----
    - The function assumes that each TIFF image file in `rois1_images` and 
    `rois2_images` has a corresponding area file
      with a `.txt` extension in `areas_folder`.
    - The ROI data is expected to be in `.roi` format.
    - The overlap images and results are saved in the "overlap" subfolder 
    within `rois1`.
    """
    file_name = []
    num_cells = []
    
    colocalization_folder_path = processed_rois_path_1.parent / f'colocalization_{images_path_1.stem}_{images_path_2.stem}'
    colocalization_folder_path.mkdir(parents=True, exist_ok=True)
    colocalization_images_path = colocalization_folder_path / 'labelled_images'
    colocalization_images_path.mkdir(parents=True, exist_ok=True)

    for processed_roi_path_1 in tqdm(list(processed_rois_path_1.glob("*.pkl")), desc="Processing images", unit="image"):
        
        # Select an image from image_path_1
        tag = processed_roi_path_1.stem[3:]
        overlapped = {}
                
        # Load and index ROIs from image_1 (ROI_1)        
        with open(processed_roi_path_1, 'rb') as file:
            processed_roi_1 = pkl.load(file)

        rois_indexed_1 = {}            
        for roi_name_1, roi_info_1 in processed_roi_1.items():
            x_coords_1, y_coords_1 = roi_info_1["x"], roi_info_1["y"]
            polygon_1 = Polygon(zip(x_coords_1, y_coords_1))
            polygon_1 = fix_polygon(polygon_1)
            if polygon_1 is not None:
                rois_indexed_1[roi_name_1] = polygon_1
        
        # Compare each ROI_1 with each ROI_2 to look for colocalization
        processed_roi_path_2 = list(processed_rois_path_2.glob(f"*{tag}.pkl"))[0]
        
        with open(processed_roi_path_2, 'rb') as file:
            processed_roi_2 = pkl.load(file)
            
        for roi_name_2, roi_info_2 in processed_roi_2.items():
            x_coords_2, y_coords_2 = roi_info_2["x"], roi_info_2["y"]
            polygon_2 = Polygon(zip(x_coords_2, y_coords_2))
            polygon_2 = fix_polygon(polygon_2)
            if polygon_2 is not None:
            
                for roi_name_1, polygon_1 in rois_indexed_1.items():
                    intersection = polygon_1.intersection(polygon_2)
                    intersection_area = intersection.area
        
                    if intersection_area > 0:
                        area_roi_1 = polygon_1.area
                        area_roi_2 = polygon_2.area
                        smaller_roi = min(area_roi_1, area_roi_2)
                        if intersection_area >= 0.8 * smaller_roi:
                            overlapped[roi_name_1] = processed_roi_1[roi_name_1]
                            break
        
        # Plot the results
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        image_path_1 = list(images_path_1.glob(f"*{tag}.tif"))[0]
        image_1 = cv2.imread(str(image_path_1), cv2.IMREAD_UNCHANGED)
        image_1 = compress(image_1)
        layer_1 = get_layer(image_1)
        layer_1 = crop(layer_1)        
        axes[0].imshow(layer_1, cmap='Greens', vmin=0, vmax=255)
        axes[0].set_title(f"Original {images_path_1.stem} image")
        axes[0].axis("off")
        
        axes[1].imshow(layer_1, cmap='Greens', vmin=0, vmax=255)
        for roi in overlapped.values():
            axes[1].plot(roi["x"], roi["y"], 'b-', linewidth=1)
        axes[1].set_title("Colocalized Cells")
        axes[1].axis("off")
        
        image_path_2 = list(images_path_2.glob(f"*{tag}.tif"))[0]
        image_2 = cv2.imread(str(image_path_2), cv2.IMREAD_UNCHANGED)
        image_2 = compress(image_2)
        layer_2 = get_layer(image_2)
        layer_2 = crop(layer_2) 
        axes[2].imshow(layer_2, cmap='Reds', vmin=0, vmax=255)
        axes[2].set_title(f"Original {images_path_2.stem} image")
        axes[2].axis("off")  
        
        axes[3].imshow(layer_2, cmap='Reds', vmin=0, vmax=255)
        for roi in overlapped.values():
            axes[3].plot(roi["x"], roi["y"], 'b-', linewidth=1)
        axes[3].set_title("Colocalized Cells")
        axes[3].axis("off")
        
        # canvas = np.zeros(layer_1.shape, dtype=np.uint8)
        # for roi in overlapped.values():
        #     coordinates = np.array([list(zip(roi["x"], roi["y"]))], dtype=np.int32)
        #     cv2.fillPoly(canvas, coordinates, 255)
        # axes[2].imshow(canvas, cmap='Blues')
        # axes[2].set_title("Colocalized Cells")
        # axes[2].axis("off")
                    
        plt.tight_layout()
        png_path = colocalization_images_path / f'{tag}.png'
        plt.savefig(png_path)
        plt.close()

        # Export the numerical results
        file_name.append(tag)
        num_cells.append(len(overlapped))

    # Raw dataframe
    df = pd.DataFrame(
        {
            "file_name": file_name,
            "num_cells": num_cells,
        }
    )
    
    df.to_csv(colocalization_folder_path / f'colocalization_{images_path_1.stem}_{images_path_2.stem}.csv', index=False)
    df.to_excel(colocalization_folder_path / f'colocalization_{images_path_1.stem}_{images_path_2.stem}.csv', index=False)






















