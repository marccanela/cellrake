"""
@author: Marc Canela
"""

import pickle as pkl
from pathlib import Path
from typing import Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom
from shapely.geometry import Polygon
from skimage.measure import label, regionprops
from sklearn.base import BaseEstimator
from tqdm import tqdm

from cellrake.utils import (
    create_stats_dict,
    crop,
    crop_cell_large,
    fix_polygon,
    get_cell_mask,
)


def filter_roi(
    layer: np.ndarray,
    roi_dict: Dict[str, Dict[str, np.ndarray]],
    cell_background_threshold: Optional[float] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    This function evaluates and filters ROIs based on their size and ratios.
    ROIs are initially pre-selected based on their area, and then further
    filtered based on a ratio threshold if specified.

    Parameters:
    ----------
    layer : np.ndarray
        A 2D NumPy array representing the image layer from which ROIs are
        extracted. The array should be of shape (height, width).

    roi_dict : dict
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
    """
    first_kept = {}
    cell_background_ratios = {}
    final_kept = {}

    for roi_name, roi_info in roi_dict.items():
        x_coords, y_coords = roi_info["x"], roi_info["y"]
        coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
        cell_mask = get_cell_mask(layer, coordinates)

        # Pre-select ROIs based on area
        props = regionprops(label(cell_mask))
        if props:
            prop = props[0]
            roi_area = prop.area

            # Filter based on area
            if roi_area <= 250:
                # Crop to the bounding box of the ROI
                layer_cropped, _, _ = crop_cell_large(layer, x_coords, y_coords)
                cell_mask_cropped = crop_cell_large(cell_mask, x_coords, y_coords)[0]
                background_mask_cropped = 1 - cell_mask_cropped

                cell_pixels = layer_cropped[cell_mask_cropped == 1]
                background_pixels = layer_cropped[background_mask_cropped == 1]

                # Calculate the mean ratio
                cell_pixels_mean = np.mean(cell_pixels)
                background_pixels_mean = np.mean(background_pixels)
                mean_ratio = cell_pixels_mean / (
                    background_pixels_mean if background_pixels_mean != 0 else 1
                )

                if mean_ratio > 1:
                    first_kept[roi_name] = roi_info
                    cell_background_ratios[roi_name] = mean_ratio

    if cell_background_threshold is None:
        return first_kept
    else:
        for roi_name, mean_ratio in cell_background_ratios.items():
            if mean_ratio >= cell_background_threshold:
                final_kept[roi_name] = first_kept[roi_name]
        return final_kept


def analyze_roi(
    roi_name: str, roi_stats: Dict[str, float], best_model: BaseEstimator
) -> Optional[str]:
    """
    This function analyzes a ROI based on its features and a pretrained model
    to determine if the ROI meets certain criteria. The model's prediction indicates
    whether the ROI should be considered based on the criteria.

    Parameters:
    ----------
    roi_name : str
        The name or identifier of the ROI being analyzed.

    roi_stats : dict
        A dictionary containing the features (stats and textures) of the ROI.
        The dictionary should have feature names as keys and their corresponding values.

    best_model : BaseEstimator
        A trained model with a `predict` method that takes a feature array as
        input and returns a prediction. The model should be compatible with `sklearn`'s `predict` method.

    Returns:
    -------
    Optional[str]
        The ROI name if the model predicts that the ROI meets the criteria (label == 1);
        otherwise, returns `None`.
    """
    # Convert ROI stats dictionary to a feature array
    feature_array = np.array(list(roi_stats.values())).reshape(1, -1)

    # Predict using the model
    prediction = best_model.predict(feature_array)

    # Check if the prediction is for class 1
    return roi_name if prediction[0] == 1 else None


def analyze_image(
    tag: str,
    layers: Dict[str, np.ndarray],
    rois: Dict[str, Dict[str, np.ndarray]],
    cmap: mcolors.Colormap,
    project_folder: Path,
    best_model: Optional[BaseEstimator],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    This function analyzes an image by processing ROIs, classifying them using a model if provided,
    and visualizing the results.

    Parameters:
    ----------
    tag : str
        Unique identifier for the image to be analyzed.

    layers : dict
        A dictionary where keys are image tags and values are 2D numpy arrays
        representing the image layers.

    rois : dict
        A dictionary where keys are image tags and values are dictionaries of ROIs.
        Each ROI is represented by its coordinates in the dictionary.

    cmap : matplotlib.colors.Colormap
        The colormap to be used for visualization.

    project_folder : Path
        The directory where the processed ROIs and visualizations will be saved.

    best_model : Optional[BaseEstimator]
        A trained model with a `predict` method for classifying ROIs. If `None`,
        a default filtering approach is used.

    Returns:
    -------
    dict
        A dictionary of ROIs considered positive by the model or filtering criteria.
        The keys are ROI names and the values are dictionaries containing the ROI information.
    """
    # Load the ROI information and image layer
    roi_dict = rois[tag]
    layer = layers[tag]

    # Process ROIs
    if best_model:
        # Extract features and classify ROIs using the model
        roi_props = create_stats_dict(roi_dict, layer)
        results = {
            roi_name: analyze_roi(roi_name, roi_stats, best_model)
            for roi_name, roi_stats in roi_props.items()
        }

        # Keep ROIs that the model classifies as positive
        keeped = {
            roi_name: roi_dict[roi_name]
            for roi_name, result in results.items()
            if result
        }

    else:
        # Filter ROIs based on a default criterion if no model is provided
        keeped = filter_roi(layer, roi_dict, cell_background_threshold=1.1)

    # Export the processed ROIs
    processed_folder = project_folder / "rois_processed"
    processed_folder.mkdir(parents=True, exist_ok=True)
    pkl_path = processed_folder / f"{tag}.pkl"
    with open(pkl_path, "wb") as file:
        pkl.dump(keeped, file)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    axes[0].imshow(layer, cmap=cmap, vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot identified ROIs
    axes[1].imshow(layer, cmap=cmap, vmin=0, vmax=255)
    for roi in keeped.values():
        axes[1].plot(roi["x"], roi["y"], "b-", linewidth=1)
    axes[1].set_title("Identified Cells")
    axes[1].axis("off")

    plt.tight_layout()
    labeled_images_folder = project_folder / "labelled_images"
    labeled_images_folder.mkdir(parents=True, exist_ok=True)
    png_path = labeled_images_folder / f"{tag}.png"
    plt.savefig(png_path)
    plt.close()

    return keeped


def iterate_predicting(
    layers: Dict[str, np.ndarray],
    rois: Dict[str, Dict[str, np.ndarray]],
    cmap: mcolors.Colormap,
    project_folder: Path,
    best_model: Optional[BaseEstimator],
) -> None:
    """
    This function processes each image by identifying positive ROIs using
    the provided model or a default filtering approach. Calculates and saves statistics
    on the number of ROIs (cells) per image.

    Parameters:
    ----------
    layers : Dict[str, np.ndarray]
        A dictionary where keys are image tags and values are 2D numpy arrays
        representing the image layers.

    rois : Dict[str, Dict[str, np.ndarray]]
        A dictionary where keys are image tags and values are dictionaries of ROIs.
        Each ROI is represented by its coordinates in the dictionary.

    cmap : mcolors.Colormap
        The colormap to be used for visualization.

    project_folder : Path
        The path to the folder where results will be saved.

    best_model : Optional[BaseEstimator]
        A trained model with a `predict` method for classifying ROIs. If `None`,
        a default filtering approach is used.

    Returns:
    -------
    None
        This function does not return a value but saves the results to CSV and Excel files.

    Notes:
    -----
    - The function assumes that each image tag in `rois` has a corresponding image layer in `layers`.
    - Results are saved as "counts.csv" and "counts.xlsx" in the `project_folder`.
    """
    results = []

    for tag in tqdm(
        rois.keys(), desc="Identifying positive segmentations", unit="image"
    ):
        try:
            # Analyze ROIs and get the filtered list
            keeped = analyze_image(tag, layers, rois, cmap, project_folder, best_model)

            # Count the number of positive ROIs (cells)
            final_count = len(keeped)
            results.append((tag, final_count))

        except Exception as e:
            print(f"Error processing {tag}: {e}")

    # Convert results to a DataFrame and save to CSV and Excel
    df = pd.DataFrame(results, columns=["file_name", "num_cells"])
    df.to_csv(project_folder / "counts.csv", index=False)
    df.to_excel(project_folder / "counts.xlsx", index=False)


def colocalize(
    processed_rois_path_1: Path,
    images_path_1: Path,
    processed_rois_path_2: Path,
    images_path_2: Path,
) -> None:
    """
    This function processes TIFF images from two sets of identified ROIs, compares them to find
    overlaps based on an 80% area overlap criterion, and exports the results as images
    and CSV files.

    Parameters:
    ----------
    processed_rois_path_1 : Path
        Path to the folder containing processed ROIs from the first set of images.

    images_path_1 : Path
        Path to the folder containing TIFF images corresponding to the first set of ROIs.

    processed_rois_path_2 : Path
        Path to the folder containing processed ROIs from the second set of images.

    images_path_2 : Path
        Path to the folder containing TIFF images corresponding to the second set of ROIs.

    Returns:
    -------
    None
        This function does not return a value but saves overlapping ROI results as images
        and data files.

    Notes:
    -----
    - The function assumes that each TIFF image file in `images_path_1` and `images_path_2`
      has a corresponding ROI file in `processed_rois_path_1` and `processed_rois_path_2`.
    - The overlap images and results are saved in the "colocalization" subfolder within
      `processed_rois_path_1`.
    """
    file_names = []
    num_cells = []

    # Create directory for colocalization results
    colocalization_folder_path = (
        processed_rois_path_1.parent.parent
        / f"colocalization_{images_path_1.stem}_{images_path_2.stem}"
    )
    colocalization_folder_path.mkdir(parents=True, exist_ok=True)
    colocalization_images_path = colocalization_folder_path / "labelled_images"
    colocalization_images_path.mkdir(parents=True, exist_ok=True)

    for processed_roi_path_1 in tqdm(
        list(processed_rois_path_1.glob("*.pkl")),
        desc="Processing images",
        unit="image",
    ):
        tag = processed_roi_path_1.stem[3:]
        overlapped = {}

        # Load ROIs from the first set of images
        with open(processed_roi_path_1, "rb") as file:
            processed_roi_1 = pkl.load(file)

        rois_indexed_1 = {}
        for roi_name_1, roi_info_1 in processed_roi_1.items():
            x_coords_1, y_coords_1 = roi_info_1["x"], roi_info_1["y"]
            polygon_1 = Polygon(zip(x_coords_1, y_coords_1))
            polygon_1 = fix_polygon(polygon_1)
            if polygon_1 is not None:
                rois_indexed_1[roi_name_1] = polygon_1

        # Compare with ROIs from the second set of images
        processed_roi_path_2 = list(processed_rois_path_2.glob(f"*{tag}.pkl"))[0]

        with open(processed_roi_path_2, "rb") as file:
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

        # Plot results
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        image_path_1 = list(images_path_1.glob(f"*{tag}.tif"))[0]

        image_1 = np.asarray(Image.open(image_path_1))
        layer_1 = zoom(image_1, zoom=0.5, order=1)
        layer_1 = crop(layer_1)
        axes[0].imshow(layer_1, cmap="Greens", vmin=0, vmax=255)
        axes[0].set_title(f"Original {images_path_1.stem} image")
        axes[0].axis("off")

        axes[1].imshow(layer_1, cmap="Greens", vmin=0, vmax=255)
        for roi in overlapped.values():
            axes[1].plot(roi["x"], roi["y"], "b-", linewidth=1)
        axes[1].set_title("Colocalized Cells")
        axes[1].axis("off")

        image_path_2 = list(images_path_2.glob(f"*{tag}.tif"))[0]
        image_2 = np.asarray(Image.open(image_path_2))
        layer_2 = zoom(image_2, zoom=0.5, order=1)
        layer_2 = crop(layer_2)
        axes[2].imshow(layer_2, cmap="Reds", vmin=0, vmax=255)
        axes[2].set_title(f"Original {images_path_2.stem} image")
        axes[2].axis("off")

        axes[3].imshow(layer_2, cmap="Reds", vmin=0, vmax=255)
        for roi in overlapped.values():
            axes[3].plot(roi["x"], roi["y"], "b-", linewidth=1)
        axes[3].set_title("Colocalized Cells")
        axes[3].axis("off")

        plt.tight_layout()
        png_path = colocalization_images_path / f"{tag}.png"
        plt.savefig(png_path)
        plt.close()

        # Export the numerical results
        file_names.append(tag)
        num_cells.append(len(overlapped))

    # Save results as CSV and Excel
    df = pd.DataFrame(
        {
            "file_name": file_names,
            "num_cells": num_cells,
        }
    )
    df.to_csv(colocalization_folder_path / "colocalization_results.csv", index=False)
    df.to_excel(colocalization_folder_path / "colocalization_results.xlsx", index=False)
