# Created by: Marc Canela

import pickle as pkl
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from sklearn.base import BaseEstimator
from tqdm import tqdm

from cellrake.utils import create_stats_dict, crop, export_data, fix_polygon


def analyze_image(
    tag: str,
    layers: Dict[str, np.ndarray],
    rois: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    cmap: mcolors.Colormap,
    project_folder: Path,
    best_model: BaseEstimator,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    """
    Analyze an image by processing ROIs, classifying them using a model, and visualizing results.

    Parameters
    ----------
    tag : str
        Unique identifier for the image to be analyzed.
    layers : Dict[str, np.ndarray]
        Dictionary where keys are image tags and values are 2D numpy arrays representing image layers.
    rois : Dict[str, Dict[str, Dict[str, np.ndarray]]]
        Dictionary where keys are image tags and values are dictionaries of ROIs with coordinates.
    cmap : mcolors.Colormap
        Colormap to be used for visualization.
    project_folder : Path
        Directory where processed ROIs and visualizations will be saved.
    best_model : BaseEstimator
        Trained model with predict method for classifying ROIs.

    Returns
    -------
    Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]
        Tuple containing positive ROIs dictionary and feature DataFrame.
    """
    # Load the ROI information and image layer
    roi_dict = rois[tag]
    layer = layers[tag]

    # Process ROIs: extract features
    roi_props = create_stats_dict(roi_dict, layer)

    input_df = pd.DataFrame.from_dict(roi_props, orient="index")
    input_df["min_intensity"] = input_df["min_intensity"].astype(int)
    input_df["max_intensity"] = input_df["max_intensity"].astype(int)
    input_df["hog_mean"] = input_df["hog_mean"].astype(float)
    input_df["hog_std"] = input_df["hog_std"].astype(float)
    input_X = input_df.values

    # Classify ROIs using the model
    prediction = best_model.predict(input_X)
    input_names = input_df.index
    results = dict(zip(input_names, prediction))

    # Keep ROIs that the model classifies as positive
    keeped = {
        roi_name: roi_dict[roi_name]
        for roi_name, result in results.items()
        if result == 1
    }

    # Create a new dictionary with sequential keys
    sorted_keys = sorted(keeped.keys(), key=lambda x: int(x.split("_")[1]))
    keeped = {f"roi_{i+1}": keeped[key] for i, key in enumerate(sorted_keys)}

    # Filter the input_df
    input_df = input_df[input_df.index.isin(keeped.keys())]

    # Plot results
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

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

    return keeped, input_df


def iterate_predicting(
    layers: Dict[str, np.ndarray],
    rois: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    cmap: mcolors.Colormap,
    best_model: BaseEstimator,
    project_folder: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process each image by identifying positive ROIs and calculating cell counts.

    Parameters
    ----------
    layers : Dict[str, np.ndarray]
        Dictionary where keys are image tags and values are 2D numpy arrays representing image layers.
    rois : Dict[str, Dict[str, Dict[str, np.ndarray]]]
        Dictionary where keys are image tags and values are dictionaries of ROIs with coordinates.
    cmap : mcolors.Colormap
        Colormap to be used for visualization.
    best_model : BaseEstimator
        Trained model with predict method for classifying ROIs.
    project_folder : Path
        Directory where processed ROIs and visualizations will be saved.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing cell counts DataFrame and concatenated features DataFrame.
    """
    results = []
    concatenated_df = pd.DataFrame()
    detected = {}

    for tag in tqdm(
        rois.keys(), desc="Identifying positive segmentations", unit="image"
    ):
        try:
            # Analyze ROIs and get the filtered list
            keeped, input_df = analyze_image(
                tag, layers, rois, cmap, project_folder, best_model
            )

            # Save keeped
            detected[tag] = keeped

            # Save features
            input_df = input_df.reset_index()
            input_df.rename(columns={"index": "rois"}, inplace=True)
            input_df["image"] = tag
            columns_order = ["image", "rois"] + [
                col for col in input_df.columns if col not in ["image", "rois"]
            ]
            input_df = input_df[columns_order]
            concatenated_df = pd.concat([concatenated_df, input_df], ignore_index=True)

            # Count the number of positive ROIs (cells)
            final_count = len(keeped)
            results.append((tag, final_count))

        except Exception as e:
            print(f"Error processing {tag}: {e}")

    # Convert results to a DataFrame and save to CSV and Excel
    df = pd.DataFrame(results, columns=["file_name", "num_cells"])

    return df, concatenated_df


def colocalize(
    processed_rois_path_1: Path,
    images_path_1: Path,
    processed_rois_path_2: Path,
    images_path_2: Path,
) -> None:
    """
    Compare two sets of ROIs to find overlapping cells based on 80% area overlap criterion.

    Parameters
    ----------
    processed_rois_path_1 : Path
        Path to PKL file with processed ROIs from the first image set.
    images_path_1 : Path
        Path to folder containing TIFF images corresponding to first ROI set.
    processed_rois_path_2 : Path
        Path to PKL file with processed ROIs from the second image set.
    images_path_2 : Path
        Path to folder containing TIFF images corresponding to second ROI set.

    Returns
    -------
    None
        Saves overlapping ROI results as images and CSV files.
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

    # Open detections
    with open(processed_rois_path_1, "rb") as file:
        rois_1 = pkl.load(file)
    with open(processed_rois_path_2, "rb") as file:
        rois_2 = pkl.load(file)

    # Start iterations
    for image_tag_1, processed_rois_1 in tqdm(
        rois_1.items(), desc="Processing images", unit="image"
    ):
        tag = image_tag_1[3:]
        overlapped = {}

        rois_indexed_1 = {}
        for roi_name_1, roi_info_1 in processed_rois_1.items():
            x_coords_1, y_coords_1 = roi_info_1["x"], roi_info_1["y"]
            polygon_1 = Polygon(zip(x_coords_1, y_coords_1))
            polygon_1 = fix_polygon(polygon_1)
            if polygon_1 is not None:
                rois_indexed_1[roi_name_1] = polygon_1

        # Compare with ROIs from the second set of images
        matching_key = next((key for key in rois_2 if key.endswith(tag)), None)
        if matching_key:
            processed_roi_2 = rois_2[matching_key]
        else:
            print(f"Skipping image '{tag}'")
            continue

        for _, roi_info_2 in processed_roi_2.items():
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
                            overlapped[roi_name_1] = processed_rois_1[roi_name_1]
                            break

        # Plot results
        _, axes = plt.subplots(1, 4, figsize=(15, 5))

        image_path_1 = list(images_path_1.glob(f"*{tag}.tif"))[0]
        image_1 = np.asarray(Image.open(image_path_1))
        layer_1 = crop(image_1)
        layer_1 = layer_1.astype(np.uint8)
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
        layer_2 = crop(image_2)
        layer_2 = layer_2.astype(np.uint8)
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
    export_data(df, colocalization_folder_path, "colocalization_results")
