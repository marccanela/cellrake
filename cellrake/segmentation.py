# Created by: Marc Canela

import math
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import draw, feature, filters, measure, morphology
from tqdm import tqdm

from cellrake.utils import crop


def convert_to_roi(
    polygons: Dict[int, np.ndarray], layer: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    This function extracts the coordinates of the polygons and converts them into ROIs.
    It clips the coordinates to ensure they lie within the bounds of the given image layer.

    Parameters:
    ----------
    polygons : dict[int, np.ndarray]
        A dictionary where each key is a label and each value is a single contour for that label.

    layer : numpy.ndarray
        A 2D NumPy array representing the image layer. The shape of the array should be
        (height, width).

    Returns:
    -------
    dict
        A dictionary where each key is a string identifier for an ROI ("roi_1", "roi_2", etc.),
        and each value is another dictionary with 'x' and 'y' keys containing the clipped
        x and y coordinates of the ROI.
    """
    # Initialize an empty dictionary to store ROIs
    rois_dict = {}

    # Extract dimensions of the layer
    layer_height, layer_width = layer.shape

    # Iterate
    for n, (label, contour) in enumerate(polygons.items(), start=1):
        # Clip the coordinates to be within the bounds of the layer
        roi_y = np.clip(contour[:, 0], 0, layer_height - 1)
        roi_x = np.clip(contour[:, 1], 0, layer_width - 1)

        # Store the x and y coordinates in the dictionary.
        rois_dict[f"roi_{n}"] = {"x": roi_x, "y": roi_y}

    return rois_dict


def iterate_segmentation(
    image_folder: Path, threshold_rel: float
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, np.ndarray]]:

    rois = {}
    layers = {}

    for tif_path in tqdm(
        list(image_folder.glob("*.tif")), desc="Preprocessing images", unit="image"
    ):
        tag = tif_path.stem
        combined_array, layer = segment_image(tif_path, threshold_rel)
        labels = measure.label(combined_array)
        polygons = extract_polygons(labels)

        rois[tag] = convert_to_roi(polygons, layer)
        layers[tag] = layer

    return rois, layers


def export_rois(
    project_folder: Path, rois: Dict[str, Dict[str, Dict[str, np.ndarray]]]
) -> None:
    """
    This function saves the ROIs for each image into a separate `.pkl` file within the `rois_raw` directory
    inside the specified `project_folder`. Each file is named according to the image's tag (filename without extension).

    Parameters:
    ----------
    project_folder : pathlib.Path
        A Path object pointing to the project directory where the ROIs will be saved.

    rois : dict[str, dict]
        A dictionary where keys are image tags (filenames without extension) and values are dictionaries of ROI data.

    Returns:
    -------
    None
    """
    # Export each ROI dictionary to a .pkl file
    for tag, rois_dict in rois.items():
        with open(str((project_folder / "rois_raw") / f"{tag}.pkl"), "wb") as file:
            pkl.dump(rois_dict, file)


def process_blob(layer: np.ndarray, blob: np.ndarray) -> Optional[np.ndarray]:
    """
    This function processes a single blob to create a binary mask based on Otsu's thresholding.

    Parameters:
    ----------
    layer : np.ndarray
        The input image layer as a 2D NumPy array.

    blob : np.ndarray
        A single blob represented by its (y, x, radius) coordinates.

    Returns:
    -------
    Optional[np.ndarray]
        A binary image corresponding to the processed blob, or None if processing fails.
    """
    # Extract the coordinates and radius from the blob
    y, x, r = blob

    # Calculate the expanded radius and ensure blob stays within boundaries
    r = r * 1.5 * math.sqrt(2)
    y = np.clip(y, r, layer.shape[0] - r)
    x = np.clip(x, r, layer.shape[1] - r)

    # Create a circular disk mask based on the blob's location and radius
    rr, cc = draw.disk((y, x), r, shape=layer.shape)
    blob_mask = np.zeros(layer.shape, dtype=bool)
    blob_mask[rr, cc] = True

    # Find the bounding box around the mask (row and column ranges)
    rows = np.any(blob_mask, axis=1)
    cols = np.any(blob_mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Crop the blob image to the bounding box
    cropped_blob_mask = blob_mask[min_row : max_row + 1, min_col : max_col + 1]
    cropped_blob_image = (
        layer[min_row : max_row + 1, min_col : max_col + 1] * cropped_blob_mask
    )

    # Apply Otsu thresholding only on the cropped blob region
    non_zero_values = cropped_blob_image[cropped_blob_image > 0]
    if len(non_zero_values) == 0:
        return None

    threshold = filters.threshold_otsu(non_zero_values)
    cropped_binary_image = cropped_blob_image > threshold

    # Clean binary image by deleting artifacts and closing holes
    cleaned = clean_binary_image(cropped_binary_image, r)

    # Return None if cleaning fails
    if cleaned is None:
        return None

    # Create a full-sized label image and place the cropped labels back into it
    restored_image = np.zeros(layer.shape)
    restored_image[min_row : max_row + 1, min_col : max_col + 1] = cleaned

    return np.asarray(restored_image, dtype=bool)


def create_combined_binary_image(layer: np.ndarray, threshold_rel: float) -> np.ndarray:
    """
    This function creates a combined binary image from detected blobs using Laplacian of Gaussian.

    Parameters:
    ----------
    layer : np.ndarray
        The input image layer as a 2D NumPy array.
    threshold_rel : float
        Minimum intensity of peaks of Laplacian-of-Gaussian (LoG).
        This should have a value between 0 and 1.

    Returns:
    -------
    np.ndarray
        A combined binary image.
    """
    if not (0 <= threshold_rel <= 1):
        raise ValueError("threshold_rel must be between 0 and 1.")

    # Detect blobs using Laplacian of Gaussian (LoG)
    blobs_log = feature.blob_log(
        layer,
        max_sigma=15,
        num_sigma=10,
        overlap=0,
        threshold=None,
        threshold_rel=threshold_rel,
    )

    # Process each blob to create a labelled mask
    binaries = []
    for blob in blobs_log:
        result = process_blob(layer, blob)
        if result is not None:
            binaries.append(result)

    # Combine binaries into one single array
    if len(binaries) > 0:
        combined_array = np.bitwise_or.reduce(binaries)
    else:
        combined_array = np.zeros_like(layer, dtype=bool)

    return combined_array


def clean_binary_image(binary_image: np.ndarray, r: float) -> Optional[np.ndarray]:

    min_disk_area = 60
    max_disk_area = 2000

    # Remove small objects
    cleaned = morphology.remove_small_objects(binary_image, min_size=min_disk_area)

    # Remove small holes in the binary image
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_disk_area * 0.8)

    # Check minimum and maximum size
    area = np.sum(cleaned)
    if area < min_disk_area or area > max_disk_area:
        return None

    return cleaned


def extract_polygons(labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    This function extracts polygons (contours) from the labeled image.

    Parameters:
    ----------
    labels : np.ndarray
        The labeled image after watershed segmentation.

    Returns:
    -------
    Dict[int, np.ndarray]
        A dictionary where keys are labels and values are numpy arrays of polygon coordinates.
    """
    polygons = {}
    unique_labels = np.unique(labels)

    for label in unique_labels[unique_labels > 0]:

        # Create a mask for the current label
        mask = labels == label

        # Find contours (polygons) in the binary mask
        contours = measure.find_contours(mask, level=0.5)

        # If there are multiple contours, choose the largest one
        if contours:
            largest_contour = max(contours, key=len)
            polygons[label] = largest_contour

    return polygons


def segment_image(
    tif_path: Path, threshold_rel: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function segments an image to identify and extract ROI polygons.

    Parameters:
    ----------
    tif_path : Path
        Path to the TIFF image file.
    threshold_rel : float
        Minimum intensity of peaks of Laplacian-of-Gaussian (LoG).
        This should have a value between 0 and 1.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - A binary array with the segmented cells.
        - The processed image layer as a NumPy array.
    """
    # Read the image in its original form (unchanged)
    layer = np.asarray(Image.open(tif_path))

    # Eliminate rows and columns that are entirely zeros
    layer = crop(layer)
    layer = layer.astype(np.uint8)

    # Create a binary image of the layer with the segmented cells
    combined_array = create_combined_binary_image(layer, threshold_rel)

    return combined_array, layer
