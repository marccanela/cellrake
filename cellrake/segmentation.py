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

# ================================================================================
# UTILITY FUNCTIONS FOR IMAGE SEGMENTATION
# ================================================================================


def clean_binary_image(
    binary_image: np.ndarray, min_area: int, max_area: int, hole_fill_ratio: float
) -> Optional[np.ndarray]:
    """
    Clean a binary image by removing small objects and holes.

    Parameters
    ----------
    binary_image : np.ndarray
        Input binary image.
    min_area : int
        Minimum object area in pixels.
    max_area : int
        Maximum object area in pixels.
    hole_fill_ratio : float
        Ratio for hole filling threshold relative to min_area.

    Returns
    -------
    Optional[np.ndarray]
        Cleaned binary image or None if size is invalid.
    """
    # Remove small objects
    cleaned = morphology.remove_small_objects(binary_image, min_size=min_area)

    # Remove small holes in the binary image
    cleaned = morphology.remove_small_holes(
        cleaned, area_threshold=int(min_area * hole_fill_ratio)
    )

    # Check minimum and maximum size
    area = np.sum(cleaned)
    if area < min_area or area > max_area:
        return None

    return cleaned


def process_blob(
    layer: np.ndarray,
    blob: np.ndarray,
    radius_expansion: float,
    min_area: int,
    max_area: int,
    hole_fill_ratio: float,
) -> Optional[np.ndarray]:
    """
    Process a detected blob to create a binary mask.

    Parameters
    ----------
    layer : np.ndarray
        Input image array.
    blob : np.ndarray
        Blob coordinates as [y, x, radius].
    radius_expansion : float
        Factor to expand blob radius for mask creation.
    min_area : int
        Minimum object area for cleaning.
    max_area : int
        Maximum object area for cleaning.
    hole_fill_ratio : float
        Ratio for hole filling threshold.

    Returns
    -------
    Optional[np.ndarray]
        Binary mask or None if processing fails.
    """
    # Extract the coordinates and radius from the blob
    y, x, r = blob

    # Calculate the expanded radius and ensure blob stays within boundaries
    r = r * radius_expansion
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
    cleaned = clean_binary_image(
        cropped_binary_image, min_area, max_area, hole_fill_ratio
    )

    # Return None if cleaning fails
    if cleaned is None:
        return None

    # Create a full-sized label image and place the cropped labels back into it
    restored_image = np.zeros(layer.shape)
    restored_image[min_row : max_row + 1, min_col : max_col + 1] = cleaned

    return np.asarray(restored_image, dtype=bool)


def create_combined_binary_image(
    layer: np.ndarray,
    threshold_rel: float,
    max_sigma: int,
    num_sigma: int,
    overlap: float,
    radius_expansion: float,
    min_area: int,
    max_area: int,
    hole_fill_ratio: float,
) -> np.ndarray:
    """
    Create a binary segmentation mask using LoG blob detection.

    Parameters
    ----------
    layer : np.ndarray
        Input image array.
    threshold_rel : float
        Detection threshold (0-1).
    max_sigma : int
        Maximum standard deviation for LoG filter.
    num_sigma : int
        Number of intermediate values between min and max sigma.
    overlap : float
        Minimum distance between blobs as fraction of blob radius.
    radius_expansion : float
        Factor to expand blob radius for mask creation.
    min_area : int
        Minimum object area for cleaning.
    max_area : int
        Maximum object area for cleaning.
    hole_fill_ratio : float
        Ratio for hole filling threshold.

    Returns
    -------
    np.ndarray
        Binary segmentation mask.
    """
    if not (0 <= threshold_rel <= 1):
        raise ValueError("threshold_rel must be between 0 and 1.")

    # Detect blobs using Laplacian of Gaussian (LoG)
    blobs_log = feature.blob_log(
        layer,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        overlap=overlap,
        threshold=None,
        threshold_rel=threshold_rel,
    )

    # Process each blob to create a labelled mask
    binaries = []
    for blob in blobs_log:
        result = process_blob(
            layer, blob, radius_expansion, min_area, max_area, hole_fill_ratio
        )
        if result is not None:
            binaries.append(result)

    # Combine binaries into one single array
    if len(binaries) > 0:
        combined_array = np.bitwise_or.reduce(binaries)
    else:
        combined_array = np.zeros_like(layer, dtype=bool)

    return combined_array


def extract_polygons(labels: np.ndarray, contour_level: float) -> Dict[int, np.ndarray]:
    """
    Extract polygon contours from labeled regions.

    Parameters
    ----------
    labels : np.ndarray
        Labeled image with integer values.
    contour_level : float
        Level for contour extraction.

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping label IDs to polygon contours.
    """
    polygons = {}
    unique_labels = np.unique(labels)

    for label in unique_labels[unique_labels > 0]:

        # Create a mask for the current label
        mask = labels == label

        # Find contours (polygons) in the binary mask
        contours = measure.find_contours(mask, level=contour_level)

        # If there are multiple contours, choose the largest one
        if contours:
            largest_contour = max(contours, key=len)
            polygons[label] = largest_contour

    return polygons


def convert_to_roi(
    polygons: Dict[int, np.ndarray], layer: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert polygon contours to ROI format with coordinate clipping.

    Parameters
    ----------
    polygons : Dict[int, np.ndarray]
        Dictionary mapping label IDs to contour arrays.
    layer : np.ndarray
        2D image array used to determine coordinate bounds.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping ROI identifiers to coordinate dictionaries with 'x' and 'y' keys.
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


# ================================================================================
# CORE FUNCTION FOR IMAGE SEGMENTATION
# ================================================================================


def segment_image(
    tif_path: Path,
    threshold_rel: float,
    max_sigma: int,
    num_sigma: int,
    overlap: float,
    radius_expansion: float,
    min_area: int,
    max_area: int,
    hole_fill_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a TIFF image to identify cellular regions.

    Parameters
    ----------
    tif_path : Path
        Path to the TIFF image file.
    threshold_rel : float
        Detection threshold (0-1).
    max_sigma : int
        Maximum standard deviation for LoG filter.
    num_sigma : int
        Number of intermediate values for LoG filter.
    overlap : float
        Minimum distance between blobs as fraction of radius.
    radius_expansion : float
        Factor to expand blob radius for mask creation.
    min_area : int
        Minimum object area for cleaning.
    max_area : int
        Maximum object area for cleaning.
    hole_fill_ratio : float
        Ratio for hole filling threshold.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Binary segmentation mask and preprocessed image.
    """
    # Read the image in its original form (unchanged)
    layer = np.asarray(Image.open(tif_path))

    # Eliminate rows and columns that are entirely zeros
    layer = crop(layer)
    layer = layer.astype(np.uint8)

    # Create a binary image of the layer with the segmented cells
    combined_array = create_combined_binary_image(
        layer,
        threshold_rel,
        max_sigma,
        num_sigma,
        overlap,
        radius_expansion,
        min_area,
        max_area,
        hole_fill_ratio,
    )

    return combined_array, layer


# ================================================================================
# WORKFLOW FUNCTION FOR ITERATING OVER IMAGES IN A FOLDER
# ================================================================================


def iterate_segmentation(
    image_folder: Path,
    threshold_rel: float,
    max_sigma: int = 15,
    num_sigma: int = 10,
    overlap: float = 0,
    radius_expansion: float = 1.5 * math.sqrt(2),
    min_area: int = 60,
    max_area: int = 2000,
    hole_fill_ratio: float = 0.8,
    contour_level: float = 0.5,
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, np.ndarray]]:
    """
    Process all TIFF images in a folder to extract ROIs.

    Parameters
    ----------
    image_folder : Path
        Path to the folder containing TIFF images.
    threshold_rel : float
        Threshold for blob detection (0-1).
    max_sigma : int
        Maximum standard deviation for LoG filter.
    num_sigma : int
        Number of intermediate values for LoG filter.
    overlap : float
        Minimum distance between blobs as fraction of radius.
    radius_expansion : float
        Factor to expand blob radius for mask creation.
    min_area : int
        Minimum object area for cleaning.
    max_area : int
        Maximum object area for cleaning.
    hole_fill_ratio : float
        Ratio for hole filling threshold.
    contour_level : float
        Level for contour extraction.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, np.ndarray]]
        ROI dictionaries and processed image layers for each image.
    """
    rois = {}
    layers = {}

    for tif_path in tqdm(
        list(image_folder.glob("*.tif")), desc="Preprocessing images", unit="image"
    ):
        tag = tif_path.stem
        combined_array, layer = segment_image(
            tif_path,
            threshold_rel,
            max_sigma,
            num_sigma,
            overlap,
            radius_expansion,
            min_area,
            max_area,
            hole_fill_ratio,
        )
        labels = measure.label(combined_array)
        polygons = extract_polygons(labels, contour_level)

        rois[tag] = convert_to_roi(polygons, layer)
        layers[tag] = layer

    return rois, layers


# ================================================================================
# COMPLEMENTARY FUNCTION TO EXPORT ROIS
# ================================================================================


def export_rois(
    project_folder: Path, rois: Dict[str, Dict[str, Dict[str, np.ndarray]]]
) -> None:
    """
    Export ROI data to pickle files.

    Parameters
    ----------
    project_folder : Path
        Path to the project directory.
    rois : Dict[str, Dict[str, Dict[str, np.ndarray]]]
        ROI data to export.

    Returns
    -------
    None
    """
    # Export each ROI dictionary to a .pkl file
    for tag, rois_dict in rois.items():
        with open(str((project_folder / "rois_raw") / f"{tag}.pkl"), "wb") as file:
            pkl.dump(rois_dict, file)
