"""
@author: Marc Canela
"""

import math
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from csbdeep.utils import normalize
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import draw, feature, filters, measure, morphology, segmentation
from tqdm import tqdm

from cellrake.utils import crop


def convert_to_roi(
    polygons: Dict[int, List], layer: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    This function extracts the coordinates of the polygons and converts them into ROIs.
    It clips the coordinates to ensure they lie within the bounds of the given image layer.

    Parameters:
    ----------
    polygon : shapely.geometry.Polygon
        A Polygon object containing the coordinates of the regions of interest.

    layer : numpy.ndarray
        A 2D NumPy array representing the image layer. The shape of the array should be
        (height, width).

    Returns:
    -------
    dict
        A dictionary where each key is a string identifier for an ROI ("roi_0", "roi_1", etc.),
        and each value is another dictionary with 'x' and 'y' keys containing the clipped
        x and y coordinates of the ROI.
    """
    # Initialize an empty dictionary to store ROIs
    rois_dict = {}

    # Extract dimensions of the layer
    layer_height, layer_width = layer.shape

    # Iterate
    for n, (label, contours) in enumerate(polygons.items(), start=1):
        for i, contour in enumerate(contours):
            roi_x = contour[:, 1]
            roi_y = contour[:, 0]

            # Clip the coordinates to be within the bounds of the layer
            roi_y = np.clip(roi_y, 0, layer_height - 1)
            roi_x = np.clip(roi_x, 0, layer_width - 1)

            # Define the key for the ROI (e.g., "roi_1_0" for the first contour of ROI 1).
            roi_key = f"roi_{n}_{i}"

            # Store the x and y coordinates in the dictionary.
            rois_dict[roi_key] = {"x": roi_x, "y": roi_y}

    return rois_dict


def iterate_segmentation(
    image_folder: Path,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    This function iterates over all `.tif` files in the given `image_folder`, applies a pre-trained StarDist model
    to segment the images, and extracts ROIs. The segmented layers and corresponding ROI data are stored in dictionaries
    with the image filename (without extension) as the key.

    Parameters:
    ----------
    image_folder : pathlib.Path
        A Path object pointing to the folder containing the `.tif` images to be segmented.

    Returns:
    -------
    tuple[dict[str, dict], dict[str, numpy.ndarray]]
        A tuple containing:
        - `rois`: A dictionary where keys are image filenames and values are dictionaries of ROI data.
        - `layers`: A dictionary where keys are image filenames and values are the corresponding segmented layers as NumPy arrays.
    """
    rois = {}
    layers = {}

    # Get a list of all .tif files in the folder
    tif_paths = list(image_folder.glob("*.tif"))

    # Iterate over each .tif file and segment the image
    for tif_path in tqdm(tif_paths, desc="Segmenting images", unit="image"):
        tag = tif_path.stem

        # Segment the image
        polygons, layer = segment_image(tif_path)

        # Extract ROIs
        rois_dict = convert_to_roi(polygons, layer)

        # Store the results in the dictionaries
        rois[tag] = rois_dict
        layers[tag] = layer

    return rois, layers


def export_rois(project_folder: Path, rois: Dict[str, Dict]) -> None:
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
    # Define the path to the 'rois_raw' folder
    raw_folder = project_folder / "rois_raw"

    # Export each ROI dictionary to a .pkl file
    for tag, rois_dict in rois.items():
        pkl_path = raw_folder / f"{tag}.pkl"
        with open(str(pkl_path), "wb") as file:
            pkl.dump(rois_dict, file)


def process_blob(layer: np.ndarray, blob: np.ndarray) -> np.ndarray:
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
    np.ndarray
        A binary image corresponding to the processed blob.
    """
    # Extract the coordinates and radius from the blob
    y, x, r = blob

    # Ensure the blob stays within the image boundaries
    y = min(max(y, r), layer.shape[0] - r)
    x = min(max(x, r), layer.shape[1] - r)

    # Create a circular disk mask based on the blob's location and radius
    rr, cc = draw.disk((y, x), r, shape=layer.shape)
    mask = np.zeros(layer.shape, dtype=bool)
    mask[rr, cc] = True

    # Apply the mask to the image, keeping only the blob area
    blob_image = np.where(mask, layer, 0)

    # Create binary by applying the Otsu threshold for the blob area
    non_zero_values = blob_image[blob_image > 0]
    threshold = filters.threshold_otsu(non_zero_values)
    binary_image = blob_image > threshold

    return binary_image


def create_combined_binary_image(layer: np.ndarray) -> np.ndarray:
    """
    This function creates a combined binary image from detected blobs using Laplacian of Gaussian.

    Parameters:
    ----------
    layer : np.ndarray
        The input image layer as a 2D NumPy array.

    Returns:
    -------
    np.ndarray
        A combined binary image.
    """
    # Detect blobs using Laplacian of Gaussian (LoG)
    blobs_log = feature.blob_log(layer, max_sigma=15, num_sigma=10, threshold=0.05)

    # Calculate the radius of the blobs
    blobs_log[:, 2] = blobs_log[:, 2] * 1.5 * math.sqrt(2)

    # Process each blob to create a binary mask
    binaries = []
    for blob in blobs_log:
        binary_image = process_blob(layer, blob)
        binaries.append(binary_image)

    # Combine all binary masks using logical OR operation
    combined_binary_image = np.logical_or.reduce(binaries)
    return combined_binary_image


def clean_binary_image(binary_image: np.ndarray) -> np.ndarray:
    """
    This function cleans the binary image by removing small objects and holes.

    Parameters:
    ----------
    binary_image : np.ndarray
        The input binary image.

    Returns:
    -------
    np.ndarray
        The cleaned binary image.
    """
    # Remove small holes in the binary image
    remove_holes = morphology.remove_small_holes(binary_image, area_threshold=100)

    # Remove small objects from the binary image
    remove_particles = morphology.remove_small_objects(remove_holes, min_size=25)

    # Perform binary closing to smooth the image
    cleaned = morphology.binary_closing(remove_particles, morphology.disk(3))
    return cleaned


def apply_watershed_segmentation(layer: np.ndarray, cleaned: np.ndarray) -> np.ndarray:
    """
    This function applies watershed segmentation to the cleaned binary image.

    Parameters:
    ----------
    layer : np.ndarray
        The original image layer as a 2D NumPy array.

    cleaned : np.ndarray
        The cleaned binary image.

    Returns:
    -------
    np.ndarray
        The labeled image after applying watershed segmentation.
    """
    # Compute the Euclidean distance transform of the binary image
    distance = distance_transform_edt(cleaned)

    # Calculate the cell radius from the maximum distance
    cell_radius = int(np.max(distance)) // 2

    # Identify local maxima in the distance map for marker generation
    coords = feature.peak_local_max(
        distance,
        min_distance=cell_radius,
        footprint=morphology.disk(cell_radius),
        labels=measure.label(cleaned),
        num_peaks_per_label=5,
    )

    # Create a mask for the local maxima
    mask = np.zeros(layer.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Label the local maxima to generate markers for watershed
    markers, _ = measure.label(mask, return_num=True)

    # Apply the watershed algorithm using the distance map and markers
    labels = segmentation.watershed(
        -distance, markers, mask=cleaned, watershed_line=True, compactness=1
    )

    return labels


def extract_polygons(labels: np.ndarray) -> Dict[int, List]:
    """
    This function extracts polygons (contours) from the labeled image.

    Parameters:
    ----------
    labels : np.ndarray
        The labeled image after watershed segmentation.

    Returns:
    -------
    Dict[int, List]
        A dictionary where keys are labels and values are lists of polygon coordinates.
    """
    polygons = {}
    for label in np.unique(labels):
        # Skip background
        if label == 0:
            continue

        # Create a mask for the current label
        mask = labels == label

        # Find contours (polygons) in the binary mask
        contours = measure.find_contours(mask, level=0.5)

        # Store the contours in the dictionary
        polygons[label] = contours

    return polygons


def segment_image(tif_path: Path) -> Tuple[Dict[int, List], np.ndarray]:
    """
    This function segments an image to identify and extract ROI polygons.

    Parameters:
    ----------
    tif_path : Path
        Path to the TIFF image file.

    Returns:
    -------
    Tuple[Dict[int, List], np.ndarray]
        A tuple containing:
        - A dictionary where keys are labels and values are lists of polygon coordinates.
        - The processed image layer as a NumPy array.
    """
    # Read the image in its original form (unchanged)
    layer = np.asarray(Image.open(tif_path))

    # Eliminate rows and columns that are entirely zeros
    layer = crop(layer)

    # Create a binary image of the layer with the segmented cells
    binary_image = create_combined_binary_image(layer)

    # Clean binary image by deleting artifacts and closing holes
    cleaned = clean_binary_image(binary_image)

    # Apply watershed segmentation of identify cells
    labels = apply_watershed_segmentation(layer, cleaned)

    # Extract the coordinates of the segmented cells
    polygons = extract_polygons(labels)

    return polygons, layer
