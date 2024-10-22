"""
@author: Marc Canela
"""

import math
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
    polygon : dict[int, np.ndarray]
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
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    This function iterates over all `.tif` files in the given `image_folder`, applies a pre-trained StarDist model
    to segment the images, and extracts ROIs. The segmented layers and corresponding ROI data are stored in dictionaries
    with the image filename (without extension) as the key.

    Parameters:
    ----------
    image_folder : pathlib.Path
        A Path object pointing to the folder containing the `.tif` images to be segmented.
    threshold_rel : float
        Minimum intensity of peaks of Laplacian-of-Gaussian (LoG).
        This should have a value between 0 and 1.

    Returns:
    -------
    tuple[dict[str, dict], dict[str, numpy.ndarray]]
        A tuple containing:
        - `rois`: A dictionary where keys are image filenames and values are dictionaries of ROI data.
        - `layers`: A dictionary where keys are image filenames and values are the corresponding segmented layers as NumPy arrays.
    """
    rois = {}
    layers = {}

    # Iterate over each .tif file and segment the image
    for tif_path in tqdm(
        list(image_folder.glob("*.tif")), desc="Segmenting images", unit="image"
    ):

        # Segment the image
        polygons, layer = segment_image(tif_path, threshold_rel)

        # Store the results in the dictionaries
        tag = tif_path.stem
        rois[tag] = convert_to_roi(polygons, layer)
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
    # Export each ROI dictionary to a .pkl file
    for tag, rois_dict in rois.items():
        with open(str((project_folder / "rois_raw") / f"{tag}.pkl"), "wb") as file:
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
    list
        A list of binary images corresponding to the processed blob.
    """
    # Extract the coordinates and radius from the blob
    y, x, r = blob

    # Calculate the expanded radius
    r = r * 1.5 * math.sqrt(2)

    # Ensure the blob stays within the image boundaries
    y = min(max(y, r), layer.shape[0] - r)
    x = min(max(x, r), layer.shape[1] - r)

    # Create a circular disk mask based on the blob's location and radius
    rr, cc = draw.disk((y, x), r, shape=layer.shape)
    blob_mask = np.zeros(layer.shape, dtype=bool)
    blob_mask[rr, cc] = True

    # Crop the blob_image to the bounding box of the mask
    min_row, max_row = np.where(np.any(blob_mask, axis=1))[0][[0, -1]]
    min_col, max_col = np.where(np.any(blob_mask, axis=0))[0][[0, -1]]

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
    cleaned_cropped_list = clean_binary_image(cropped_binary_image, r)
    if len(cleaned_cropped_list) == 0:
        return None

    # Apply watershed segmentation of identify cells
    labels_cropped_list = [
        labels_cropped
        for cleaned_cropped in cleaned_cropped_list
        if (labels_cropped := apply_watershed_segmentation(cleaned_cropped)) is not None
    ]
    if len(labels_cropped_list) == 0:
        return None

    # Create a full-sized label image and place the cropped labels back into it
    labels_list = [np.zeros(layer.shape, dtype=np.uint16) for _ in labels_cropped_list]
    for i, labels_cropped in enumerate(labels_cropped_list):
        labels_list[i][min_row : max_row + 1, min_col : max_col + 1] = labels_cropped

    return labels_list


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
            if isinstance(result, list):
                binaries.extend(result)

    if len(binaries) == 0:
        return np.zeros_like(layer, dtype=np.uint8)

    # Combine all binary masks using logical OR operation
    combined_array = np.zeros_like(binaries[0], dtype=np.uint16)
    next_label = 1

    # Collect all labels and their sizes from all segment arrays
    label_sizes = {}

    for seg_array in binaries:
        unique_labels = np.unique(seg_array[seg_array != 0])
        for label in unique_labels:
            if label not in label_sizes:
                mask = seg_array == label
                size = np.sum(mask)
                label_sizes[label] = size

    # Sort labels by their size (smallest first)
    sorted_labels = sorted(label_sizes.keys(), key=lambda l: label_sizes[l])

    # Apply the labels to the combined_array in sorted order
    for label in sorted_labels:
        for seg_array in binaries:
            mask = seg_array == label
            combined_array[mask] = next_label
            next_label += 1

    return combined_array


def clean_binary_image(binary_image: np.ndarray, r: float) -> np.ndarray:
    """
    This function cleans the binary image by removing small holes and retaining
    all masks that are larger than 50% of the area of the largest mask.

    Parameters:
    ----------
    binary_image : np.ndarray
        The input binary image.
    r : float
        The radius used to create a disk for morphological operations.

    Returns:
    -------
    list
        A list of binary masks for each valid label that are not too rectangular.
    """
    # Remove small holes in the binary image
    cleaned = morphology.remove_small_holes(
        binary_image, area_threshold=np.sum(morphology.disk(int(r / 1.5)))
    )

    # Label connected regions in the cleaned binary image
    labeled_mask = measure.label(cleaned)

    # Skip if there's only the background
    if len(np.unique(labeled_mask)) == 1:
        return []

    # Calculate the properties of each labeled region
    region_props = measure.regionprops(labeled_mask)

    # Get the areas of the regions
    areas = np.array([region.area for region in region_props])

    # Identify the largest area
    max_area = areas.max()

    # Disk area for morphological operations
    min_disk_area = np.sum(morphology.disk(4))

    # Filter valid labels
    valid_labels = [
        region.label
        for region in region_props
        if region.area >= 0.5 * max_area and region.area >= min_disk_area
    ]

    # Create and validate masks for each valid label
    masks = []

    for label in valid_labels:
        mask = labeled_mask == label

        # Check if the mask is too rectangular
        height, width = mask.shape
        ellipse_area = np.pi * (width / 2) * (height / 2)
        mask_area = np.sum(mask)
        rect_area = height * width
        diff = mask_area - ellipse_area

        if diff <= 0.5 * (rect_area - ellipse_area):
            masks.append(mask.astype(bool))

    return masks


def apply_watershed_segmentation(cleaned: np.ndarray) -> np.ndarray:
    """
    This function applies watershed segmentation to the cleaned binary image.

    Parameters:
    ----------
    cleaned : np.ndarray
        The cleaned binary image.

    Returns:
    -------
    np.ndarray
        The labeled image after applying watershed segmentation.
    """
    # Compute the Euclidean distance transform of the binary image
    distance = distance_transform_edt(cleaned)
    distance = filters.gaussian(distance, sigma=1.0)

    # Calculate the cell radius from the maximum distance
    cell_radius = int(np.max(distance))
    if cell_radius == 0:
        return None

    # Create a disk for footprint
    disk = morphology.disk(int(cell_radius))

    # Identify local maxima in the distance map for marker generation
    actual_area = np.sum(cleaned)
    single_area = np.sum(disk)
    predicted_peaks = actual_area / single_area
    if predicted_peaks < 1.5:
        return cleaned.astype(int)

    predicted_peaks = int(predicted_peaks) + 1

    coords = feature.peak_local_max(
        distance,
        min_distance=cell_radius,
        threshold_rel=0.6,
        footprint=disk,
        labels=measure.label(cleaned),
        num_peaks_per_label=predicted_peaks,
    )

    if len(coords) == 1:
        return cleaned.astype(int)

    # Create a mask for the local maxima
    mask = np.zeros(cleaned.shape, dtype=bool)
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

        # If there are multiple contours, choose the largest one
        if contours:
            largest_contour = max(contours, key=lambda c: len(c))
            polygons[label] = largest_contour

    return polygons


def segment_image(
    tif_path: Path, threshold_rel: float
) -> Tuple[Dict[int, List], np.ndarray]:
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
    Tuple[Dict[int, List], np.ndarray]
        A tuple containing:
        - A dictionary where keys are labels and values are lists of polygon coordinates.
        - The processed image layer as a NumPy array.
    """
    # Read the image in its original form (unchanged)
    layer = np.asarray(Image.open(tif_path))

    # Eliminate rows and columns that are entirely zeros
    layer = crop(layer)
    layer = layer.astype(np.uint8)

    # Create a binary image of the layer with the segmented cells
    combined_array = create_combined_binary_image(layer, threshold_rel)

    # Extract the coordinates of the segmented cells
    polygons = extract_polygons(combined_array)

    return polygons, layer
