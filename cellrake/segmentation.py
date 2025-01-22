"""
@author: Marc Canela
"""

import math
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import draw, feature, filters, measure, morphology, segmentation
from sklearn.cluster import KMeans
from tqdm import tqdm

from cellrake.utils import crop, crop_cell_large


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
    This function iterates over all `.tif` files in the given `image_folder,` extracting all the potential regions of interest (ROIs) that may be positive.
    The segmented layers and corresponding ROI data are stored in dictionaries with the image filename (without extension) as the key.

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

    # Iterate over each .tif file and pre-process the arrays
    print("Step 1 out of 4: Image preprocessing.")
    layers = {}
    combined_arrays = {}
    for tif_path in tqdm(
        list(image_folder.glob("*.tif")), desc="Preprocessing images", unit="image"
    ):
        tag = tif_path.stem
        combined_array, layer = segment_image(tif_path, threshold_rel)
        combined_arrays[tag] = combined_array
        layers[tag] = layer

    # Apply watershed
    print("Step 2 out of 4: Selection of images to watershed.")
    divide = {}
    non_divide = {}
    for tag, combined_array in tqdm(
        combined_arrays.items(), desc="Processing images", unit="image"
    ):
        global_user_input = global_watershed(layers[tag])
        if global_user_input == "n":
            non_divide[tag] = combined_array
        elif global_user_input == "y":
            divide[tag] = combined_array

    rois = {}

    if len(divide) != 0:
        print("Step 3 out of 4: Manual watershed.")
        for tag, combined_array in tqdm(
            divide.items(), desc="Processing images", unit="image"
        ):
            watershed_array = preprocess_watershed(combined_array, layers[tag])
            polygons = extract_polygons(watershed_array)
            rois[tag] = convert_to_roi(polygons, layers[tag])

    if len(non_divide) != 0:
        print("Step 4 out of 4: Automatic watershed.")
        for tag, combined_array in tqdm(
            non_divide.items(), desc="Processing images", unit="image"
        ):
            labels = measure.label(combined_array)
            polygons = extract_polygons(labels)
            rois[tag] = convert_to_roi(polygons, layers[tag])

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
    y = np.clip(y, r, layer.shape[0] - r)
    x = np.clip(x, r, layer.shape[1] - r)

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
    labels_cropped_list = clean_binary_image(cropped_binary_image, r)
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
            if isinstance(result, list):
                binaries.extend(result)

    if len(binaries) == 0:
        return np.zeros_like(layer, dtype=np.uint8)

    # Combine binaries into one single array
    combined_array = np.any(np.array(binaries), axis=0).astype(np.uint16)

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
    if np.max(labeled_mask) == 0:
        return []

    # Precompute the minimum disk area
    min_disk_area = np.sum(morphology.disk(4))

    # Calculate region areas using np.bincount
    region_areas = np.bincount(labeled_mask.ravel())[1:]
    max_area = region_areas.max()

    # Filter valid labels based on area threshold
    valid_labels = (
        np.where((region_areas >= 0.5 * max_area) & (region_areas >= min_disk_area))[0]
        + 1
    )

    # Skip if non valid labels
    if len(valid_labels) == 0:
        return []

    # Create and validate masks for each valid label
    masks = []

    for label in valid_labels:
        mask = labeled_mask == label

        # Check if the mask is too rectangular
        height, width = mask.shape
        ellipse_area = np.pi * (width / 2) * (height / 2)
        mask_area = np.sum(mask)
        rect_area = height * width
        area_difference = mask_area - ellipse_area

        if area_difference <= 0.5 * (rect_area - ellipse_area):
            masks.append(mask.astype(bool))

    return masks


def preprocess_watershed(
    combined_array: np.ndarray,
    layer: np.ndarray,
):

    labels = measure.label(combined_array)
    unique_labels = np.unique(labels)
    watershed_array = np.zeros_like(combined_array, dtype=np.uint16)

    # Compute background intensity
    background_mask = 1 - combined_array
    background_pixels = layer[background_mask == 1]
    nonzero_background_pixels = background_pixels[background_pixels != 0]
    mean_background = np.mean(nonzero_background_pixels)

    n = 1

    for mylabel in unique_labels:
        if mylabel == 0:
            continue

        cleaned = labels == mylabel

        # Compute the Euclidean distance transform of the binary image
        distance = distance_transform_edt(cleaned)
        distance = filters.gaussian(distance, sigma=1.0)

        # Calculate the cell radius from the maximum distance
        cell_radius = int(np.max(distance))
        if cell_radius == 0:
            continue

        # Check Signal-to-Background Ratio (SBR)
        intensities = layer[cleaned]
        blob_intensity = np.mean(intensities)
        sbr = blob_intensity / (mean_background if mean_background != 0 else 1)
        if sbr > 1.5:
            watershed_array[cleaned] = n
            n += 1
            continue

        # Create a disk for footprint
        disk = morphology.disk(int(cell_radius))

        # Identify local maxima in the distance map for marker generation
        actual_area = np.sum(cleaned)
        single_area = np.sum(disk)
        predicted_peaks = actual_area / single_area
        if predicted_peaks < 1.5:
            watershed_array[cleaned] = n
            n += 1
            continue

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
            watershed_array[cleaned] = n
            n += 1
            continue

        # Extract coordinates
        contours = measure.find_contours(cleaned, level=0.5)
        largest_contour = max(contours, key=lambda c: len(c))
        polygons = {1: largest_contour}
        roi = convert_to_roi(polygons, layer)
        roi_info = roi["roi_1"]

        # User validation before segmentation
        user_input_value = user_input_watershed(cleaned, layer, roi_info)

        if user_input_value == 1:
            watershed_array[cleaned] = n
            n += 1
            continue

        # Perform watershed
        coords = np.column_stack(np.nonzero(cleaned))
        kmeans = KMeans(n_clusters=user_input_value, random_state=42)
        kmeans.fit(coords)
        markers = np.zeros_like(cleaned, dtype=int)
        for i, coord in enumerate(coords):
            markers[tuple(coord)] = (
                kmeans.labels_[i] + 1
            )  # Assign cluster IDs as marker labels
        watershed_labels = segmentation.watershed(-distance, markers, mask=cleaned)

        # Adjust the obtained numbers
        unique_watershed_labels = np.unique(watershed_labels)
        mapping = {}
        for watershed_label in unique_watershed_labels:
            if watershed_label != 0:
                mapping[watershed_label] = n
                n += 1

        new_labels = np.vectorize(lambda x: mapping.get(x, 0))(watershed_labels)

        watershed_array = np.where(new_labels == 0, watershed_array, new_labels)

    return watershed_array


def user_input_watershed(
    cleaned: np.ndarray, layer: np.ndarray, roi_info: Dict[str, Any]
):

    print("Enter the number of ROIs to be segmented in.")
    print("Please enter an integer different from zero (1, 2...):\n")

    x_coords, y_coords = roi_info["x"], roi_info["y"]

    # Set up the plot with four subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # Full image with ROI highlighted
    axes[0].imshow(layer, cmap="viridis")
    axes[0].plot(x_coords, y_coords, "b-", linewidth=1)
    axes[0].axis("off")  # Hide the axis

    # Full image without ROI highlighted
    axes[1].imshow(layer, cmap="viridis")
    axes[1].axis("off")  # Hide the axis

    # Cropped image with padding, ROI highlighted
    layer_cropped_small, x_coords_cropped, y_coords_cropped = crop_cell_large(
        layer, x_coords, y_coords, padding=120
    )
    axes[2].imshow(layer_cropped_small, cmap="viridis")
    axes[2].plot(x_coords_cropped, y_coords_cropped, "b-", linewidth=1)
    axes[2].axis("off")  # Hide the axis

    # Cropped image without ROI highlighted
    axes[3].imshow(layer_cropped_small, cmap="viridis")
    axes[3].axis("off")  # Hide the axis

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    # Ask for user input
    while True:
        try:
            user_input_value = int(
                input("Please enter an integer different from zero (1, 2...):\n")
            )
            if user_input_value != 0:
                break
            else:
                print("Invalid input. Zero is not allowed.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    plt.close(fig)

    return user_input_value


def global_watershed(layer: np.ndarray):

    print("Do you want to apply watershed in this image?")
    print("Please enter yes (y) or no (n):\n")

    # Set up the plot with four subplots
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))

    # Full image without ROI highlighted
    ax.imshow(layer, cmap="viridis")
    ax.axis("off")  # Hide the axis

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    # Ask for user input
    while True:
        try:
            user_input_value = input("Please enter yes (y) or no (n):")
            if user_input_value in ["y", "n"]:
                break
            else:
                print("Invalid input. Please enter yes (y) or no (n).")
        except ValueError:
            print("Invalid input. Please enter yes (y) or no (n).")

    plt.close(fig)

    return user_input_value


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

    return combined_array, layer
