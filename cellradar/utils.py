"""
Created on Sat Aug 10 11:12:16 2024
@author: mcanela
"""

import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops


def get_layer(image):
    """
    This function checks if the image is 2D or 3D. If the image is 2D,
    it directly returns that layer. If the image is 3D, it finds the
    first layer that contains any non-zero values and returns it. If all
    layers are empty (i.e., all values are zero), the function returns `None`.

    Parameters:
    ----------
    image : numpy.ndarray
        A 2D or 3D NumPy array representing the image. The shape of the array
        should be (height, width) for 2D or (height, width, num_layers) for 3D.

    Returns:
    -------
    numpy.ndarray or None
        The single 2D layer if the image is 2D, the first non-empty layer
        if the image is 3D, or `None` if all layers are empty.
    """
    if image.ndim == 2:
        return image

    layer_sums = np.sum(image, axis=(0, 1))
    non_zero_layer_indices = np.nonzero(layer_sums)[0]

    if non_zero_layer_indices.size > 0:
        return image[:, :, non_zero_layer_indices[0]]
    else:
        return None


def build_project(image_folder):

    project_folder = image_folder.parent / f"{image_folder.stem}_analysis"
    project_folder.mkdir(parents=True, exist_ok=True)

    rois_raw_folder = project_folder / "rois_raw"
    rois_raw_folder.mkdir(parents=True, exist_ok=True)

    rois_predicted_folder = project_folder / "rois_processed"
    rois_predicted_folder.mkdir(parents=True, exist_ok=True)

    labelledimg_folder = project_folder / "labelled_images"
    labelledimg_folder.mkdir(parents=True, exist_ok=True)

    return project_folder


def change_names(folder_path, number: int = 300):

    # Iterate over all files in the folder
    for filename in folder_path.iterdir():
        if filename.is_file() and filename.suffix == ".nd2":
            parts = filename.stem.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                try:
                    # Extract and modify the second element
                    original_number = int(parts[1])
                    new_number = original_number + number
                    parts[1] = str(new_number)

                    # Create the new filename
                    new_filename = "_".join(parts) + filename.suffix

                    # Construct the full file paths
                    new_file_path = filename.with_name(new_filename)

                    # Rename the file
                    filename.rename(new_file_path)

                except Exception as e:
                    print(f"Error processing {filename.name}: {e}")


def get_cell_mask(layer, coordinates):
    """
    Create a binary mask from the given Region of Interest (ROI) coordinates.
    This function creates a binary mask where the specified ROI is filled with
    ones (1) and the rest of the mask is zeros (0). The mask is created
    based on the coordinates provided, which define the polygonal region
    to be filled.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer. The shape of the array
        (height, width) determines the dimensions of the resulting mask.

    coordinates : list of numpy.ndarray
        A list where each element is a NumPy array of shape (N, 2)
        representing the vertices of a polygon. These coordinates specify
        the regions to be filled in the mask.

    Returns:
    -------
    numpy.ndarray
        A binary mask of the same shape as `layer`. Pixels inside the defined
        regions are set to `1`, and all other pixels are set to `0`.
    """
    mask = np.zeros(layer.shape, dtype=np.uint8)
    cv2.fillPoly(mask, coordinates, 1)
    return mask


def crop_cell(layer, x_coords, y_coords):
    """
    Crop a rectangular region from an image layer based on given coordinates.
    This function extracts a rectangular subregion from the provided image
    `layer`. The rectangle is defined by the minimum and maximum x and y
    coordinates. The subregion is extracted by slicing the `layer` array.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which the region is
        to be cropped. The shape of the array should be (height, width).

    x_coords : list or numpy.ndarray
        A list or array of x-coordinates defining the boundary of the
        rectangular region to crop. These coordinates determine the horizontal
        extent of the region.

    y_coords : list or numpy.ndarray
        A list or array of y-coordinates defining the boundary of the
        rectangular region to crop. These coordinates determine the vertical
        extent of the region.

    Returns:
    -------
    numpy.ndarray
        A 2D NumPy array representing the cropped region of the image layer.
    """
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return layer[y_min : y_max + 1, x_min : x_max + 1]


def convert_to_roi(polygon, layer):

    polygon_coord = polygon["coord"]
    rois_dict = {}

    layer_height, layer_width = layer.shape

    for n in range(polygon_coord.shape[0]):
        roi_y = polygon_coord[n, 0, :]
        roi_x = polygon_coord[n, 1, :]

        # Clip the coordinates to be within the bounds of the layer
        roi_y = np.clip(roi_y, 0, layer_height - 1)
        roi_x = np.clip(roi_x, 0, layer_width - 1)

        rois_dict[f"roi_{n}"] = {"x": roi_x, "y": roi_y}

    return rois_dict


def extract_roi_stats(layer, roi_info):
    """
    This function processes a given ROI within an image layer by calculating
    several statistical and texture features.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which the ROI is
        extracted. The shape of the array should be (height, width).

    roi_info : dict
        A dictionary containing the coordinates of the ROI. It should have, at
        least, the following keys:
        - "x": A list or array of x-coordinates of the ROI vertices.
        - "y": A list or array of y-coordinates of the ROI vertices.

    Returns:
    -------
    cell_mask: numpy.ndarray
        A binary 2D NumPy array with the same size as the layer, containing 1
        at the position of the ROi, else 0.

    stats_dict: dict
        Dictionary of statistics and texture features.
    """
    x_coords, y_coords = roi_info["x"], roi_info["y"]
    coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)

    # Create cell mask
    cell_mask = get_cell_mask(layer, coordinates)

    # Crop to the bounding box of the ROI to reduce computation
    layer_cropped = crop_cell(layer, x_coords, y_coords)
    cell_mask_cropped = crop_cell(cell_mask, x_coords, y_coords)

    background_mask_cropped = 1 - cell_mask_cropped

    cell_pixels = layer[cell_mask == 1]
    cell_pixels_mean = np.mean(cell_pixels)
    background_pixels = layer_cropped[background_mask_cropped == 1]
    background_pixels_mean = np.mean(background_pixels)

    layer_cropped_masked = layer_cropped * cell_mask_cropped
    layer_masked = layer * cell_mask

    # Calculate LBP
    lbp = local_binary_pattern(layer_cropped_masked, P=8, R=1, method="uniform")

    # Calculate GLCM features
    glcm = graycomatrix(
        layer_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )

    # Calculate regionprops
    prop = regionprops(label(cell_mask_cropped))[0]

    # HOG features
    hog_descriptor = cv2.HOGDescriptor()
    h = hog_descriptor.compute(layer_masked).flatten()  # Flatten the HOG descriptor

    stats_dict = {
        "mean_intensity": cell_pixels_mean,
        "median_intensity": np.median(cell_pixels),
        "sd_intensity": np.std(cell_pixels),
        "min_intensity": np.min(cell_pixels),
        "max_intensity": np.max(cell_pixels),
        # 'mean_background': background_pixels_mean,
        # 'median_background':  np.median(background_pixels),
        # 'sd_background': np.std(background_pixels),
        # 'min_background': np.min(background_pixels),
        # 'max_background': np.max(background_pixels),
        "mean_ratio": cell_pixels_mean
        / (background_pixels_mean if background_pixels_mean != 0 else 1),
        "mean_difference": cell_pixels_mean - background_pixels_mean,
        "mean_all_ratio": None,
        "mean_all_difference": None,
        "mean_max_ratio": None,
        "mean_max_difference": None,
        "lbp_mean": np.mean(lbp),
        "lbp_std": np.std(lbp),
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
        "area": prop.area,
        "perimeter": prop.perimeter,
        "eccentricity": prop.eccentricity,
        "major_axis_length": prop.major_axis_length,
        "minor_axis_length": prop.minor_axis_length,
        "solidity": prop.solidity,
        "extent": prop.extent,
        "hog_mean": np.mean(h),
        "hog_std": np.std(h),
    }

    return cell_mask, stats_dict


def create_stats_dict(roi_dict, layer):

    cell_masks = []
    roi_props = {}

    for roi_name, roi_info in roi_dict.items():
        cell_mask, stats_dict = extract_roi_stats(layer, roi_info)
        roi_props[roi_name] = stats_dict
        cell_masks.append(cell_mask)

    # Compute all background
    all_cell_mask = np.bitwise_or.reduce(cell_masks)
    all_background_mask = 1 - all_cell_mask
    all_background_pixels = layer[all_background_mask == 1]
    all_background_pixels_nonzero = all_background_pixels[all_background_pixels != 0]
    mean_all_background = np.mean(all_background_pixels_nonzero)

    cell_means_image = []
    for stats_dict in roi_props.values():
        mean_intensity = stats_dict["mean_intensity"]
        stats_dict["mean_all_ratio"] = mean_intensity / mean_all_background
        stats_dict["mean_all_difference"] = mean_intensity - mean_all_background
        cell_means_image.append(mean_intensity)

    # Compute max mean cell intensity
    max_mean_intensity = max(cell_means_image)
    for stats_dict in roi_props.values():
        mean_intensity = stats_dict["mean_intensity"]
        stats_dict["mean_max_ratio"] = max_mean_intensity / mean_intensity
        stats_dict["mean_max_difference"] = max_mean_intensity - mean_intensity

    return roi_props


def crop_cell_large(layer, x_coords, y_coords, padding=None):

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    if padding is None:
        x_padding = x_max - x_min
    else:
        x_padding = padding
    x_min, x_max = x_min - x_padding, x_max + x_padding

    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    if padding is None:
        y_padding = y_max - y_min
    else:
        y_padding = padding
    y_min, y_max = y_min - y_padding, y_max + y_padding

    layer_height, layer_width = layer.shape[:2]
    x_min = max(0, x_min)
    x_max = min(layer_width - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(layer_height - 1, y_max)

    layer_cropped = layer[y_min : y_max + 1, x_min : x_max + 1]
    x_coords_cropped = [x - x_min for x in x_coords]
    y_coords_cropped = [y - y_min for y in y_coords]
    return layer_cropped, x_coords_cropped, y_coords_cropped


def compress(image, compress_n=2):

    height, width = image.shape[:2]

    return cv2.resize(image, (width // compress_n, height // compress_n))


def crop(layer):

    # Eliminate rows and columns that are entirely zeros
    layer = layer[~np.all(layer == 0, axis=1)]
    layer = layer[:, ~np.all(layer == 0, axis=0)]

    return layer


def fix_polygon(polygon):
    if not polygon.is_valid:
        # Try to fix by applying a small buffer
        polygon = polygon.buffer(0)
        # Check if the polygon is still invalid
        if not polygon.is_valid:
            print("Polygon could not be fixed. Skipping this polygon.")
            return None
    return polygon
