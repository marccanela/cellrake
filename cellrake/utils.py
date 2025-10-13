# Created by: Marc Canela

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import uniform
from shapely.geometry import Polygon
from skimage.draw import polygon
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def export_data(data, project_folder, file_name):
    """
    Saves data to CSV and Excel files in the specified project folder.

    Parameters:
    ----------
    data : pd.DataFrame
        Data to save.

    project_folder : Path
        Folder to save the files.

    file_name : str
        Base name of the files (without extensions).
    """
    data.to_csv(project_folder / f"{file_name}.csv", index=False)
    data.to_excel(project_folder / f"{file_name}.xlsx", index=False)


def get_cell_mask(layer: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    This function generates a binary mask where the specified ROI regions,
    defined by polygonal coordinates, are filled with ones (1). All other
    areas in the mask are set to zeros (0). The dimensions of the mask match
    the dimensions of the input image layer.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer. The shape of this array
        (height, width) determines the dimensions of the resulting mask.

    coordinates : list of numpy.ndarray
        A NumPy array of shape (N, 2) that represents the vertices of a polygon
        that defines a ROI. This polygon specifies the area to be filled in the mask.

    Returns:
    -------
    numpy.ndarray
        A binary mask of the same shape as `layer`. Pixels within the defined
        polygonal regions are set to `1`, and all other pixels are set to `0`.
    """

    mask = np.zeros(layer.shape, dtype=np.uint8)

    # Extract x and y coordinates separately
    r = coordinates[0, :, 1]  # y-coordinates
    c = coordinates[0, :, 0]  # x-coordinate

    # Get the indices of the pixels that are inside the polygon
    rr, cc = polygon(r, c, mask.shape)

    # Fill the mask
    mask[rr, cc] = 1

    return mask


def crop_cell(
    layer: np.ndarray,
    x_coords: Union[List[int], np.ndarray],
    y_coords: Union[List[int], np.ndarray],
) -> np.ndarray:
    """
    This function extracts a rectangular subregion from the provided image `layer`.
    The rectangle is defined by the minimum and maximum x and y coordinates.
    The subregion is extracted by slicing the `layer` array.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which the region is to be cropped.
        The shape of the array should be (height, width).

    x_coords : list or numpy.ndarray
        A list or array of x-coordinates defining the horizontal extent of the rectangular region to crop.
        The function calculates the minimum and maximum x-coordinates to determine the horizontal boundaries.

    y_coords : list or numpy.ndarray
        A list or array of y-coordinates defining the vertical extent of the rectangular region to crop.
        The function calculates the minimum and maximum y-coordinates to determine the vertical boundaries.

    Returns:
    -------
    numpy.ndarray
        A 2D NumPy array representing the cropped region of the image layer. The dimensions of the
        cropped region are determined by the min and max x and y coordinates.
    """
    # Extract min and max coordinates and convert to integers
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Crop the rectangular region from the layer
    return layer[y_min : y_max + 1, x_min : x_max + 1]


def extract_roi_stats(
    layer: np.ndarray, roi_info: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    This function computes a binary mask for the ROI, crops the image and mask to
    the bounding box of the ROI, and calculates a range of statistical and texture
    features from the cropped and masked region.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer. The shape should be (height, width).

    roi_info : dict
        A dictionary containing the coordinates of the ROI with the following keys:
        - "x": A list or array of x-coordinates of the ROI vertices.
        - "y": A list or array of y-coordinates of the ROI vertices.

    Returns:
    -------
    cell_mask : numpy.ndarray
        A binary 2D NumPy array with the same size as `layer`, where pixels within the ROI are set to 1.

    stats_dict : dict
        A dictionary containing various statistical and texture features extracted from the ROI:
        - "mean_intensity": Mean pixel intensity within the ROI.
        - "median_intensity": Median pixel intensity within the ROI.
        - "sd_intensity": Standard deviation of pixel intensities within the ROI.
        - "min_intensity": Minimum pixel intensity within the ROI.
        - "max_intensity": Maximum pixel intensity within the ROI.
        - "mean_ratio": Ratio of mean intensity of the ROI to the background.
        - "mean_difference": Difference between mean intensity of the ROI and the background.
        - "lbp_mean": Mean Local Binary Pattern (LBP) value within the ROI.
        - "lbp_std": Standard deviation of LBP values within the ROI.
        - "contrast": Contrast from the Gray Level Co-occurrence Matrix (GLCM).
        - "correlation": Correlation from the GLCM.
        - "energy": Energy from the GLCM.
        - "homogeneity": Homogeneity from the GLCM.
        - "area": Area of the ROI.
        - "perimeter": Perimeter of the ROI.
        - "eccentricity": Eccentricity of the ROI.
        - "major_axis_length": Length of the major axis of the ROI.
        - "minor_axis_length": Length of the minor axis of the ROI.
        - "solidity": Solidity of the ROI.
        - "extent": Extent of the ROI.
        - "hog_mean": Mean Histogram of Oriented Gradients (HOG) descriptor value.
        - "hog_std": Standard deviation of the HOG descriptor values.
    """
    # Extract ROI coordinates from the dictionary
    x_coords, y_coords = roi_info["x"], roi_info["y"]
    coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)

    # Create a binary mask for the ROI
    cell_mask = get_cell_mask(layer, coordinates)

    # If the mask is empty, skip
    if np.mean(cell_mask) == 0:
        cell_mask = None
        stats_dict = None
        return cell_mask, stats_dict

    # Crop the layer and mask to the bounding box of the ROI
    layer_cropped = crop_cell(layer, x_coords, y_coords)
    cell_mask_cropped = crop_cell(cell_mask, x_coords, y_coords)

    # Create a background mask
    background_mask_cropped = 1 - cell_mask_cropped

    # Extract pixel values for the ROI and background
    cell_pixels = layer[cell_mask == 1]
    cell_pixels_mean = np.mean(cell_pixels)
    background_pixels = layer_cropped[background_mask_cropped == 1]
    background_pixels_mean = (
        np.mean(background_pixels) if background_pixels.size > 0 else cell_pixels_mean
    )

    # Mask the cropped layer and calculate texture features
    layer_cropped_masked = layer_cropped * cell_mask_cropped
    layer_masked = layer * cell_mask

    # Calculate Local Binary Pattern (LBP)
    lbp = local_binary_pattern(layer_cropped_masked, P=8, R=1, method="uniform")

    # Calculate Gray Level Co-occurrence Matrix (GLCM) features
    glcm = graycomatrix(
        layer_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )

    # Calculate region properties
    prop = regionprops(label(cell_mask_cropped))[0]

    height, width = layer_cropped_masked.shape[:2]
    pixels_per_cell_val = 4
    cells_per_block_val = 2

    # Set the required minimum size based on the HOG parameters
    min_height = pixels_per_cell_val * cells_per_block_val
    min_width = pixels_per_cell_val * cells_per_block_val

    # Calculate the amount of padding needed
    pad_height = max(0, min_height - height)
    pad_width = max(0, min_width - width)

    # Apply padding to the image
    if pad_height > 0 or pad_width > 0:
        padded_image = np.pad(
            layer_cropped_masked,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )
    else:
        padded_image = layer_cropped_masked

    # Compute Histogram of Oriented Gradients (HOG) features
    h_values, _ = hog(
        padded_image,
        orientations=9,
        pixels_per_cell=(pixels_per_cell_val, pixels_per_cell_val),
        cells_per_block=(cells_per_block_val, cells_per_block_val),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )

    # Flatten the HOG features if needed (though skimage.hog already returns a flat array if feature_vector=True)
    hog_descriptor_values = h_values.flatten()

    # Create a dictionary to store the extracted features
    stats_dict = {
        "mean_intensity": cell_pixels_mean,
        "median_intensity": np.median(cell_pixels),
        "sd_intensity": np.std(cell_pixels),
        "min_intensity": np.min(cell_pixels),
        "max_intensity": np.max(cell_pixels),
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
        "hog_mean": np.mean(hog_descriptor_values),
        "hog_std": np.std(hog_descriptor_values),
    }

    return cell_mask, stats_dict


def create_stats_dict(
    roi_dict: Dict[str, Dict[str, Any]], layer: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    This function calculates features for each ROI, combines the results, and computes additional statistics
    based on the entire set of ROIs, including background statistics.

    Parameters:
    ----------
    roi_dict : dict
        A dictionary where each key is a name for a ROI and each value is another dictionary containing
        the coordinates of the ROI with keys "x" and "y".

    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which the ROIs are extracted. The shape should be (height, width).

    Returns:
    -------
    dict
        A dictionary with ROI names as keys and dictionaries of statistical features as values. Each features dictionary
        includes computed statistics for each ROI, including additional ratios and differences.
    """
    cell_masks = []
    roi_props = {}

    # Extract statistics for each ROI
    for roi_name, roi_info in roi_dict.items():
        cell_mask, stats_dict = extract_roi_stats(layer, roi_info)

        if cell_mask is not None and stats_dict is not None:
            roi_props[roi_name] = stats_dict
            cell_masks.append(cell_mask)

    # Compute combined background statistics
    combined_cell_mask = np.bitwise_or.reduce(cell_masks)
    combined_background_mask = 1 - combined_cell_mask
    combined_background_pixels = layer[combined_background_mask == 1]
    nonzero_background_pixels = combined_background_pixels[
        combined_background_pixels != 0
    ]

    # Handle case when all background pixels are zero to avoid division by zero
    mean_background = (
        np.mean(nonzero_background_pixels) if nonzero_background_pixels.size > 0 else 0
    )

    # Update statistics with additional ratios and differences
    for stats_dict in roi_props.values():
        mean_intensity = stats_dict["mean_intensity"]
        stats_dict["mean_all_ratio"] = mean_intensity / (
            mean_background if mean_background != 0 else 1
        )
        stats_dict["mean_all_difference"] = mean_intensity - mean_background

    # Compute maximum mean intensity across all ROIs
    max_mean_intensity = max(
        stats_dict["mean_intensity"] for stats_dict in roi_props.values()
    )
    for stats_dict in roi_props.values():
        mean_intensity = stats_dict["mean_intensity"]
        stats_dict["mean_max_ratio"] = max_mean_intensity / (
            mean_intensity if mean_intensity != 0 else 1
        )
        stats_dict["mean_max_difference"] = max_mean_intensity - mean_intensity

    return roi_props


def crop_cell_large(
    layer: np.ndarray,
    x_coords: List[float],
    y_coords: List[float],
    padding: Optional[int] = None,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    This function extracts a subregion from the image `layer` based on the provided x and y coordinates
    and adjusts these coordinates relative to the cropped image. Padding can be applied to extend the
    crop area beyond the bounding box of the ROI.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D or 3D NumPy array representing the image layer. The shape should be (height, width) or (height, width, channels).

    x_coords : list of float
        A list of x-coordinates defining the boundary of the rectangular region to crop.

    y_coords : list of float
        A list of y-coordinates defining the boundary of the rectangular region to crop.

    padding : int or None, optional
        An integer value specifying the amount of padding to add to the crop area. If None, padding
        will be set to the width or height of the bounding box of the ROI. Defaults to None.

    Returns:
    -------
    Tuple[np.ndarray, List[float], List[float]]
        - The cropped image layer as a NumPy array.
        - The x-coordinates adjusted to the cropped image.
        - The y-coordinates adjusted to the cropped image.
    """
    # Determine cropping boundaries
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Apply padding if specified
    if padding is not None:
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
    else:
        x_padding = x_max - x_min
        y_padding = y_max - y_min
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

    # Ensure cropping boundaries are within image dimensions
    layer_height, layer_width = layer.shape[:2]
    x_min = max(0, x_min)
    x_max = min(layer_width - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(layer_height - 1, y_max)

    # Crop the image
    layer_cropped = layer[y_min : y_max + 1, x_min : x_max + 1]

    # Adjust coordinates relative to the cropped image
    x_coords_cropped = [x - x_min for x in x_coords]
    y_coords_cropped = [y - y_min for y in y_coords]

    return layer_cropped, x_coords_cropped, y_coords_cropped


def fix_polygon(polygon: Polygon) -> Polygon:
    """
    This function checks if the provided polygon is valid. If it is not valid, it attempts to fix the polygon by
    applying a buffer with a width of zero. If the polygon is still invalid after this operation, a message is printed
    and `None` is returned.

    Parameters:
    ----------
    polygon : shapely.geometry.Polygon
        The input polygon that needs validation and potential correction.

    Returns:
    -------
    shapely.geometry.Polygon or None
        A valid polygon if the correction was successful, or `None` if the polygon could not be fixed.
    """
    if not polygon.is_valid:
        # Attempt to fix the polygon by applying a zero-width buffer
        polygon = polygon.buffer(0)

        # Check if the polygon is still invalid after buffering
        if not polygon.is_valid:
            print("Polygon could not be fixed. Skipping this polygon.")
            return None

    return polygon


def crop(layer: np.ndarray) -> np.ndarray:
    """
    This function trims the input image layer to remove any rows or columns that contain only zero values.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer. The shape of the array should be (height, width).

    Returns:
    -------
    numpy.ndarray
        The cropped image layer with zero-only rows and columns removed.
    """
    # Remove rows that are entirely zeros
    layer = layer[~np.all(layer == 0, axis=1)]

    # Remove columns that are entirely zeros
    layer = layer[:, ~np.all(layer == 0, axis=0)]

    return layer


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    This function trains an SVM model with hyperparameter tuning using RandomizedSearchCV

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, a 2D array where each row is a sample and each column is a feature.

    y_train : np.ndarray
        The training labels, a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    best_model: The best estimator found by the random search, ready for prediction.
    """

    # Create a pipeline with scaling, PCA, and SVM
    pipeline_steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        (
            "svm",
            SVC(
                kernel="rbf", probability=True, class_weight="balanced", random_state=42
            ),
        ),
    ]
    pipeline = Pipeline(pipeline_steps)

    # Define the distribution of hyperparameters for RandomizedSearchCV
    param_dist = {
        "svm__C": uniform(1, 100),  # Regularization parameter
    }

    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=50,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )

    # Fit the model to the training data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return best_model


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str = "rf"
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    This function trains a Random Forest Classifier or Extra Trees Classifier with hyperparameter tuning using RandomizedSearchCV

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, typically a 2D array where each row represents a sample and each column represents a feature.

    y_train : np.ndarray
        The training labels, typically a 1D array where each element is the label for the corresponding sample in X_train.

    model_type : str, optional
        The type of model to train. Options are 'rf' (Random Forest) or 'et' (Extra Trees). Default is 'rf'.

    Returns:
    -------
    best_model: The best estimator found by the random search, which is a RandomForestClassifier or ExtraTreesClassifier.
    """

    if model_type == "et":
        rf = ExtraTreesClassifier(class_weight="balanced", random_state=42)
    else:
        rf = RandomForestClassifier(class_weight="balanced", random_state=42)

    # Define the hyperparameter grid
    param_dist = {
        "n_estimators": [int(x) for x in np.linspace(start=50, stop=500, num=10)],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [int(x) for x in np.linspace(10, 100, num=10)] + [None],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 8, 10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
    }

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=50,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0,  # Verbosity level for detailed output
        error_score="raise",
    )

    # Fit RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return best_model


def train_logreg(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    This function trains a Logistic Regression model with hyperparameter tuning using RandomizedSearchCV

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, typically a 2D array where each row represents a sample and each column represents a feature.

    y_train : np.ndarray
        The training labels, typically a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    best_model: The best estimator found by the random search, which is a Pipeline containing PCA and LogisticRegression.
    """

    # Define the pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("log_reg", LogisticRegression(class_weight="balanced", random_state=42)),
        ]
    )

    # Define the hyperparameter grid
    param_dist = {
        "log_reg__C": uniform(1, 100),
    }

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=50,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0,  # Verbosity level for detailed output
        error_score="raise",
    )

    # Fit RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return best_model


def train_model(X_labeled, y_labeled, model_type):
    """
    Trains a machine learning model based on the specified model type.

    Parameters:
    ----------
    X_labeled (pd.DataFrame): The labeled feature data.
    y_labeled (pd.Series): The labeled target data.
    model_type (str): The type of model to train. Options are "svm" for Support Vector Machine,
    "rf" for Random Forest, "et" for Extra Trees, and "logreg" for Logistic Regression.

    Returns:
    ----------
    best_model: The trained machine learning model.
    metrics: A dictionary containing evaluation metrics for the model.
    """
    if model_type == "svm":
        best_model = train_svm(X_labeled, y_labeled)
    elif model_type in ["rf", "et"]:
        best_model = train_rf(X_labeled, y_labeled, model_type)
    elif model_type == "logreg":
        best_model = train_logreg(X_labeled, y_labeled)

    return best_model


def evaluate_model(best_model, X, y):
    """
    Evaluate the performance of a given model using cross-validated predictions.

    Parameters:
    ----------
    best_model (sklearn.base.BaseEstimator): The model to be evaluated.
    X (pd.DataFrame or np.ndarray): The feature matrix.
    y (pd.Series or np.ndarray): The target vector.

    Returns:
    ----------
    dict: A dictionary containing the following metrics:
        - 'roc_auc' (float): The ROC AUC score.
        - 'ap' (float): The average precision score.
        - 'precision' (float): The precision score.
        - 'recall' (float): The recall score.
        - 'f1_score' (float): The F1 score.
    """
    # Get cross-validated predictions and probabilities
    y_pred_proba = cross_val_predict(best_model, X, y, cv=3, method="predict_proba")
    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
    y_proba = y_pred_proba[:, 1]

    # Calculate metrics
    roc_auc = roc_auc_score(y, y_proba)
    ap = average_precision_score(y, y_proba)
    precision_score_value = precision_score(y, y_pred)
    recall_score_value = recall_score(y, y_pred)
    f1_score_value = f1_score(y, y_pred)

    metrics = {
        "roc_auc": roc_auc,
        "ap": ap,
        "precision": precision_score_value,
        "recall": recall_score_value,
        "f1_score": f1_score_value,
    }

    return metrics
