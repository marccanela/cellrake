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


def export_data(data: pd.DataFrame, project_folder: Path, file_name: str) -> None:
    """
    Save data to CSV and Excel files in the specified project folder.

    Parameters
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
    Generate a binary mask for ROI regions defined by polygonal coordinates.

    Parameters
    ----------
    layer : np.ndarray
        2D image layer determining mask dimensions.
    coordinates : np.ndarray
        Array of shape (N, 2) representing polygon vertices defining ROI.

    Returns
    -------
    np.ndarray
        Binary mask with ROI pixels set to 1, others to 0.
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
    Extract a rectangular subregion from an image layer.

    Parameters
    ----------
    layer : np.ndarray
        2D image layer to crop from.
    x_coords : Union[List[int], np.ndarray]
        X-coordinates defining horizontal crop boundaries.
    y_coords : Union[List[int], np.ndarray]
        Y-coordinates defining vertical crop boundaries.

    Returns
    -------
    np.ndarray
        Cropped rectangular region from the image layer.
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
    Extract statistical and texture features from a region of interest (ROI).

    Parameters
    ----------
    layer : np.ndarray
        2D image layer for feature extraction.
    roi_info : Dict[str, Any]
        Dictionary with ROI coordinates containing "x" and "y" keys.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Binary mask and dictionary of extracted features including intensity,
        texture, and morphological statistics.
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
    Calculate features for multiple ROIs with background statistics.

    Parameters
    ----------
    roi_dict : Dict[str, Dict[str, Any]]
        Dictionary mapping ROI names to coordinate dictionaries.
    layer : np.ndarray
        2D image layer for feature extraction.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with ROI names as keys and feature dictionaries as values.
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
    Extract a padded subregion from an image layer with coordinate adjustment.

    Parameters
    ----------
    layer : np.ndarray
        2D or 3D image layer to crop from.
    x_coords : List[float]
        X-coordinates defining the crop boundary.
    y_coords : List[float]
        Y-coordinates defining the crop boundary.
    padding : Optional[int]
        Padding to add around crop area. If None, uses ROI dimensions.

    Returns
    -------
    Tuple[np.ndarray, List[float], List[float]]
        Cropped image layer and adjusted coordinates.
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
    Validate and fix invalid polygons using buffer operations.

    Parameters
    ----------
    polygon : Polygon
        Input polygon to validate and potentially fix.

    Returns
    -------
    Polygon
        Valid polygon or None if correction failed.
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
    Remove rows and columns containing only zero values from image layer.

    Parameters
    ----------
    layer : np.ndarray
        2D image layer to trim.

    Returns
    -------
    np.ndarray
        Cropped image layer with zero-only rows and columns removed.
    """
    # Remove rows that are entirely zeros
    layer = layer[~np.all(layer == 0, axis=1)]

    # Remove columns that are entirely zeros
    layer = layer[:, ~np.all(layer == 0, axis=0)]

    return layer


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Train an SVM model with hyperparameter tuning using RandomizedSearchCV.

    Parameters
    ----------
    X_train : np.ndarray
        Training features array.
    y_train : np.ndarray
        Training labels array.

    Returns
    -------
    Pipeline
        Best SVM estimator found by random search.
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
) -> Union[RandomForestClassifier, ExtraTreesClassifier]:
    """
    Train a Random Forest or Extra Trees classifier with hyperparameter tuning.

    Parameters
    ----------
    X_train : np.ndarray
        Training features array.
    y_train : np.ndarray
        Training labels array.
    model_type : str
        Model type: 'rf' for Random Forest or 'et' for Extra Trees.

    Returns
    -------
    Union[RandomForestClassifier, ExtraTreesClassifier]
        Best estimator found by random search.
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


def train_logreg(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Train a Logistic Regression model with hyperparameter tuning.

    Parameters
    ----------
    X_train : np.ndarray
        Training features array.
    y_train : np.ndarray
        Training labels array.

    Returns
    -------
    Pipeline
        Best estimator pipeline with PCA and LogisticRegression.
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


def train_model(
    X_labeled: np.ndarray, y_labeled: np.ndarray, model_type: str
) -> Union[Pipeline, RandomForestClassifier, ExtraTreesClassifier]:
    """
    Train a machine learning model based on the specified model type.

    Parameters
    ----------
    X_labeled : np.ndarray
        Labeled feature data.
    y_labeled : np.ndarray
        Labeled target data.
    model_type : str
        Model type: "svm", "rf", "et", or "logreg".

    Returns
    -------
    Union[Pipeline, RandomForestClassifier, ExtraTreesClassifier]
        Trained machine learning model.
    """
    if model_type == "svm":
        best_model = train_svm(X_labeled, y_labeled)
    elif model_type in ["rf", "et"]:
        best_model = train_rf(X_labeled, y_labeled, model_type)
    elif model_type == "logreg":
        best_model = train_logreg(X_labeled, y_labeled)

    return best_model


def evaluate_model(
    best_model: Union[Pipeline, RandomForestClassifier, ExtraTreesClassifier],
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate model performance using cross-validated predictions.

    Parameters
    ----------
    best_model : Union[Pipeline, RandomForestClassifier, ExtraTreesClassifier]
        Model to be evaluated.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics: roc_auc, ap, precision, recall, f1_score.
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
