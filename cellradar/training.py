"""
@author: Marc Canela
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from cellradar.utils import create_stats_dict, crop_cell_large


def user_input(
    roi_dict: Dict[str, Dict[str, np.ndarray]], layer: np.ndarray
) -> Dict[str, Dict[str, int]]:
    """
    This function visually displays each ROI overlaid on the image layer and
    prompts the user to classify the ROI as either a cell (1) or non-cell (0).
    The results are stored in a dictionary with the ROI names as keys and the
    labels as values.

    Parameters:
    ----------
    roi_dict : dict
        A dictionary containing the coordinates of the ROIs. Each entry should
        have at least the following keys:
        - "x": A list or array of x-coordinates of the ROI vertices.
        - "y": A list or array of y-coordinates of the ROI vertices.

    layer : numpy.ndarray
        A 2D NumPy array representing the image layer on which the ROIs are overlaid.
        The shape of the array should be (height, width).

    Returns:
    -------
    dict
        A dictionary where keys are the ROI names and values are dictionaries with
        a key "label" and an integer value representing the user's classification:
        1 for cell, 0 for non-cell.
    """
    labels = {}

    for roi_name, roi_info in roi_dict.items():
        x_coords, y_coords = roi_info["x"], roi_info["y"]

        # Set up the plot with four subplots
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        # Full image with ROI highlighted
        axes[0].imshow(layer, cmap="Reds")
        axes[0].plot(x_coords, y_coords, "b-", linewidth=1)
        axes[0].axis("off")  # Hide the axis

        # Full image without ROI highlighted
        axes[1].imshow(layer, cmap="Reds")
        axes[1].axis("off")  # Hide the axis

        # Cropped image with padding, ROI highlighted
        layer_cropped_small, x_coords_cropped, y_coords_cropped = crop_cell_large(
            layer, x_coords, y_coords, padding=120
        )
        axes[2].imshow(layer_cropped_small, cmap="Reds")
        axes[2].plot(x_coords_cropped, y_coords_cropped, "b-", linewidth=1)
        axes[2].axis("off")  # Hide the axis

        # Cropped image without ROI highlighted
        axes[3].imshow(layer_cropped_small, cmap="Reds")
        axes[3].axis("off")  # Hide the axis

        plt.tight_layout()
        plt.show()
        plt.pause(0.1)

        # Ask for user input
        user_input_value = input("Please enter 1 (cell) or 0 (non-cell): ")
        while user_input_value not in ["1", "0"]:
            user_input_value = input("Invalid input. Please enter 1 or 0: ")

        labels[roi_name] = {"label": int(user_input_value)}
        plt.close(fig)

    return labels


def label_rois(rois: Dict[str, dict], layers: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    This function processes the provided ROIs by calculating various statistical and texture features
    for each ROI in each image layer. It then prompts the user to label each ROI as a cell or non-cell.
    Finally, the features and labels are combined into a DataFrame for further analysis or model training.

    Parameters:
    ----------
    rois : dict
        A dictionary where keys are image tags and values are dictionaries of ROIs.
        Each ROI dictionary contains the coordinates of the ROI.

    layers : dict
        A dictionary where keys are image tags and values are 2D NumPy arrays representing
        the image layers from which the ROIs were extracted.

    Returns:
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to an ROI and contains its features and label.
    """

    # Step 1: Extract statistical features from each ROI
    roi_props_dict = {}
    for tag in tqdm(rois.keys(), desc="Extracting input features", unit="image"):
        roi_dict = rois[tag]
        layer = layers[tag]
        roi_props_dict[tag] = create_stats_dict(roi_dict, layer)

    # Step 2: Flatten the dictionary structure for input features
    input_features = {}
    for tag, all_rois in roi_props_dict.items():
        for roi_num, stats in all_rois.items():
            input_features[f"{tag}_{roi_num}"] = stats

    # Step 3: Collect user labels for each ROI
    labels_dict = {}
    for tag in rois.keys():
        roi_dict = rois[tag]
        layer = layers[tag]
        labels_dict[tag] = user_input(roi_dict, layer)

    # Step 4: Flatten the dictionary structure for labels
    input_labels = {}
    for tag, all_labels in labels_dict.items():
        for roi_num, labels in all_labels.items():
            input_labels[f"{tag}_{roi_num}"] = labels

    # Step 5: Combine the features and labels into a DataFrame
    data = {}
    for key in set(input_features.keys()).union(input_labels.keys()):
        if key in input_features and key in input_labels:
            data[key] = {**input_features[key], **input_labels[key]}

    df = pd.DataFrame.from_dict(data, orient="index")

    # Ensure specific columns have the correct data types
    df["min_intensity"] = df["min_intensity"].astype(int)
    df["max_intensity"] = df["max_intensity"].astype(int)
    df["hog_mean"] = df["hog_mean"].astype(float)
    df["hog_std"] = df["hog_std"].astype(float)

    return df


def random_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function splits a DataFrame into random train and test subsets.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to split, where the last column is the target variable.

    test_size : float, optional
        The proportion of the data to include in the test split (default is 0.2).

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the training features, training labels, testing features, and testing labels:
        - X_train: Features for the training set.
        - y_train: Labels for the training set.
        - X_test: Features for the testing set.
        - y_test: Labels for the testing set.
    """

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)

    # Extracting features and labels
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    return X_train, y_train, X_test, y_test


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[RandomizedSearchCV, Pipeline]:
    """
    This function trains an SVM model with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, a 2D array where each row is a sample and each column is a feature.

    y_train : np.ndarray
        The training labels, a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    Tuple[RandomizedSearchCV, Pipeline]
        - random_search: The RandomizedSearchCV object containing the results of the search.
        - best_model: The best estimator found by the random search, ready for prediction.
    """

    # Create a pipeline with scaling, PCA, and SVM
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("svm", SVC(kernel="rbf", random_state=42)),
        ]
    )

    # Define the distribution of hyperparameters for RandomizedSearchCV
    param_dist = {
        "pca__n_components": uniform(0.5, 0.5),  # Number of components for PCA
        "svm__C": uniform(1, 100),  # Regularization parameter C for SVM
        "svm__gamma": uniform(0.001, 0.1),  # Kernel coefficient for RBF kernel
    }

    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1, verbose=0
    )

    # Fit the model to the training data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return random_search, best_model


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[RandomizedSearchCV, RandomForestClassifier]:
    """
    This function trains a Random Forest Classifier with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, typically a 2D array where each row represents a sample and each column represents a feature.

    y_train : np.ndarray
        The training labels, typically a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    Tuple[RandomizedSearchCV, RandomForestClassifier]
        - random_search: The RandomizedSearchCV object containing the results of the search.
        - best_model: The best estimator found by the random search, which is a RandomForestClassifier.
    """

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ["auto", "sqrt"]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)] + [None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    param_dist = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1, verbose=0
    )

    # Fit RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return random_search, best_model


def train_logreg(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[RandomizedSearchCV, Pipeline]:
    """
    This function trains a Logistic Regression model with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, typically a 2D array where each row represents a sample and each column represents a feature.

    y_train : np.ndarray
        The training labels, typically a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    Tuple[RandomizedSearchCV, Pipeline]
        - random_search: The RandomizedSearchCV object containing the results of the search.
        - best_model: The best estimator found by the random search, which is a Pipeline containing PCA and LogisticRegression.
    """

    # Define the pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("log_reg", LogisticRegression(random_state=42)),
        ]
    )

    # Define the hyperparameter grid
    param_dist = {
        "pca__n_components": uniform(0.5, 0.5),
        "log_reg__C": uniform(1, 100),
    }

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1, verbose=0
    )

    # Fit RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return random_search, best_model


def evaluate(
    image_folder: Path,
    best_model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    plot: Optional[bool] = False,
) -> None:
    """
    This function evaluates the performance of a machine learning model using cross-validation and save the metrics.

    Parameters:
    ----------
    image_folder : Path
        The folder where the evaluation results will be saved. The results will be saved as a CSV file named 'evaluation.csv'.

    best_model : Pipeline
        The trained model pipeline to evaluate. This model should be an estimator object that supports `cross_val_predict`.

    X : np.ndarray
        The feature matrix used for evaluation. It should be a 2D array where each row represents a sample and each column represents a feature.

    y : np.ndarray
        The true labels corresponding to `X`. It should be a 1D array with labels for each sample.

    plot : Optional[bool], default=False
        If True, a Precision-Recall curve will be plotted.

    Returns:
    -------
    None
        This function does not return any value. It saves the evaluation metrics to a CSV file and optionally plots a Precision-Recall curve.
    """

    # Predict labels using cross-validation
    y_pred = cross_val_predict(best_model, X, y, cv=3)

    # Try to get decision scores or probability scores
    try:
        y_scores = cross_val_predict(best_model, X, y, cv=3, method="decision_function")
    except (AttributeError, NotImplementedError):
        y_scores = cross_val_predict(best_model, X, y, cv=3, method="predict_proba")
        y_scores = y_scores[:, 1]  # Use the probability for the positive class

    # Calculate metrics
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_scores)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(
        {
            "metric": ["precision", "recall", "f1", "roc_auc"],
            "score": [precision, recall, f1, roc],
        }
    )
    evaluate_path = image_folder.parent / "evaluation.csv"
    metrics_df.to_csv(evaluate_path, index=False)

    # Print confusion matrix
    print(f"Confusion matrix: \n{confusion_matrix(y, y_pred)}")

    # Optionally plot Precision-Recall curve
    if plot:
        precisions, recalls, _ = precision_recall_curve(y, y_scores)
        plt.plot(recalls, precisions, linewidth=2, label="Precision-Recall curve")
        plt.xlabel("Recall (TP / (TP + FN))")
        plt.ylabel("Precision (TP / (TP + FP))")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.show()
