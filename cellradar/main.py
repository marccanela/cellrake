"""
@author: Marc Canela
"""

import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from predicting import iterate_predicting
from segmentation import export_rois, iterate_segmentation
from sklearn.pipeline import Pipeline
from training import (
    label_rois,
    random_train_test_split,
    train_logreg,
    train_rf,
    train_svm,
)
from utils import build_project


def analyze(image_folder: Path, cmap: str = "Reds", best_model: Pipeline = None):
    """
    This function processes images located in the `image_folder` by:
    1. Building a project directory.
    2. Segmenting the images to identify regions of interest (ROIs).
    3. Exporting the segmented ROIs to the project folder.
    4. Applying a prediction model (optional) to the segmented ROIs.

    Parameters:
    ----------
    image_folder : Path
        A `Path` object representing the folder containing TIFF image files to analyze.
    cmap : str, optional
        The color map to use for visualization when plotting results using matplotlib. Default is "Reds".
        It should be one of the available color maps in matplotlib, such as 'Reds', 'Greens', etc.
    best_model : Pipeline, optional
        A scikit-learn pipeline object used for predictions. This model should be previously obtained
        through functions like `train` or `expand_retrain`. If not provided, a standard filter will be used.

    Returns:
    -------
    None
    """

    # Create a project folder for organizing results
    project_folder = build_project(image_folder)

    # Segment images to obtain two dictionaries: 'rois' and 'layers'
    rois, layers = iterate_segmentation(image_folder)

    # Export segmented ROIs to the project folder
    export_rois(project_folder, rois)

    # Apply the prediction model to the layers and ROIs
    iterate_predicting(layers, rois, cmap, project_folder, best_model)


def train(image_folder, model_type="svm"):

    # Segment images
    rois, layers = iterate_segmentation(image_folder)

    # Extract features and labels
    df = label_rois(rois, layers)
    features_path = image_folder.parent / "features.csv"
    df.to_csv(features_path, index=False)

    # Train model
    X_train, y_train, X_test, y_test = random_train_test_split(df)
    if model_type == "svm":
        random_search, best_model = train_svm(X_train, y_train)
    elif model_type == "rf":
        random_search, best_model = train_rf(X_train, y_train)
    elif model_type == "logreg":
        random_search, best_model = train_logreg(X_train, y_train)

    model_path = image_folder.parent / f"best_model_{model_type}.pkl"
    with open(model_path, "wb") as file:
        pkl.dump(best_model, file)

    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", random_search.best_score_)

    return best_model, X_train, y_train, X_test, y_test


def expand_retrain(image_folder, df, model_type="svm"):

    # Segment images
    rois, layers = iterate_segmentation(image_folder)

    # Extract features and labels
    df_2 = label_rois(rois, layers)

    # Combine dataframes
    combined_df = pd.concat([df, df_2], ignore_index=False)
    features_path = image_folder.parent / "expanded_features.csv"
    combined_df.to_csv(features_path, index=True)

    # Train model
    X_train, y_train, X_test, y_test = random_train_test_split(combined_df)
    if model_type == "svm":
        random_search, best_model = train_svm(X_train, y_train)
    elif model_type == "rf":
        random_search, best_model = train_rf(X_train, y_train)
    elif model_type == "logreg":
        random_search, best_model = train_logreg(X_train, y_train)

    model_path = image_folder.parent / f"expanded_best_model_{model_type}.pkl"
    with open(model_path, "wb") as file:
        pkl.dump(best_model, file)

    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", random_search.best_score_)

    return best_model, X_train, y_train, X_test, y_test
