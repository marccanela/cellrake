"""
@author: Marc Canela
"""

import pickle as pkl
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from cellrake.predicting import iterate_predicting
from cellrake.segmentation import export_rois, iterate_segmentation
from cellrake.training import (
    label_rois,
    random_train_test_split,
    train_logreg,
    train_rf,
    train_svm,
)
from cellrake.utils import build_project


def analyze(
    image_folder: Path, cmap: str = "Reds", best_model: Optional[Pipeline] = None
) -> None:
    """
    This function processes TIFF images located in the `image_folder` by:
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
    best_model : Optional[Pipeline], optional
        A scikit-learn pipeline object used for predictions. This model should be previously obtained
        through functions like `train` or `expand_retrain`. If not provided, a standard filter will be used.

    Returns:
    -------
    None
    """

    # Ensure the provided color map is valid
    if cmap not in plt.colormaps():
        raise ValueError(
            f"Invalid colormap '{cmap}'. Available options are: {', '.join(plt.colormaps())}"
        )

    # Create a project folder for organizing results
    project_folder = build_project(image_folder)

    # Segment images to obtain two dictionaries: 'rois' and 'layers'
    rois, layers = iterate_segmentation(image_folder)

    # Export segmented ROIs to the project folder
    export_rois(project_folder, rois)

    # Apply the prediction model to the layers and ROIs
    if best_model:
        print("Applying a trained model to make predictions...")
    else:
        print("Lacking a trained model. Applying basic intensity threshold...")

    iterate_predicting(layers, rois, cmap, project_folder, best_model)


def train(image_folder: Path, model_type: str = "svm"):
    """
    This function trains a machine learning model using segmented images from the specified folder.

    Parameters:
    ----------
    image_folder : Path
        The folder containing TIFF images to be segmented and used for training.
    model_type : str, optional
        The type of model to train. Options are 'svm', 'rf' (Random Forest), or 'logreg' (Logistic Regression).
        Default is 'svm'.

    Returns:
    -------
    best_model : Pipeline
        The trained model with the best parameters.
    X_train : numpy.ndarray
        Features for training.
    y_train : numpy.ndarray
        Labels for training.
    X_test : numpy.ndarray
        Features for testing.
    y_test : numpy.ndarray
        Labels for testing.
    """

    # Segment images to obtain ROIs and layers
    rois, layers = iterate_segmentation(image_folder)

    # Extract features and labels from ROIs
    df = label_rois(rois, layers)
    features_path = image_folder.parent / "features.csv"
    df.to_csv(features_path, index=False)

    # Split the data into training and test sets
    X_train, y_train, X_test, y_test = random_train_test_split(df)

    # Train the specified model
    if model_type == "svm":
        random_search, best_model = train_svm(X_train, y_train)
    elif model_type == "rf":
        random_search, best_model = train_rf(X_train, y_train)
    elif model_type == "logreg":
        random_search, best_model = train_logreg(X_train, y_train)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Choose from 'svm', 'rf', or 'logreg'."
        )

    # Save the trained model
    model_path = image_folder.parent / f"best_model_{model_type}.pkl"
    with open(model_path, "wb") as file:
        pkl.dump(best_model, file)

    # Print the best parameters and cross-validation score
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", random_search.best_score_)

    return best_model, X_train, y_train, X_test, y_test


def expand_retrain(image_folder: Path, df: pd.DataFrame, model_type: str = "svm"):
    """
    This function expands the training dataset with additional features, retrains a machine learning model,
    and saves the updated model.

    Parameters:
    ----------
    image_folder : Path
        The folder containing TIFF images to segment and extract new features.
    df : pd.DataFrame
        The existing dataframe with features and labels to which new features will be added.
    model_type : str, optional
        The type of model to train. Options are 'svm', 'rf' (Random Forest), or 'logreg' (Logistic Regression).
        Default is 'svm'.

    Returns:
    -------
    best_model : object
        The retrained model with the best parameters.
    X_train : numpy.ndarray
        Features for training.
    y_train : numpy.ndarray
        Labels for training.
    X_test : numpy.ndarray
        Features for testing.
    y_test : numpy.ndarray
        Labels for testing.
    """

    # Segment images and extract new features
    rois, layers = iterate_segmentation(image_folder)
    df_2 = label_rois(rois, layers)

    # Combine existing and new features
    combined_df = pd.concat([df, df_2], ignore_index=True)
    features_path = image_folder.parent / "expanded_features.csv"
    combined_df.to_csv(features_path, index=False)

    # Split the combined dataframe into training and testing sets
    X_train, y_train, X_test, y_test = random_train_test_split(combined_df)

    # Train the specified model
    if model_type == "svm":
        random_search, best_model = train_svm(X_train, y_train)
    elif model_type == "rf":
        random_search, best_model = train_rf(X_train, y_train)
    elif model_type == "logreg":
        random_search, best_model = train_logreg(X_train, y_train)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Choose from 'svm', 'rf', or 'logreg'."
        )

    # Save the retrained model
    model_path = image_folder.parent / f"expanded_best_model_{model_type}.pkl"
    with open(model_path, "wb") as file:
        pkl.dump(best_model, file)

    # Print the best parameters and cross-validation score
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", random_search.best_score_)

    return best_model, X_train, y_train, X_test, y_test
