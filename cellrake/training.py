"""
@author: Marc Canela
"""

from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from cellrake.utils import create_stats_dict, crop_cell_large


def user_input(roi_values: np.ndarray, layer: np.ndarray) -> Dict[str, Dict[str, int]]:
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
    x_coords, y_coords = roi_values["x"], roi_values["y"]

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

    plt.close(fig)

    return user_input_value


def create_subset_df(
    rois: Dict[str, dict], layers: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    This function processes the provided ROIs by calculating various statistical and texture features
    for each ROI in each image layer. It clusters the features into two groups (approx. positive and
    negative ROIs) and returns a sample dataframe of features with a balanced number of both clusters.

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
        A DataFrame where each row corresponds to an ROI and each column contains its features.
    """

    # Extract statistical features from each ROI
    roi_props_dict = {}
    for tag in tqdm(rois.keys(), desc="Extracting input features", unit="image"):
        roi_dict = rois[tag]
        layer = layers[tag]
        roi_props_dict[tag] = create_stats_dict(roi_dict, layer)

    # Flatten the dictionary structure for input features
    input_features = {}
    for tag, all_rois in roi_props_dict.items():
        for roi_num, stats in all_rois.items():
            input_features[f"{tag}_{roi_num}"] = stats
    features_df = pd.DataFrame.from_dict(input_features, orient="index")

    # Cluster the features to aproximate the positive/negative classes
    kmeans_2_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("kmeans", KMeans(n_clusters=2, random_state=42)),
        ]
    )
    best_clusters = kmeans_2_pipeline.fit_predict(features_df)
    features_df["cluster"] = best_clusters

    # Equilibrate both classes
    cluster_counts = features_df["cluster"].value_counts()
    size = np.min(cluster_counts)

    sampled_dfs = []
    for cluster in features_df["cluster"].unique():
        cluster_df = features_df[features_df["cluster"] == cluster]

        if len(cluster_df) >= size:
            sampled_df = cluster_df.sample(n=size, random_state=42)
        else:
            sampled_df = cluster_df

        sampled_dfs.append(sampled_df)

    subset_df = pd.concat(sampled_dfs)

    # Ensure specific columns have the correct data types
    subset_df["min_intensity"] = subset_df["min_intensity"].astype(int)
    subset_df["max_intensity"] = subset_df["max_intensity"].astype(int)
    subset_df["hog_mean"] = subset_df["hog_mean"].astype(float)
    subset_df["hog_std"] = subset_df["hog_std"].astype(float)

    return subset_df


def active_learning(
    subset_df: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    model_type: str = "svm",
) -> Union[Pipeline, RandomForestClassifier]:
    """
    The function begins by splitting the dataset into training and testing sets, with a small
    portion of the training set manually labeled. It then enters a loop where the model is trained,
    evaluated, and used to predict the uncertainty of the unlabeled instances. The most uncertain
    instances are selected for manual labeling, added to the labeled dataset, and the process repeats
    until the improvement in model performance becomes negligible.

    Parameters:
    ----------
    subset_df : pd.DataFrame
        A DataFrame where each row corresponds to an ROI and each column contains its features.

    rois : dict
        A dictionary where keys are image tags and values are dictionaries of ROIs.
        Each ROI dictionary contains the coordinates of the ROI.

    layers : dict
        A dictionary where keys are image tags and values are 2D NumPy arrays representing
        the image layers from which the ROIs were extracted.

    model_type : str, optional
        The type of model to train. Options are 'svm' (Support Vector Machine), 'rf' (Random Forest),
        or 'logreg' (Logistic Regression). Default is 'svm'.

    Returns:
    -------
    sklearn Pipeline or RandomForestClassifier
        The best estimator found by active learning.
    """

    # Split the dataset into train and test sets
    test_size = min(int(0.2 * len(subset_df)), 100)
    train_X, test_X = train_test_split(
        subset_df, test_size=test_size, stratify=subset_df["cluster"], random_state=42
    )

    # Label 10% of the data (max 100 instances) of train_X set
    initial_sample_size = min(int(0.1 * len(train_X)), 50)
    train_X_labeled, train_X_unlabeled = train_test_split(
        train_X,
        train_size=initial_sample_size,
        stratify=train_X["cluster"],
        random_state=42,
    )

    print("Label the train set:")
    train_y_labeled = manual_labeling(train_X_labeled, rois, layers)

    # Check if all labels are zeros
    while (
        train_y_labeled["label_column"].astype(int).nunique() == 1
        and train_y_labeled["label_column"].astype(int).unique()[0] == 0
    ):
        print("All labels in the the train set are zeros. Re-sampling and re-labeling.")
        train_X_labeled, train_X_unlabeled = train_test_split(
            train_X, train_size=initial_sample_size, stratify=train_X["cluster"]
        )
        train_y_labeled = manual_labeling(train_X_labeled, rois, layers)

    # Label the test_X set
    print("Label the test set:")
    test_y = manual_labeling(test_X, rois, layers)

    # Create datasets
    X_labeled = train_X_labeled.drop(columns=["cluster"])
    y_labeled = train_y_labeled["label_column"].astype(int)
    X_unlabeled = train_X_unlabeled.drop(columns=["cluster"])

    # Prepare test dataset
    X_test = test_X.drop(columns=["cluster"])
    y_test = test_y["label_column"].astype(int)

    # Initialize the active learning
    previous_loss = None
    min_delta = 0.001
    iteration = 1
    performance_scores = []
    models = []
    while True:

        # 1) Train
        if model_type == "svm":
            model = train_svm(X_labeled.values, y_labeled.values)
        elif model_type == "rf":
            model = train_rf(X_labeled.values, y_labeled.values)
        elif model_type == "logreg":
            model = train_logreg(X_labeled.values, y_labeled.values)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. Choose from 'svm', 'rf', or 'logreg'."
            )

        models.append(model)

        # 2) Evaluate (cross-entropy loss)
        f1 = f1_score(y_test, model.predict(X_test.values))
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
        validation_loss = log_loss(y_test, model.predict_proba(X_test.values))

        performance_scores.append(
            {
                "iteration": iteration,
                "f1": f1,
                "roc_auc": roc_auc,
                "loss": validation_loss,
            }
        )

        # Check if the improvement in performance is minimal
        if previous_loss is not None and (previous_loss - validation_loss) < min_delta:
            print("Loss improvement is minimal, stopping the iteration.")

            # If loss has increased, revert to the previous model
            if validation_loss > previous_loss:
                model = models[-2]
                performance_scores = performance_scores[:-1]
            break

        # Update previous_loss for the next iteration
        previous_loss = validation_loss
        iteration += 1  # Increment iteration count

        # 3) Predict and Extract
        # Calculate uncertainty
        # Closer to 1 indicates that the model is highly uncertain about that instance
        # Uncertainty is calculated based on how close the probabilities are to a 50/50 split
        uncertainties = 1 - np.max(model.predict_proba(X_unlabeled.values), axis=1)

        # Sort by uncertainty (highest uncertainty first)
        uncertain_indices = np.argsort(uncertainties)[-initial_sample_size // 2 :]

        # Recollect Uncertain Instances for Manual Labeling
        X_uncertain = X_unlabeled.iloc[uncertain_indices]

        # 4) Label
        y_uncertain = manual_labeling(X_uncertain, rois, layers)
        y_uncertain = y_uncertain["label_column"].astype(int)

        # Add newly labeled data to the labeled dataset
        X_labeled = pd.concat([X_labeled, X_uncertain])
        y_labeled = pd.concat([y_labeled, y_uncertain])

        # Remove the newly labeled data from the unlabeled dataset
        X_unlabeled = X_unlabeled.drop(X_uncertain.index)

    # Plotting the evolution of ROC-AUC and F1-Score over iterations
    # Export performance scores
    performance_df = pd.DataFrame(performance_scores)

    # Plot the F1 Score, ROC-AUC, and Loss for each iteration
    plt.figure(figsize=(10, 6))

    # Plotting each metric
    plt.plot(
        performance_df["iteration"], performance_df["f1"], label="F1 Score", marker="o"
    )
    plt.plot(
        performance_df["iteration"],
        performance_df["roc_auc"],
        label="ROC-AUC",
        marker="o",
    )
    plt.plot(
        performance_df["iteration"],
        performance_df["loss"],
        label="Cross-entropy Loss",
        marker="o",
    )

    # Adding labels and title
    plt.xlabel("Iteration")
    plt.ylabel("Score / Loss")
    plt.title("Performance Metrics Over Training Iterations")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    return model, performance_df


def manual_labeling(
    features_df: pd.DataFrame, rois: Dict[str, dict], layers: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    This function asks the user to label the images corresponding to the features_df.

    Parameters:
    ----------
    features_df: pd.DataFrame
        The training features where each row is a sample and each column is a feature.

    rois : dict
        A dictionary where keys are image tags and values are dictionaries of ROIs.
        Each ROI dictionary contains the coordinates of the ROI.

    layers : dict
        A dictionary where keys are image tags and values are 2D NumPy arrays representing
        the image layers from which the ROIs were extracted.

    Returns:
    -------
    pd.DataFrame
        A dataframe with the manual labels under the column "label_column"
    """
    if features_df.empty:
        raise ValueError("The features DataFrame is empty. Please provide a valid one.")

    index_list = features_df.index.tolist()

    labels_dict = {}
    n = 1
    for index in index_list:
        print(f"Image {n} out of {len(index_list)}.")
        tag = index.split("_roi")[0]
        roi = f"roi{index.split('_roi')[1]}"
        layer = layers[tag]
        roi_values = rois[tag][roi]
        labels_dict[index] = user_input(roi_values, layer)
        n += 1

    labels_df = pd.DataFrame.from_dict(
        labels_dict, orient="index", columns=["label_column"]
    )

    return labels_df


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
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
    best_model: The best estimator found by the random search, ready for prediction.
    """

    # Create a pipeline with scaling, PCA, and SVM
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
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
        pipeline,
        param_dist,
        n_iter=100,
        cv=5,
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


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
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
    best_model: The best estimator found by the random search, which is a RandomForestClassifier.
    """

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid
    n_estimators = [
        int(x) for x in np.linspace(start=200, stop=1500, num=10)
    ]  # 200-2000
    max_features = ["sqrt", "log2", None]
    max_depth = [int(x) for x in np.linspace(5, 50, num=11)]  # 10-110
    min_samples_split = [10, 20, 30]  # 2, 5, 10
    min_samples_leaf = [5, 10, 20]  # 1, 2, 4
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
        rf,
        param_dist,
        n_iter=100,
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )

    # Fit RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return best_model


def train_logreg(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
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
    best_model: The best estimator found by the random search, which is a Pipeline containing PCA and LogisticRegression.
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
        pipeline,
        param_dist,
        n_iter=100,
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )

    # Fit RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)

    # Retrieve the best model from the random search
    best_model = random_search.best_estimator_

    return best_model
