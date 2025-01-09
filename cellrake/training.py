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
from sklearn.metrics import (
    auc,
    f1_score,
    log_loss,
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

from cellrake.utils import balance_classes, create_stats_dict, crop_cell_large


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

    if features_df.empty:
        raise ValueError("The features DataFrame is empty. Please provide a valid one.")

    # Cluster the features to aproximate the positive/negative classes
    kmeans_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("kmeans", KMeans(n_clusters=10, random_state=42)),
        ]
    )
    best_clusters = kmeans_pipeline.fit_predict(features_df)
    features_df["cluster"] = best_clusters

    subset_df = features_df.copy()

    # # Equilibrate both classes
    # cluster_counts = features_df["cluster"].value_counts()
    # size = np.min(cluster_counts)

    # sampled_dfs = []
    # for cluster in features_df["cluster"].unique():
    #     cluster_df = features_df[features_df["cluster"] == cluster]

    #     if len(cluster_df) >= size:
    #         sampled_df = cluster_df.sample(n=size, random_state=42)
    #     else:
    #         sampled_df = cluster_df

    #     sampled_dfs.append(sampled_df)

    # subset_df = pd.concat(sampled_dfs)

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

    # Identify the nature of the clusters
    pool_X = subset_df.copy()
    clusters = pool_X["cluster"].unique()

    exploratory_dfs = []
    for cluster in clusters:
        cluster_df = pool_X[pool_X["cluster"] == cluster]
        if len(cluster_df) >= 5:
            sampled_df = cluster_df.sample(n=5, random_state=42)
        else:
            sampled_df = cluster_df
        exploratory_dfs.append(sampled_df)

    exploratory_df = pd.concat(exploratory_dfs)
    pool_X.drop(exploratory_df.index, inplace=True)  # Unselected entries

    exploratory_df_labeled = manual_labeling(exploratory_df, rois, layers)
    labelled_X = exploratory_df
    labelled_y = exploratory_df_labeled

    while True:
        positives = labelled_y["label_column"].astype(int).sum()
        print(f"You have identified {positives} positive segmentations.")
        if pool_X.empty:
            break
        more_labeling = input("Do you want to label more data? (y/n): ").strip().lower()
        if more_labeling == "n":
            break

        # Identify clusters enriched in manual labeling "1" and "0"
        enriched_clusters_1 = []
        enriched_clusters_0 = []

        for cluster in clusters:
            cluster_indices = labelled_X[labelled_X["cluster"] == cluster].index
            cluster_labels = labelled_y.loc[cluster_indices, "label_column"].astype(int)
            if cluster_labels.sum() > len(cluster_labels) / 2:
                enriched_clusters_1.append(cluster)
            else:
                enriched_clusters_0.append(cluster)

        # If enriched_clusters_1 is empty, make the condition less strict
        if not enriched_clusters_1:
            for cluster in clusters:
                cluster_indices = labelled_X[labelled_X["cluster"] == cluster].index
                cluster_labels = labelled_y.loc[cluster_indices, "label_column"].astype(
                    int
                )
                if cluster_labels.sum() > 0:
                    enriched_clusters_1.append(cluster)

        # Extract 5 instances from each enriched cluster
        if len(enriched_clusters_1) > 0:
            num_clusters_1 = len(enriched_clusters_1)
            num_clusters_0 = min(len(enriched_clusters_0), num_clusters_1)

            selected_clusters_0 = np.random.choice(
                enriched_clusters_0, num_clusters_0, replace=False
            ).tolist()
            selected_clusters = enriched_clusters_1 + selected_clusters_0
        else:
            selected_clusters = clusters

        exploratory_dfs = []
        for cluster in selected_clusters:
            cluster_df = pool_X[pool_X["cluster"] == cluster]
            if len(cluster_df) >= 5:
                sampled_df = cluster_df.sample(n=5, random_state=42)
            else:
                sampled_df = cluster_df
            exploratory_dfs.append(sampled_df)

        exploratory_df = pd.concat(exploratory_dfs)
        pool_X = pool_X.drop(exploratory_df.index)

        exploratory_df_labeled = manual_labeling(exploratory_df, rois, layers)
        labelled_X = pd.concat([labelled_X, exploratory_df], ignore_index=True)
        labelled_y = pd.concat([labelled_y, exploratory_df_labeled], ignore_index=True)

    pool_X = pool_X.drop(columns=["cluster"])

    # Balance train dataset
    labelled_Xy = pd.concat([labelled_X, labelled_y], axis=1)
    labelled_Xy = labelled_Xy.drop(columns=["cluster"])
    balanced_Xy = balance_classes(labelled_Xy)

    # Prepare datasets
    X_labeled = balanced_Xy.drop(columns=["label_column"])
    y_labeled = balanced_Xy["label_column"].astype(int)

    # Initialize the ACTIVE LEARNING
    previous_loss = None
    min_delta = 0.001
    iteration = 0
    patience = 3
    models_list = []
    metrics_list = []
    while True:

        # Train
        if model_type == "svm":
            best_model, metrics = train_svm(X_labeled.values, y_labeled.values)
        elif model_type == "rf":
            best_model, metrics = train_rf(X_labeled.values, y_labeled.values)
        elif model_type == "logreg":
            best_model, metrics = train_logreg(X_labeled.values, y_labeled.values)
        else:
            print(f"Unsupported model type: {model_type}. Using 'svm' as default.")
            best_model, metrics = train_svm(X_labeled.values, y_labeled.values)

        models_list.append(best_model)
        metrics_list.append(metrics)

        iteration += 1  # Increment iteration count
        print(f"Iteration {iteration}")

        # Check if the improvement in performance is minimal
        current_loss = metrics["binary_cross_entropy_loss"]
        if previous_loss is not None:
            if current_loss > previous_loss:
                patience -= 1
                print(f"Current loss increased to {current_loss}")
                print(f"Patience left: {patience}")
                if patience == 0:
                    print(f"Stopping the iteration due to divergence")
                    break
            elif (previous_loss - current_loss) < min_delta:
                patience -= 1
                print(f"Current loss: {current_loss}")
                print(f"Minimal loss improvement ({previous_loss - current_loss})")
                print(f"Patience left: {patience}")
                if patience == 0:
                    print(f"Stopping the iteration due to minimal improvement")
                    break
            else:
                patience = 3  # Reset patience
                print(f"Current loss: {current_loss}")
        else:
            print(f"Current loss: {current_loss}")

        if pool_X.empty:
            break

        previous_loss = current_loss

        # Predict and Extract (Least Confidence Sampling)

        # Predict probabilities for the unlabeled pool
        probs = best_model.predict_proba(pool_X.values)

        # Compute uncertainties using Least Confidence
        uncertainties = 1 - np.max(probs, axis=1)

        # Define batch size
        if len(pool_X) < 20:
            batch_size = len(pool_X)
        else:
            batch_size = 20

        # Get indices of the most uncertain samples
        query_indices = np.argsort(uncertainties)[-batch_size:]

        # Extract the most uncertain samples
        X_uncertain = pool_X.iloc[query_indices]

        # Label the uncertain samples
        y_uncertain = manual_labeling(X_uncertain, rois, layers)

        # Remove the newly labeled data from the pool dataset
        pool_X = pool_X.drop(X_uncertain.index)

        # Combine the newly labeled data with the previously labeled dataset
        uncertain_Xy = pd.concat([X_uncertain, y_uncertain], axis=1)
        labelled_Xy = pd.concat([labelled_Xy, uncertain_Xy], ignore_index=True)

        # Balance the newly labeled dataset and prepare for the next iteration
        balanced_Xy = balance_classes(labelled_Xy)
        X_labeled = balanced_Xy.drop(columns=["label_column"])
        y_labeled = balanced_Xy["label_column"].astype(int)

    # Export performance scores
    performance_df = pd.DataFrame(metrics_list)

    # Roll back to the fourth model from the end
    rollback_index = -4
    best_model = models_list[rollback_index]
    print(
        f"Rolling back to the model from interation (iteration {len(models_list) + rollback_index + 1})"
    )

    # Plot the PR-AUC, Precision, Recall, F1 Score, and Loss for each iteration
    plt.figure(figsize=(12, 8))

    # Plotting each metric
    plt.plot(
        performance_df.index + 1,
        performance_df["pr_auc"],
        label="PR-AUC",
        marker="o",
    )
    plt.plot(
        performance_df.index + 1,
        performance_df["precision"],
        label="Precision",
        marker="o",
    )
    plt.plot(
        performance_df.index + 1,
        performance_df["recall"],
        label="Recall",
        marker="o",
    )
    plt.plot(
        performance_df.index + 1,
        performance_df["f1_score"],
        label="F1 Score",
        marker="o",
    )
    plt.plot(
        performance_df.index + 1,
        performance_df["binary_cross_entropy_loss"],
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

    return best_model, performance_df


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


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    This function trains an SVM model with hyperparameter tuning using RandomizedSearchCV and performs cross-validation
    to extract PR-AUC, precision, recall, F1 score, and binary cross-entropy loss.

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, a 2D array where each row is a sample and each column is a feature.

    y_train : np.ndarray
        The training labels, a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    best_model: The best estimator found by the random search, ready for prediction.
    metrics: A dictionary containing the cross-validated metrics.
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

    # Perform cross-validation to get predictions
    y_pred_proba = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba"
    )[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_train, y_pred_proba)
    pr_auc = auc(recall, precision)
    precision_score_value = precision_score(y_train, y_pred)
    recall_score_value = recall_score(y_train, y_pred)
    f1_score_value = f1_score(y_train, y_pred)
    binary_cross_entropy_loss = log_loss(y_train, y_pred_proba)

    metrics = {
        "pr_auc": pr_auc,
        "precision": precision_score_value,
        "recall": recall_score_value,
        "f1_score": f1_score_value,
        "binary_cross_entropy_loss": binary_cross_entropy_loss,
    }

    return best_model, metrics


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    This function trains a Random Forest Classifier with hyperparameter tuning using RandomizedSearchCV and performs cross-validation
    to extract PR-AUC, precision, recall, F1 score, and binary cross-entropy loss.

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, typically a 2D array where each row represents a sample and each column represents a feature.

    y_train : np.ndarray
        The training labels, typically a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    best_model: The best estimator found by the random search, which is a RandomForestClassifier.
    metrics: A dictionary containing the cross-validated metrics.
    """

    # Initialize RandomForestClassifier for hyperparameter tuning and model training
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

    # Perform cross-validation to get predictions
    y_pred_proba = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba"
    )[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_train, y_pred_proba)
    pr_auc = auc(recall, precision)
    precision_score_value = precision_score(y_train, y_pred)
    recall_score_value = recall_score(y_train, y_pred)
    f1_score_value = f1_score(y_train, y_pred)
    binary_cross_entropy_loss = log_loss(y_train, y_pred_proba)

    metrics = {
        "pr_auc": pr_auc,
        "precision": precision_score_value,
        "recall": recall_score_value,
        "f1_score": f1_score_value,
        "binary_cross_entropy_loss": binary_cross_entropy_loss,
    }

    return best_model, metrics


def train_logreg(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    This function trains a Logistic Regression model with hyperparameter tuning using RandomizedSearchCV and performs cross-validation
    to extract PR-AUC, precision, recall, F1 score, and binary cross-entropy loss.

    Parameters:
    ----------
    X_train : np.ndarray
        The training features, typically a 2D array where each row represents a sample and each column represents a feature.

    y_train : np.ndarray
        The training labels, typically a 1D array where each element is the label for the corresponding sample in X_train.

    Returns:
    -------
    best_model: The best estimator found by the random search, which is a Pipeline containing PCA and LogisticRegression.
    metrics: A dictionary containing the cross-validated metrics.
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

    # Perform cross-validation to get predictions
    y_pred_proba = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba"
    )[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_train, y_pred_proba)
    pr_auc = auc(recall, precision)
    precision_score_value = precision_score(y_train, y_pred)
    recall_score_value = recall_score(y_train, y_pred)
    f1_score_value = f1_score(y_train, y_pred)
    binary_cross_entropy_loss = log_loss(y_train, y_pred_proba)

    metrics = {
        "pr_auc": pr_auc,
        "precision": precision_score_value,
        "recall": recall_score_value,
        "f1_score": f1_score_value,
        "binary_cross_entropy_loss": binary_cross_entropy_loss,
    }

    return best_model, metrics
