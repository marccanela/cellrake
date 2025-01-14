"""
@author: Marc Canela
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cellrake.utils import (
    balance_classes,
    create_stats_dict,
    crop_cell_large,
    evaluate_model,
    extract_uncertain_samples,
    train_model,
)


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

    # Ensure specific columns have the correct data types
    subset_df["min_intensity"] = subset_df["min_intensity"].astype(int)
    subset_df["max_intensity"] = subset_df["max_intensity"].astype(int)
    subset_df["hog_mean"] = subset_df["hog_mean"].astype(float)
    subset_df["hog_std"] = subset_df["hog_std"].astype(float)

    return subset_df


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


def active_learning(
    subset_df: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    model_type: str = "svm",
    metric_to_optimize: str = "f1_score",
) -> Tuple[Pipeline, pd.DataFrame]:
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
        The type of model to train. Options are 'svm', 'rf', 'et', or 'logreg'. Default is 'svm'.

    metric_to_optimize : str, optional
        The metric to optimize during active learning. Default is 'f1_score'.

    Returns:
    -------
    Tuple[Pipeline, pd.DataFrame]
        The best estimator found by active learning and a DataFrame with performance metrics.
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

    initial_positives_size = max(min(int(len(pool_X) * 0.005), 50), 25)
    n_extract_per_cluster_1 = 5

    while True:
        positives = labelled_y["label_column"].astype(int).sum()
        if positives >= initial_positives_size or pool_X.empty:
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
            if len(cluster_df) >= n_extract_per_cluster_1:
                sampled_df = cluster_df.sample(
                    n=n_extract_per_cluster_1, random_state=42
                )
            else:
                sampled_df = cluster_df
            exploratory_dfs.append(sampled_df)

        exploratory_df = pd.concat(exploratory_dfs)
        pool_X = pool_X.drop(exploratory_df.index)

        exploratory_df_labeled = manual_labeling(exploratory_df, rois, layers)
        labelled_X = pd.concat([labelled_X, exploratory_df], ignore_index=True)
        labelled_y = pd.concat([labelled_y, exploratory_df_labeled], ignore_index=True)

    # Balance train dataset
    labelled_y = labelled_y.astype(int)
    labelled_Xy = pd.concat([labelled_X, labelled_y], axis=1)
    balanced_Xy, _ = balance_classes(labelled_Xy)

    # Prepare datasets
    X_labeled = balanced_Xy.drop(columns=["label_column", "cluster"])
    y_labeled = balanced_Xy["label_column"].astype(int)

    # Pre-active learning loop
    initial_model = train_model(X_labeled.values, y_labeled.values, model_type)
    X_certain, X_uncertain, pool_X = extract_uncertain_samples(
        initial_model, pool_X, clusters, 10, skip_certain=False
    )

    if not X_certain.empty:
        y_certain = initial_model.predict(X_certain.drop(columns=["cluster"]).values)
        y_certain_df = pd.DataFrame(
            y_certain, index=X_certain.index, columns=["label_column"]
        )
        certain_Xy = pd.concat([X_certain, y_certain_df], axis=1)
        labelled_Xy = pd.concat([labelled_Xy, certain_Xy], ignore_index=True)

    y_uncertain = manual_labeling(X_uncertain, rois, layers)
    uncertain_Xy = pd.concat([X_uncertain, y_uncertain], axis=1)
    labelled_Xy = pd.concat([labelled_Xy, uncertain_Xy], ignore_index=True)
    labelled_Xy["label_column"] = labelled_Xy["label_column"].astype(int)

    # Split the labeled data into training and testing sets (80-20 split)
    labelled_Xy_train, labelled_Xy_validation = train_test_split(
        labelled_Xy, test_size=0.2, random_state=42, stratify=labelled_Xy["cluster"]
    )

    # Balance the newly labeled dataset and prepare for the next iteration
    balanced_Xy, excluded_Xy = balance_classes(labelled_Xy_train)

    # Prepare the validation set
    labelled_Xy_validation = pd.concat(
        [labelled_Xy_validation, excluded_Xy], ignore_index=True
    )
    if len(labelled_Xy_validation) > 1000:
        labelled_Xy_validation = labelled_Xy_validation.sample(n=1000, random_state=42)
    X_validation = labelled_Xy_validation.drop(columns=["label_column", "cluster"])
    y_validation = labelled_Xy_validation["label_column"].astype(int)

    # Active learning loop
    iteration = 0
    previous_performance = 0
    best_performance = 0
    best_iteration = 0
    patience = 3
    models_list = []
    metrics_train_list = []
    metrics_validation_list = []
    certain_pool = pd.DataFrame()
    uncertain_pool = pd.DataFrame()

    print("Starting active learning loop...")
    print(f"Optimizing the model based on {metric_to_optimize}")

    while True:

        iteration += 1  # Increment iteration count
        print(f"Iteration {iteration}")

        X_labeled = balanced_Xy.drop(columns=["label_column", "cluster"])
        y_labeled = balanced_Xy["label_column"].astype(int)

        best_model = train_model(X_labeled.values, y_labeled.values, model_type)
        models_list.append(best_model)

        metrics_train = evaluate_model(best_model, X_labeled.values, y_labeled.values)
        metrics_train_list.append(metrics_train)

        metrics_validation = evaluate_model(
            best_model, X_validation.values, y_validation.values
        )
        metrics_validation_list.append(metrics_validation)

        if pool_X.empty:
            break

        # Check if the improvement in performance is minimal
        current_performance = metrics_validation[metric_to_optimize]
        if current_performance <= best_performance:
            patience -= 1
            print(f"Current performance decreased to {current_performance}")
            print(f"Patience left: {patience}")
            if patience == 0:
                print(f"Stopping the iteration due to divergence")
                break
        elif (current_performance - previous_performance) < 0.01:
            patience -= 1
            print(f"Current performance: {current_performance}")
            print(
                f"Minimal performance improvement ({current_performance - previous_performance})"
            )
            print(f"Patience left: {patience}")
            if patience == 0:
                print(f"Stopping the iteration due to minimal improvement")
                break
        else:
            patience = 3  # Reset patience
            print(f"Current performance: {current_performance}")

        previous_performance = current_performance
        if current_performance > best_performance:
            best_performance = current_performance
            best_iteration = iteration

        # Predict and Extract (Least Confidence Sampling)
        X_certain, X_uncertain, pool_X = extract_uncertain_samples(
            best_model, pool_X, clusters, 5, skip_certain=True
        )

        # Combine the newly certain data with the previous dataset
        if not X_certain.empty:
            y_certain = best_model.predict(X_certain.drop(columns=["cluster"]).values)
            y_certain_df = pd.DataFrame(
                y_certain, index=X_certain.index, columns=["label_column"]
            )
            certain_Xy = pd.concat([X_certain, y_certain_df], axis=1)
            certain_Xy["label_column"] = certain_Xy["label_column"].astype(int)
            certain_pool = pd.concat([certain_pool, certain_Xy], ignore_index=True)

        if not certain_pool.empty:
            balanced_certain, excluded_certain = balance_classes(certain_pool)
            certain_pool = excluded_certain
            if not balanced_certain.empty:
                balanced_certain["label_column"] = balanced_certain[
                    "label_column"
                ].astype(int)
                balanced_Xy = pd.concat(
                    [balanced_Xy, balanced_certain], ignore_index=True
                )

        # Combine the newly uncertain data with the previous dataset
        if not X_uncertain.empty:
            y_uncertain = manual_labeling(X_uncertain, rois, layers)
            uncertain_Xy = pd.concat([X_uncertain, y_uncertain], axis=1)
            uncertain_Xy["label_column"] = uncertain_Xy["label_column"].astype(int)
            uncertain_pool = pd.concat(
                [uncertain_pool, uncertain_Xy], ignore_index=True
            )

        if not uncertain_pool.empty:
            balanced_uncertain, excluded_uncertain = balance_classes(uncertain_pool)
            if not balanced_certain.empty:
                balanced_certain["label_column"] = balanced_certain[
                    "label_column"
                ].astype(int)
                balanced_Xy = pd.concat(
                    [balanced_Xy, balanced_uncertain], ignore_index=True
                )
            if not excluded_uncertain.empty:
                certain_pool = pd.concat(
                    [certain_pool, excluded_uncertain], ignore_index=True
                )
                certain_pool["label_column"] = certain_pool["label_column"].astype(int)
                balanced_mix, excluded_mix = balance_classes(certain_pool)
                balanced_Xy = pd.concat([balanced_Xy, balanced_mix], ignore_index=True)
                certain_pool = excluded_mix

    # Export performance scores
    performance_df_train = pd.DataFrame(metrics_train_list)
    performance_df_validation = pd.DataFrame(metrics_validation_list)

    # Roll back to the best model
    best_model = models_list[best_iteration - 1]
    print(f"Rolling back to the model from interation (iteration {best_iteration})")

    # Plot the metrics
    metrics = list(metrics_validation_list[0].keys())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 20))

    for i, metric in enumerate(metrics):
        axes[i].plot(
            performance_df_train.index + 1,
            performance_df_train[metric],
            label="Training",
            marker="o",
        )
        axes[i].plot(
            performance_df_validation.index + 1,
            performance_df_validation[metric],
            label="Validation",
            marker="o",
        )
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylim([0, 1])
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()
    plt.show()

    return best_model, performance_df_train, performance_df_validation
