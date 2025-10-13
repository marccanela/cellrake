# Created by: Marc Canela

from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from tqdm import tqdm

from cellrake.utils import (
    create_stats_dict,
    crop_cell_large,
    evaluate_model,
    train_model,
)

# ================================================================================
# USER INTERACTION UTILITIES
# ================================================================================


def user_input(
    roi_values: Dict[str, np.ndarray], layer: np.ndarray, padding: int = 120
) -> str:
    """
    Display ROI overlaid on image and prompt user for cell/non-cell classification.

    Parameters
    ----------
    roi_values : Dict[str, np.ndarray]
        Dictionary with 'x' and 'y' keys containing ROI coordinates.
    layer : np.ndarray
        2D image array for ROI visualization.
    padding : int, default=120
        Padding pixels for cropped ROI visualization.

    Returns
    -------
    str
        User classification: '1' for cell, '0' for non-cell.
    """
    x_coords, y_coords = roi_values["x"], roi_values["y"]

    # Set up the plot with four subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # Full image with ROI highlighted
    axes[0].imshow(layer, cmap="viridis")
    axes[0].plot(x_coords, y_coords, "m-", linewidth=1)
    axes[0].axis("off")  # Hide the axis

    # Full image without ROI highlighted
    axes[1].imshow(layer, cmap="viridis")
    axes[1].axis("off")  # Hide the axis

    # Cropped image with padding, ROI highlighted
    layer_cropped_small, x_coords_cropped, y_coords_cropped = crop_cell_large(
        layer, x_coords, y_coords, padding=padding
    )
    axes[2].imshow(layer_cropped_small, cmap="viridis")
    axes[2].plot(x_coords_cropped, y_coords_cropped, "m-", linewidth=1)
    axes[2].axis("off")  # Hide the axis

    # Cropped image without ROI highlighted
    axes[3].imshow(layer_cropped_small, cmap="viridis")
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


def manual_labeling(
    features_df: pd.DataFrame, rois: Dict[str, dict], layers: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Prompt user to manually label ROIs for training data.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features where each row is a sample.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.

    Returns
    -------
    pd.DataFrame
        DataFrame with manual labels in "label_column".
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


# ================================================================================
# SEMI-SUPERVISED LEARNING ALGORITHMS
# ================================================================================


def label_speading(
    subset_df: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    samples: int,
    label_spreading_kernel: str,
    random_state: int,
) -> pd.DataFrame:
    """
    Apply semi-supervised learning using label spreading for pseudo-labeling.

    Parameters
    ----------
    subset_df : pd.DataFrame
        DataFrame with features and cluster assignments.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    samples : int
        Number of samples to manually label from each cluster.
    label_spreading_kernel : str
        Kernel type for label spreading algorithm.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with pseudo-labels and confidence metrics.
    """
    # Identify the nature of the clusters
    pool_X = subset_df.copy()

    # Explore and label the clusters
    exploratory_dfs = []
    for cluster in pool_X["cluster"].unique():
        cluster_df = pool_X[pool_X["cluster"] == cluster]
        if len(cluster_df) >= samples:
            sampled_df = cluster_df.sample(n=samples, random_state=random_state)
        else:
            sampled_df = cluster_df
        exploratory_dfs.append(sampled_df)

    exploratory_df = pd.concat(exploratory_dfs)
    exploratory_df_labeled = manual_labeling(exploratory_df, rois, layers)

    # Extract labelled features and labels
    X_labeled = exploratory_df.drop(columns="cluster").values
    y_labeled = exploratory_df_labeled["label_column"].values.astype(int)

    # Standardize the labeled data
    scaler = StandardScaler()
    X_labeled_standardized = scaler.fit_transform(X_labeled)

    # Oversample the minority class
    smote = SMOTE(sampling_strategy="minority", random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_labeled_standardized, y_labeled)

    # Standardize the pool_X data
    X_pool = pool_X.drop(columns="cluster").values
    X_pool_standardized = scaler.transform(X_pool)  # Use the same scaler as above

    # Combine the resampled labeled data with the standardized pool_X (unlabeled data)
    X_combined = np.vstack((X_resampled, X_pool_standardized))
    y_combined = np.concatenate([y_resampled, [-1] * len(X_pool)])

    # Initialize and fit the LabelSpreading model
    ls = LabelSpreading(kernel=label_spreading_kernel)
    ls.fit(X_combined, y_combined)

    # Separate back into labeled and pool data
    labels_combined = ls.transduction_
    labels_pool = labels_combined[len(X_resampled) :]

    # Compute probabilities
    probabilities = ls.label_distributions_
    pool_probabilities = probabilities[len(X_resampled) :]
    is_zero_probabilities = np.all(pool_probabilities == [0, 0], axis=1)
    myentropy = entropy(pool_probabilities.T, axis=0)

    # Create new total_df
    total_df = subset_df.copy()
    total_df["labels"] = labels_pool
    total_df["is_zero_prob"] = is_zero_probabilities
    total_df["myentropy"] = myentropy

    # Keep original manually labelled
    exploratory_df_labeled["label_column"] = exploratory_df_labeled[
        "label_column"
    ].astype(int)
    for idx in exploratory_df_labeled.index:
        if idx in total_df.index:
            total_df.at[idx, "labels"] = exploratory_df_labeled.at[idx, "label_column"]
    total_df["manual"] = False
    common_indices = total_df.index.intersection(exploratory_df_labeled.index)
    total_df.loc[common_indices, "manual"] = True

    return total_df


# ================================================================================
# VISUALIZATION UTILITIES
# ================================================================================


def plot_pca(
    total_df: pd.DataFrame,
    project_folder: Union[str, Path],
    entropy_threshold: float,
    plot_dpi: int,
    random_state: int,
) -> None:
    """
    Create 2D PCA visualization of labeled and pseudo-labeled data.

    Parameters
    ----------
    total_df : pd.DataFrame
        DataFrame with features, labels, and confidence metrics.
    project_folder : Union[str, Path]
        Path to project folder for saving the plot.
    entropy_threshold : float
        Entropy threshold for determining confidence in pseudo-labels.
    plot_dpi : int
        DPI resolution for saved plot.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    None
    """
    # Create colormaps
    pastel1_colors = ["#fbb4ae", "#b3cde3"]  # First two colors of Pastel1
    set1_colors = ["#e41a1c", "#377eb8"]  # First two colors of Set1

    # Reduce the dimensions of the data to 2D using PCA for visualization
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=random_state)),
        ]
    )
    X = total_df.drop(
        columns=["cluster", "labels", "is_zero_prob", "myentropy", "manual"]
    ).values
    X = pipeline.fit_transform(X)

    # Rotated loading scores
    loading_scores = pipeline.named_steps["pca"].components_
    max_contributing_features = np.argmax(np.abs(loading_scores), axis=1)
    feature_names = total_df.drop(
        columns=["cluster", "labels", "is_zero_prob", "myentropy", "manual"]
    ).columns
    most_contributing_features = [feature_names[i] for i in max_contributing_features]

    # Build masks
    label_0 = total_df.labels == 0
    label_1 = total_df.labels == 1
    correct_prob = total_df.is_zero_prob == False
    confident = total_df.myentropy < entropy_threshold
    manual = total_df.manual == True

    # Plot the data points
    plt.figure()

    # Negative pseudo-labeled
    mask_negative_pseudo = label_0 & correct_prob & confident & ~manual
    X_negative_pseudo = X[mask_negative_pseudo]
    plt.scatter(
        X_negative_pseudo[:, 0],
        X_negative_pseudo[:, 1],
        color=pastel1_colors[1],
        marker="x",
        label="Negative Pseudo-labeled",
    )

    # Positive pseudo-labeled
    mask_positive_pseudo = label_1 & correct_prob & confident & ~manual
    X_positive_pseudo = X[mask_positive_pseudo]
    plt.scatter(
        X_positive_pseudo[:, 0],
        X_positive_pseudo[:, 1],
        color=pastel1_colors[0],
        marker="x",
        label="Positive Pseudo-labeled",
    )

    # Negative labeled
    mask_negative_manual = label_0 & manual
    X_negative_manual = X[mask_negative_manual]
    plt.scatter(
        X_negative_manual[:, 0],
        X_negative_manual[:, 1],
        color=set1_colors[1],
        edgecolors="k",
        marker="o",
        label="Negative Labeled",
    )

    # Positive labeled
    mask_positive_manual = label_1 & manual
    X_positive_manual = X[mask_positive_manual]
    plt.scatter(
        X_positive_manual[:, 0],
        X_positive_manual[:, 1],
        color=set1_colors[0],
        edgecolors="k",
        marker="o",
        label="Positive Labeled",
    )

    # Add legend and title
    plt.legend()
    plt.title("2D PCA Visualization of Labeled and Pseudo-labeled Data")
    plt.xlabel(f"PC1 ({most_contributing_features[0]})")
    plt.ylabel(f"PC2 ({most_contributing_features[1]})")

    # Save the plot as a PNG file
    output_path = f"{project_folder}/pca_label_spreading.png"
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close()


def plot_pca_train_test(
    total_df: pd.DataFrame,
    X: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    project_folder: Union[str, Path],
    plot_dpi: int,
    random_state: int,
) -> None:
    """
    Create 2D PCA visualization of training and testing data splits.

    Parameters
    ----------
    total_df : pd.DataFrame
        DataFrame with features for determining feature names.
    X : np.ndarray
        Complete feature array for fitting PCA.
    X_train : np.ndarray
        Training feature array.
    X_test : np.ndarray
        Testing feature array.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Testing labels.
    project_folder : Union[str, Path]
        Path to project folder for saving the plot.
    plot_dpi : int
        DPI resolution for saved plot.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    None
    """
    # Reduce to 2D using PCA
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=random_state)),
        ]
    )
    pipeline.fit(X)
    X_2d_train = pipeline.transform(X_train)
    X_2d_test = pipeline.transform(X_test)

    # Create colormaps
    set1_colors = ["#e41a1c", "#377eb8"]  # First two colors of Set1

    # Rotated loading scores
    loading_scores = pipeline.named_steps["pca"].components_
    max_contributing_features = np.argmax(np.abs(loading_scores), axis=1)
    feature_names = total_df.drop(
        columns=["cluster", "labels", "is_zero_prob", "myentropy", "manual"]
    ).columns
    most_contributing_features = [feature_names[i] for i in max_contributing_features]

    # Build masks
    label_0_train = y_train == 0
    label_1_train = y_train == 1
    label_0_test = y_test == 0
    label_1_test = y_test == 1

    # Negative Test
    X_2d_test_0 = X_2d_test[label_0_test]
    plt.scatter(
        X_2d_test_0[:, 0],
        X_2d_test_0[:, 1],
        color=set1_colors[1],
        marker="x",
        label="Negative Test",
    )

    # Positive Test
    X_2d_test_1 = X_2d_test[label_1_test]
    plt.scatter(
        X_2d_test_1[:, 0],
        X_2d_test_1[:, 1],
        color=set1_colors[0],
        marker="x",
        label="Positive Test",
    )

    # Negative Train
    X_2d_train_0 = X_2d_train[label_0_train]
    plt.scatter(
        X_2d_train_0[:, 0],
        X_2d_train_0[:, 1],
        color=set1_colors[1],
        edgecolors="k",
        marker="o",
        label="Negative Train",
    )

    # Positive Train
    X_2d_train_1 = X_2d_train[label_1_train]
    plt.scatter(
        X_2d_train_1[:, 0],
        X_2d_train_1[:, 1],
        color=set1_colors[0],
        edgecolors="k",
        marker="o",
        label="Positive Train",
    )

    # Add legend and title
    plt.legend()
    plt.title("2D PCA Visualization of Train-Test")
    plt.xlabel(f"PC1 ({most_contributing_features[0]})")
    plt.ylabel(f"PC2 ({most_contributing_features[1]})")

    # Save the plot as a PNG file
    output_path = f"{project_folder}/pca_train_test.png"
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close()


# ================================================================================
# WORKFLOW ORCHESTRATION
# ================================================================================


def handle_pseudo_labels(
    project_folder: Union[str, Path],
    subset_df: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    samples: int,
    label_spreading_kernel: str,
    plot_entropy_threshold: float,
    plot_dpi: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Generate pseudo-labels using label spreading and create visualization.

    Parameters
    ----------
    project_folder : Union[str, Path]
        Path to project folder for saving plots.
    subset_df : pd.DataFrame
        DataFrame with features and cluster assignments.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    samples : int
        Number of samples to manually label from each cluster.
    label_spreading_kernel : str
        Kernel type for label spreading algorithm.
    plot_entropy_threshold : float
        Entropy threshold for plot visualization confidence.
    plot_dpi : int
        DPI resolution for saved plots.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with pseudo-labels and confidence metrics.
    """
    total_df = label_speading(
        subset_df,
        rois,
        layers,
        samples,
        label_spreading_kernel,
        random_state,
    )
    plot_pca(total_df, project_folder, plot_entropy_threshold, plot_dpi, random_state)
    print("Pseudo-labeled dataset generated.")
    return total_df


# ================================================================================
# MAIN EXPORTED FUNCTIONS
# ================================================================================


def create_subset_df(
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    pca_variance_ratio: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Extract features from ROIs and cluster them into groups for training.

    Parameters
    ----------
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    pca_variance_ratio : float, default=0.95
        Proportion of variance to retain in PCA dimensionality reduction.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with ROI features and cluster assignments.
    """

    # Extract statistical features from each ROI
    roi_props_dict = {}
    for tag in tqdm(rois.keys(), desc="Extracting input features", unit="image"):
        roi_dict = rois[tag]
        layer = layers[tag]

        # Check if roi_dict is empty
        if not roi_dict:
            continue

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
            ("pca", PCA(n_components=pca_variance_ratio, random_state=random_state)),
            ("kmeans", KMeans(n_clusters=2, random_state=random_state)),
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


def train_classifier(
    subset_df: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    samples: int,
    model_type: str,
    project_folder: Union[str, Path],
    entropy_threshold: float = 0.025,
    max_train_samples: int = 1000,
    max_test_samples: int = 1000,
    dataset_size_threshold: int = 2000,
    default_train_ratio: float = 0.8,
    label_spreading_kernel: str = "knn",
    plot_entropy_threshold: float = 0.2,
    plot_dpi: int = 300,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train a classifier using semi-supervised learning with label spreading.

    Parameters
    ----------
    subset_df : pd.DataFrame
        DataFrame with ROI features and cluster assignments.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    samples : int
        Number of samples to draw from each cluster for initial labeling.
    model_type : str
        Type of model to train ('svm', 'rf', 'et', or 'logreg').
    project_folder : Union[str, Path]
        Path to project folder for saving results and models.
    entropy_threshold : float, default=0.025
        Confidence threshold for pseudo-label selection based on entropy.
    max_train_samples : int, default=1000
        Maximum number of training samples when dataset is large.
    max_test_samples : int, default=1000
        Maximum number of testing samples when dataset is large.
    dataset_size_threshold : int, default=2000
        Dataset size above which to use fixed sample limits.
    default_train_ratio : float, default=0.8
        Training ratio for standard train/test split on smaller datasets.
    label_spreading_kernel : str, default="knn"
        Kernel type for label spreading algorithm.
    plot_entropy_threshold : float, default=0.2
        Entropy threshold for plot visualization confidence.
    plot_dpi : int, default=300
        DPI resolution for saved plots.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[Pipeline, pd.DataFrame]
        Trained classifier pipeline and performance metrics DataFrame.
    """
    # Label spreading
    total_df = handle_pseudo_labels(
        project_folder,
        subset_df,
        rois,
        layers,
        samples,
        label_spreading_kernel,
        plot_entropy_threshold,
        plot_dpi,
        random_state,
    )

    # Prepare features and labels
    print("Proceeding with the preliminar model...")
    correct_prob = total_df.is_zero_prob == False
    confident = total_df.myentropy < entropy_threshold
    mask = correct_prob & confident

    raw_features = total_df.drop(
        columns=["cluster", "labels", "is_zero_prob", "myentropy", "manual"]
    ).values
    raw_labels = total_df.labels.values.astype(int)
    raw_clusters = total_df.cluster.values.astype(int)

    X = raw_features[mask]
    y = raw_labels[mask]
    clusters = raw_clusters[mask]

    # Split into train/test
    length = len(X)

    if length > dataset_size_threshold:
        train_size = max_train_samples / length

        # Initial split with proportional stratification
        X_train, X_temp, y_train, y_temp, _, clusters_temp = train_test_split(
            X,
            y,
            clusters,
            train_size=train_size,
            random_state=random_state,
            stratify=clusters,
        )

        # Adjust the remaining temp data to ensure exactly max_test_samples samples for testing
        test_size_adjusted = max_test_samples / len(X_temp)
        X_test, _, y_test, _, _, _ = train_test_split(
            X_temp,
            y_temp,
            clusters_temp,
            train_size=test_size_adjusted,
            random_state=random_state,
            stratify=clusters_temp,
        )
    else:
        # Standard train/test split
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X,
            y,
            clusters,
            train_size=default_train_ratio,
            random_state=random_state,
            stratify=clusters,
        )

    plot_pca_train_test(
        total_df,
        X,
        X_train,
        X_test,
        y_train,
        y_test,
        project_folder,
        plot_dpi,
        random_state,
    )

    # Train classifier
    best_model = train_model(X_train, y_train, model_type)

    # Evaluations
    print(f"Proceeding with evaluating the {model_type} classifier...")
    metrics_train = evaluate_model(best_model, X_train, y_train)
    metrics_test = evaluate_model(best_model, X_test, y_test)
    metrics_combined = pd.DataFrame({"Train": metrics_train, "Test": metrics_test})
    print(metrics_combined)

    return best_model, metrics_combined
