"""
Created on Thu Jul 25 13:07:53 2024
@author: mcanela
TRAINING AN SVM TO CLASSIFY CELLS
"""

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
from utils import create_stats_dict, crop_cell_large


def user_input(roi_dict, layer):

    labels = {}
    for roi_name, roi_info in roi_dict.items():

        x_coords, y_coords = roi_info["x"], roi_info["y"]

        # Identify as cell (1) or non-cell (0)
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        axes[0].imshow(layer, cmap="Reds")
        axes[0].plot(x_coords, y_coords, "b-", linewidth=1)
        axes[0].axis("off")  # Hide the axis

        axes[1].imshow(layer, cmap="Reds")
        axes[1].axis("off")  # Hide the axis

        layer_cropped_small, x_coords_cropped, y_coords_cropped = crop_cell_large(
            layer, x_coords, y_coords, 250
        )
        axes[2].imshow(layer_cropped_small, cmap="Reds")
        axes[2].plot(x_coords_cropped, y_coords_cropped, "b-", linewidth=1)
        axes[2].axis("off")  # Hide the axis

        axes[3].imshow(layer_cropped_small, cmap="Reds")
        axes[3].axis("off")  # Hide the axis

        plt.tight_layout()
        plt.show()
        plt.pause(0.1)

        # Ask for user input
        user_input = input("Please enter 1 or 0: ")
        while user_input not in ["1", "0"]:
            user_input = input("Invalid input. Please enter 1 or 0: ")
        labels[roi_name] = {"label": int(user_input)}
        plt.close(fig)

    return labels


def label_rois(rois, layers):

    roi_props_dict = {}
    for tag in tqdm(rois.keys(), desc="Extracting input features", unit="image"):
        roi_dict = rois[tag]
        layer = layers[tag]
        roi_props_dict[tag] = create_stats_dict(roi_dict, layer)

    input_features = {}
    for tag, all_rois in roi_props_dict.items():
        for roi_num, stats in all_rois.items():
            input_features[f"{tag}_{roi_num}"] = stats

    labels_dict = {}
    for tag in rois.keys():
        roi_dict = rois[tag]
        layer = layers[tag]
        labels_dict[tag] = user_input(roi_dict, layer)

    input_labels = {}
    for tag, all_labels in labels_dict.items():
        for roi_num, labels in all_labels.items():
            input_labels[f"{tag}_{roi_num}"] = labels

    # Combine all info into a dataframe
    data = {}
    for key in set(input_features.keys()).union(input_labels.keys()):
        if key in input_features and key in input_labels:
            data[key] = {**input_features[key], **input_labels[key]}

    df = pd.DataFrame.from_dict(data, orient="index")
    df["min_intensity"] = df["min_intensity"].astype(int)
    df["max_intensity"] = df["max_intensity"].astype(int)
    df["hog_mean"] = df["hog_mean"].astype(float)
    df["hog_std"] = df["hog_std"].astype(float)

    return df


def random_train_test_split(df, test_size=0.2):

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    return X_train, y_train, X_test, y_test


def train_svm(X_train, y_train):

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("svm", SVC(kernel="rbf", random_state=42)),
        ]
    )

    param_dist = {
        "pca__n_components": uniform(0.5, 0.5),
        "svm__C": uniform(1, 100),
        "svm__gamma": uniform(0.001, 0.1),
    }

    random_search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    return random_search, best_model


def train_rf(X_train, y_train):

    rf = RandomForestClassifier(random_state=42)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_dist = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1, verbose=2
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    return random_search, best_model


def train_logreg(X_train, y_train):

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("log_reg", LogisticRegression(random_state=42)),
        ]
    )

    param_dist = {
        "pca__n_components": uniform(0.5, 0.5),
        "log_reg__C": uniform(1, 100),
    }

    random_search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    return random_search, best_model


def evaluate(image_folder, best_model, X, y, plot=False):

    y_pred = cross_val_predict(best_model, X, y, cv=3)

    try:
        y_scores = cross_val_predict(best_model, X, y, cv=3, method="decision_function")
    except (AttributeError, NotImplementedError):
        y_scores = cross_val_predict(best_model, X, y, cv=3, method="predict_proba")
        y_scores = y_scores[:, 1]

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_scores)

    df = pd.DataFrame(
        {
            "metric": ["precision", "recall", "f1", "roc_auc"],
            "score": [precision, recall, f1, roc],
        }
    )

    evaluate_path = image_folder.parent / "evaluation.csv"
    df.to_csv(evaluate_path, index=False)

    print(f"Confusion matrix: \n{confusion_matrix(y, y_pred)}")

    if plot == True:
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
        plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
        plt.xlabel("Recall TP/(TP+FN)")
        plt.ylabel("Precision TP/(TP+FP)")
        plt.show()
