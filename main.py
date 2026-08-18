# ============================================================
# MULTI-CLASS WEAPON CLASSIFICATION USING MACHINE LEARNING
# HOG (8100) + LBP (10) = 8110 FEATURES
# StandardScaler + SVM
#
# Classes:
# 0 -> Knife
# 1 -> Pistol
# 2 -> Rifle
# 3 -> AK47
# 4 -> Shotgun
# ============================================================


# =========================
# IMPORT LIBRARIES
# =========================

import os
import cv2
import joblib
import numpy as np

from skimage.feature import hog, local_binary_pattern

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================
# CONFIGURATION
# =========================

DATASET_PATH = "dataset"

MODEL_PATH = "weapon_svm_model.pkl"

IMAGE_SIZE = (128, 128)

TEST_SIZE = 0.20

RANDOM_STATE = 42


# =========================
# CLASS DEFINITIONS
# =========================

CLASSES = {
    "knife": 0,
    "pistol": 1,
    "rifle": 2,
    "ak47": 3,
    "shotgun": 4
}


LABEL_NAMES = {
    0: "Knife",
    1: "Pistol",
    2: "Rifle",
    3: "AK47",
    4: "Shotgun"
}


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    """
    Extract HOG + LBP features.

    HOG = 8100 features
    LBP = 10 features

    Total = 8110 features
    """

    # -------------------------
    # READ IMAGE
    # -------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )


    # -------------------------
    # RESIZE IMAGE
    # -------------------------

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )


    # -------------------------
    # CONVERT TO GRAYSCALE
    # -------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # -------------------------
    # HOG FEATURE EXTRACTION
    # -------------------------

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


    # -------------------------
    # LBP FEATURE EXTRACTION
    # -------------------------

    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )


    # -------------------------
    # CREATE LBP HISTOGRAM
    # -------------------------

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(11),
        range=(0, 10)
    )


    # -------------------------
    # NORMALIZE LBP FEATURES
    # -------------------------

    lbp_hist = lbp_hist.astype(
        np.float32
    )

    lbp_hist = lbp_hist / (
        lbp_hist.sum() + 1e-7
    )


    # -------------------------
    # FEATURE FUSION
    # -------------------------

    features = np.concatenate([
        hog_features,
        lbp_hist
    ])


    # Safety check
    if len(features) != 8110:

        raise ValueError(
            f"Expected 8110 features, "
            f"but got {len(features)}"
        )


    return features


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    X = []
    y = []

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)


    # Loop through every class

    for class_name, label in CLASSES.items():

        class_folder = os.path.join(
            DATASET_PATH,
            class_name
        )


        # Check folder

        if not os.path.exists(class_folder):

            print(
                f"\nWARNING: Folder not found: "
                f"{class_folder}"
            )

            continue


        print(
            f"\nProcessing class: "
            f"{class_name.upper()}"
        )


        image_count = 0


        # Loop through images

        for filename in os.listdir(class_folder):

            if not filename.lower().endswith(
                VALID_EXTENSIONS
            ):
                continue


            image_path = os.path.join(
                class_folder,
                filename
            )


            try:

                # Extract features

                features = extract_features(
                    image_path
                )


                # Store feature vector

                X.append(features)


                # Store class label

                y.append(label)


                image_count += 1


                print(
                    f"Processed: {filename}"
                )


            except Exception as e:

                print(
                    f"ERROR: {filename} -> {e}"
                )


        print(
            f"Total {class_name} images: "
            f"{image_count}"
        )


    # Convert to NumPy arrays

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(y)


    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Total images: {len(X)}")

    print(f"Feature matrix X shape: {X.shape}")

    print(f"Labels y shape: {y.shape}")


    # Display class distribution

    print("\nClass Distribution:")

    for class_name, label in CLASSES.items():

        count = np.sum(
            y == label
        )

        print(
            f"{class_name}: {count}"
        )


    return X, y


# ============================================================
# TRAIN SVM MODEL
# ============================================================

def train_model(X, y):

    print("\n" + "=" * 60)
    print("SPLITTING DATASET")
    print("=" * 60)


    # Split dataset

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )


    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )


    print("\n" + "=" * 60)
    print("TRAINING SVM MODEL")
    print("=" * 60)


    # -------------------------
    # STANDARD SCALER + SVM
    # -------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "svm",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ])


    # -------------------------
    # TRAIN MODEL
    # -------------------------

    model.fit(
        X_train,
        y_train
    )


    print(
        "\nSVM Training Completed Successfully!"
    )


    return (
        model,
        X_test,
        y_test
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)


    # Prediction

    y_pred = model.predict(
        X_test
    )


    # -------------------------
    # ACCURACY
    # -------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )


    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(

            y_test,

            y_pred,

            labels=list(
                LABEL_NAMES.keys()
            ),

            target_names=list(
                LABEL_NAMES.values()
            ),

            zero_division=0
        )
    )


    # -------------------------
    # CONFUSION MATRIX
    # -------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,

        labels=list(
            LABEL_NAMES.keys()
        )
    )


    print(
        "Confusion Matrix:\n"
    )

    print(cm)


    return accuracy


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )


    model = joblib.load(
        MODEL_PATH
    )


    return model


# ============================================================
# PREDICT NEW IMAGE
# ============================================================

def predict_weapon(
    model,
    image_path
):

    print("\n" + "=" * 60)
    print("PREDICTING IMAGE")
    print("=" * 60)


    # Extract features

    features = extract_features(
        image_path
    )


    # Convert 1D feature vector to 2D

    features = features.reshape(
        1,
        -1
    )


    # Predict class

    prediction = model.predict(
        features
    )[0]


    # Predict probabilities

    probabilities = model.predict_proba(
        features
    )[0]


    # Confidence

    confidence = np.max(
        probabilities
    )


    # Class name

    weapon_name = LABEL_NAMES[
        prediction
    ]


    print(
        f"\nPredicted Class: "
        f"{weapon_name}"
    )


    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )


    print(
        "\nAll Class Probabilities:"
    )


    # Display probability for every class

    for label, probability in zip(
        model.classes_,
        probabilities
    ):

        print(
            f"{LABEL_NAMES[label]}: "
            f"{probability * 100:.2f}%"
        )


    return (
        weapon_name,
        confidence
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------
    # STEP 1:
    # LOAD DATASET AND EXTRACT FEATURES
    # --------------------------------------------------------

    X, y = load_dataset()


    # Check if dataset is empty

    if len(X) == 0:

        raise ValueError(
            "Dataset is empty. "
            "Please check your dataset folders."
        )


    # --------------------------------------------------------
    # STEP 2:
    # TRAIN SVM MODEL
    # --------------------------------------------------------

    model, X_test, y_test = train_model(
        X,
        y
    )


    # --------------------------------------------------------
    # STEP 3:
    # EVALUATE MODEL
    # --------------------------------------------------------

    accuracy = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # STEP 4:
    # SAVE MODEL
    # --------------------------------------------------------

    save_model(
        model
    )


    # --------------------------------------------------------
    # STEP 5:
    # TEST A NEW IMAGE
    # --------------------------------------------------------

    TEST_IMAGE = (
        "test_images/test_weapon.jpg"
    )
    if os.path.exists(
        TEST_IMAGE
    ):
        predict_weapon(
            model,
            TEST_IMAGE
        )

    else:
        print(
            "\nTest image not found."
        )
        print(
            f"Add an image at: "
            f"{TEST_IMAGE}"
        )