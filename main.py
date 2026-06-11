from src.data_loader import FaceDataLoader
from src.utils import (show_sample_faces, show_eigenfaces)
from src.pca import PCAFaceRecognizer
from src.train import split_dataset
from src.ann_model import (
    ANNFaceClassifier
)
from src.evaluate import (
    plot_accuracy_vs_k
)
from src.test import (
    test_imposters
)



print(
    "\nTraining PCA + ANN "
    "Face Recognition System..."
)

dataset_path = "dataset/faces"

loader = FaceDataLoader(dataset_path)

face_database, labels, label_map = (
    loader.load_dataset()
)

pca = PCAFaceRecognizer(k=75)

pca.label_map = label_map

mean_face = pca.compute_mean_face(
    face_database
)

mean_centered_faces = (
    pca.mean_zero_faces(
        face_database
    )
)

covariance_matrix = (
    pca.compute_covariance_matrix()
)

eigenvalues, eigenvectors = (
    pca.compute_eigen_components(
        covariance_matrix
    )
)

top_k_eigenvectors = (
    pca.select_top_k_components(
        eigenvalues,
        eigenvectors
    )
)

eigenfaces = (
    pca.generate_eigenfaces(
        top_k_eigenvectors
    )
)

show_eigenfaces(
    eigenfaces
)

face_signatures = (
    pca.generate_face_signatures(
        eigenfaces
    )
)

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
) = split_dataset(
    face_signatures,
    labels
)

import numpy as np

unique, counts = np.unique(
    y_train,
    return_counts=True
)

unique, counts = np.unique(
    y_test,
    return_counts=True
)


ann_classifier = (
    ANNFaceClassifier()
)

ann_classifier.train(
    X_train,
    y_train
)

predictions = (
    ann_classifier.predict(
        X_test
    )
)

accuracy = (
    ann_classifier.calculate_accuracy(
        X_test,
        y_test
    )
)

print("\nBest Model Results")
print("-" * 30)

print(
    f"Best k Value: "
    f"{75}"
)

print(
    f"Model Accuracy: "
    f"{accuracy * 100:.2f}%"
)

k_values = [
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200
]

accuracies = []

print(
    "\nRunning Accuracy "
    "vs k Experiment..."
)

print("-" * 30)

for k in k_values:

    pca = PCAFaceRecognizer(
        k=k
    )

    mean_face = (
        pca.compute_mean_face(
            face_database
        )
    )

    mean_centered_faces = (
        pca.mean_zero_faces(
            face_database
        )
    )

    covariance_matrix = (
        pca.compute_covariance_matrix()
    )

    (
        eigenvalues,
        eigenvectors
    ) = pca.compute_eigen_components(
        covariance_matrix
    )

    top_k_eigenvectors = (
        pca.select_top_k_components(
            eigenvalues,
            eigenvectors
        )
    )

    eigenfaces = (
        pca.generate_eigenfaces(
            top_k_eigenvectors
        )
    )

    face_signatures = (
        pca.generate_face_signatures(
            eigenfaces
        )
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = split_dataset(
        face_signatures,
        labels
    )

    ann_classifier = (
        ANNFaceClassifier()
    )

    ann_classifier.train(
        X_train,
        y_train
    )

    accuracy = (
        ann_classifier.calculate_accuracy(
            X_test,
            y_test
        )
    )

    accuracy_percent = (
        accuracy * 100
    )

    accuracies.append(
        accuracy_percent
    )

    print(
    f"k = {k:<3} "
    f"→ "
    f"{accuracy_percent:.2f}%"
)

plot_accuracy_vs_k(
    k_values,
    accuracies
)

test_imposters(

    imposter_folder=
    "dataset/imposters",

    pca=pca,

    eigenfaces=
    eigenfaces,

    scaler=scaler,

    ann_classifier=
    ann_classifier
)