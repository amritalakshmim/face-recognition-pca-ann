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




dataset_path = "dataset/faces"

loader = FaceDataLoader(dataset_path)

face_database, labels, label_map = (
    loader.load_dataset()
)

print("Face Database Shape:")
print(face_database.shape)

pca = PCAFaceRecognizer(k=100)

mean_face = pca.compute_mean_face(
    face_database
)

mean_centered_faces = (
    pca.mean_zero_faces(
        face_database
    )
)

print("\nMean Face Shape:")
print(mean_face.shape)

print("\nMean Centered Face Shape:")
print(mean_centered_faces.shape)


covariance_matrix = (
    pca.compute_covariance_matrix()
)

print("\nCovariance Matrix Shape:")
print(covariance_matrix.shape)

eigenvalues, eigenvectors = (
    pca.compute_eigen_components(
        covariance_matrix
    )
)

print("\nEigenvalues Shape:")
print(eigenvalues.shape)

print("\nEigenvectors Shape:")
print(eigenvectors.shape)

top_k_eigenvectors = (
    pca.select_top_k_components(
        eigenvalues,
        eigenvectors
    )
)

print("\nTop K Eigenvectors Shape:")
print(top_k_eigenvectors.shape)

eigenfaces = (
    pca.generate_eigenfaces(
        top_k_eigenvectors
    )
)

print("\nEigenfaces Shape:")
print(eigenfaces.shape)

show_eigenfaces(
    eigenfaces
)

face_signatures = (
    pca.generate_face_signatures(
        eigenfaces
    )
)

print("\nFace Signatures Shape:")
print(face_signatures.shape)

print("\nFace Signature Sample:")
print(face_signatures[:, 0][:10])

(
    X_train,
    X_test,
    y_train,
    y_test
) = split_dataset(
    face_signatures,
    labels
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Labels Shape:")
print(y_train.shape)

print("\nTesting Labels Shape:")
print(y_test.shape)

import numpy as np

print("\nTraining Label Counts:")
unique, counts = np.unique(
    y_train,
    return_counts=True
)

print(dict(zip(unique, counts)))

print("\nTesting Label Counts:")
unique, counts = np.unique(
    y_test,
    return_counts=True
)

print(dict(zip(unique, counts)))


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

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

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

for k in k_values:

    print(f"\nRunning for k = {k}")

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
        y_test
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
        f"Accuracy: "
        f"{accuracy_percent:.2f}%"
    )


print("\nFinal Results:")

for k, accuracy in zip(
    k_values,
    accuracies
):

    print(
        f"k={k}: "
        f"{accuracy:.2f}%"
    )

plot_accuracy_vs_k(
    k_values,
    accuracies
)