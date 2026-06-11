from src.data_loader import (
    FaceDataLoader
)

from src.utils import (
    show_eigenfaces
)

from src.pca import (
    PCAFaceRecognizer
)

from src.train import (
    split_dataset
)

from src.ann_model import (
    ANNFaceClassifier
)

from src.evaluate import (
    plot_accuracy_vs_k
)

from src.test import (
    test_imposters
)


def run_pca_pipeline(
    face_database,
    labels,
    k
):
    """
    Execute complete PCA
    face recognition pipeline.

    Steps:
    ------
    1. Mean face calculation
    2. Mean-zero transformation
    3. Covariance matrix
    4. Eigen decomposition
    5. Top-k selection
    6. Eigenface generation
    7. Face signature creation
    8. Train/test split
    9. ANN training
    10. Accuracy evaluation

    Parameters
    ----------
    face_database :
    numpy.ndarray
        Face image matrix.

    labels :
    numpy.ndarray
        Face labels.

    k : int
        Number of PCA
        components.

    Returns
    -------
    tuple
        pca
        eigenfaces
        scaler
        ann_classifier
        accuracy
    """

    pca = PCAFaceRecognizer(
        k=k
    )

    pca.label_map = (
        label_map
    )

    pca.compute_mean_face(
        face_database
    )

    pca.mean_zero_faces(
        face_database
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
        ann_classifier
        .calculate_accuracy(
            X_test,
            y_test
        )
    )

    return (
        pca,
        eigenfaces,
        scaler,
        ann_classifier,
        accuracy
    )


print(
    "\nTraining PCA + "
    "ANN Face Recognition "
    "System..."
)

# Load dataset
dataset_path = (
    "dataset/faces"
)

loader = FaceDataLoader(
    dataset_path
)

(
    face_database,
    labels,
    label_map
) = loader.load_dataset()

# Best k value
best_k = 75

(
    pca,
    eigenfaces,
    scaler,
    ann_classifier,
    accuracy
) = run_pca_pipeline(
    face_database,
    labels,
    best_k
)

# Display eigenfaces
show_eigenfaces(
    eigenfaces
)

print(
    "\nBest Model Results"
)

print("-" * 30)

print(
    f"Best k Value: "
    f"{best_k}"
)

print(
    f"Model Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# Accuracy vs k
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

    (
        _,
        _,
        _,
        _,
        accuracy
    ) = run_pca_pipeline(
        face_database,
        labels,
        k
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

# Test unknown faces
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