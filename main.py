from src.data_loader import FaceDataLoader
from src.utils import show_sample_faces
from src.pca import PCAFaceRecognizer


dataset_path = "dataset/faces"

loader = FaceDataLoader(dataset_path)

face_database, labels, label_map = (
    loader.load_dataset()
)

print("Face Database Shape:")
print(face_database.shape)

pca = PCAFaceRecognizer(k=50)

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

feature_vectors = (
    pca.select_top_k_components(
        eigenvalues,
        eigenvectors
    )
)

print("\nTop K Feature Vectors Shape:")
print(feature_vectors.shape)

eigenfaces = (
    pca.generate_eigenfaces(
        feature_vectors
    )
)

print("\nEigenfaces Shape:")
print(eigenfaces.shape)