import numpy as np


class PCAFaceRecognizer:
    def __init__(self, k=50):
        self.k = k

        self.mean_face = None
        self.mean_centered_faces = None

    def compute_mean_face(
            self,
            face_database
    ):
        
        self.mean_face = np.mean(
            face_database,
            axis=1,
            keepdims=True
        )

        return self.mean_face
    
    def mean_zero_faces(
            self,
            face_database
    ):
        
        self.mean_centered_faces = (
            face_database - 
            self.mean_face
        )

        return self.mean_centered_faces
    
    def compute_covariance_matrix(
            self
    ):
        
        covariance_matrix = np.dot(
            self.mean_centered_faces.T,
            self.mean_centered_faces
        )

        return covariance_matrix
    
    def compute_eigen_components(
            self,
            covariance_matrix
    ):
        
        eigenvalues, eigenvectors = np.linalg.eig(
            covariance_matrix
        )

        return eigenvalues, eigenvectors
    
    def select_top_k_components(
            self,
            eigenvalues,
            eigenvectors
    ):

        sorted_indices = np.argsort(
            eigenvalues
        )[::-1]

        sorted_eigenvectors = (
            eigenvectors[:,sorted_indices]
        )

        selected_vectors = (
            sorted_eigenvectors[:, :self.k]
        )

        feature_vectors = np.dot(
            self.mean_centered_faces,
            selected_vectors
        )

        return feature_vectors
    
    def generate_eigenfaces(
            self,
            feature_vectors
    ):
        eigenfaces = []

        for i in range(
            feature_vectors.shape[1]
        ):
            eigenface = (
                feature_vectors[:, i]
            )

            normalized_face = (
                eigenface / 
                np.linalg.norm(
                    eigenface
                )
            )

            eigenfaces.append(
                normalized_face
            )

        eigenfaces = np.array(
            eigenfaces
        ).T

        return eigenfaces