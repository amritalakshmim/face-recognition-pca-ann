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
        
        U, S, Vt = np.linalg.svd(
            covariance_matrix
        )

        eigenvalues = S

        eigenvectors = U

        return (
            eigenvalues, 
            eigenvectors
        )
    
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

        top_k_eigenvectors = (
            sorted_eigenvectors[
                :, :self.k
            ]
        )

        return top_k_eigenvectors
    
    def generate_eigenfaces(
            self,
            top_k_eigenvectors
    ):
        eigenfaces = np.dot(
            top_k_eigenvectors.T,
            self.mean_centered_faces.T
        )

        norms = np.linalg.norm(
            eigenfaces,
            axis=0,
            keepdims=True
        )

        eigenfaces = (
            eigenfaces / norms
        )

        return eigenfaces
    
    def generate_face_signatures(
            self,
            eigenfaces
    ):
        face_signatures = np.dot(
            eigenfaces,
            self.mean_centered_faces
        )

        return face_signatures