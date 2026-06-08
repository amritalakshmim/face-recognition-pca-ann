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