import numpy as np


class PCAFaceRecognizer:
    """
    PCA-based Face Recognition system.

    This class performs the
    Principal Component Analysis
    (PCA) steps for face recognition.

    Steps:
    1. Compute mean face
    2. Mean-zero transformation
    3. Covariance matrix generation
    4. Eigen decomposition
    5. Top-k feature selection
    6. Eigenfaces generation
    7. Face signature generation

    Attributes
    ----------
    k : int
        Number of principal
        components (eigenvectors)
        to retain.

    mean_face : numpy.ndarray
        Mean face vector.

    mean_centered_faces :
    numpy.ndarray
        Mean-zero face matrix.
    """

    def __init__(
        self,
        k=50
    ):
        """
        Initialize PCA recognizer.

        Parameters
        ----------
        k : int, optional
            Number of top
            eigenvectors to keep.
            Default is 50.
        """

        self.k = k

        self.mean_face = None
        self.mean_centered_faces = None

    def compute_mean_face(
        self,
        face_database
    ):
        """
        Compute mean face vector.

        Mean face is calculated
        across all face images.

        Parameters
        ----------
        face_database :
        numpy.ndarray
            Face matrix of shape:
            (pixels, images)

        Returns
        -------
        numpy.ndarray
            Mean face vector.
        """

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
        """
        Perform mean-zero
        transformation.

        Subtract mean face
        from every face image.

        Parameters
        ----------
        face_database :
        numpy.ndarray
            Original face matrix.

        Returns
        -------
        numpy.ndarray
            Mean-centered faces.
        """

        self.mean_centered_faces = (
            face_database
            - self.mean_face
        )

        return (
            self.mean_centered_faces
        )

    def compute_covariance_matrix(
        self
    ):
        """
        Compute surrogate
        covariance matrix.

        Instead of computing
        a large covariance matrix
        of size (mn × mn),
        surrogate covariance
        is used for efficiency.

        Returns
        -------
        numpy.ndarray
            Covariance matrix.
        """

        covariance_matrix = (
            np.dot(
                self.mean_centered_faces.T,
                self.mean_centered_faces
            )
        )

        return covariance_matrix

    def compute_eigen_components(
        self,
        covariance_matrix
    ):
        """
        Compute eigenvalues
        and eigenvectors using SVD.

        Parameters
        ----------
        covariance_matrix :
        numpy.ndarray
            Surrogate covariance
            matrix.

        Returns
        -------
        tuple
            eigenvalues :
            numpy.ndarray

            eigenvectors :
            numpy.ndarray
        """

        U, S, Vt = (
            np.linalg.svd(
                covariance_matrix
            )
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
        """
        Select top-k
        principal components.

        Eigenvectors are sorted
        based on descending
        eigenvalues.

        Parameters
        ----------
        eigenvalues :
        numpy.ndarray

        eigenvectors :
        numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Top-k eigenvectors.
        """

        sorted_indices = (
            np.argsort(
                eigenvalues
            )[::-1]
        )

        sorted_eigenvectors = (
            eigenvectors[
                :,
                sorted_indices
            ]
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
        """
        Generate eigenfaces.

        Project mean-centered
        faces onto selected
        eigenvectors.

        Parameters
        ----------
        top_k_eigenvectors :
        numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Normalized eigenfaces.
        """

        eigenfaces = np.dot(
            top_k_eigenvectors.T,
            self.mean_centered_faces.T
        )

        # Normalize eigenfaces
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
        """
        Generate face signatures.

        Project each face
        image into eigenspace.

        Parameters
        ----------
        eigenfaces :
        numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Face signatures
            for classification.
        """

        face_signatures = (
            np.dot(
                eigenfaces,
                self.mean_centered_faces
            )
        )

        return face_signatures