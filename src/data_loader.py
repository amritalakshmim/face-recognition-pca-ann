import os
import cv2
import numpy as np


class FaceDataLoader:
    """
    Load and preprocess face images
    from the dataset folder.

    This class reads images from
    person-wise folders, converts
    them into grayscale, resizes
    them to a fixed size, and
    transforms each image into
    a flattened column vector.

    Attributes
    ----------
    dataset_path : str
        Path to the dataset folder.

    image_size : tuple
        Target image dimensions
        (width, height).
    """

    def __init__(
        self,
        dataset_path,
        image_size=(100, 100)
    ):
        """
        Initialize FaceDataLoader.

        Parameters
        ----------
        dataset_path : str
            Path containing
            face image folders.

        image_size : tuple, optional
            Size to resize all
            images into.
            Default is (100, 100).
        """

        self.dataset_path = dataset_path
        self.image_size = image_size

    def load_dataset(self):
        """
        Load and preprocess
        face images.

        Steps:
        1. Read each image.
        2. Convert to grayscale.
        3. Resize image.
        4. Flatten into vector.
        5. Store labels.

        Returns
        -------
        face_database : numpy.ndarray
            Matrix of face images
            with shape:
            (pixels, total_images)

        labels : numpy.ndarray
            Numerical labels
            corresponding to
            each image.

        label_map : dict
            Mapping between
            label IDs and
            person names.
        """

        faces = []
        labels = []
        label_map = {}

        label_id = 0

        # Iterate through
        # each person's folder
        for person_name in os.listdir(
            self.dataset_path
        ):

            person_path = os.path.join(
                self.dataset_path,
                person_name
            )

            # Skip non-folder files
            if not os.path.isdir(
                person_path
            ):
                continue

            # Store label mapping
            label_map[
                label_id
            ] = person_name

            # Read all images
            # inside folder
            for image_name in os.listdir(
                person_path
            ):

                image_path = os.path.join(
                    person_path,
                    image_name
                )

                # Read image
                image = cv2.imread(
                    image_path
                )

                # Skip corrupted images
                if image is None:
                    continue

                # Convert image
                # to grayscale
                gray_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                # Resize image
                resized_image = cv2.resize(
                    gray_image,
                    self.image_size
                )

                # Convert image
                # matrix to vector
                flattened_image = (
                    resized_image
                    .flatten()
                )

                # Store face vector
                faces.append(
                    flattened_image
                )

                # Store label
                labels.append(
                    label_id
                )

            label_id += 1

        # Convert into
        # NumPy arrays
        face_database = (
            np.array(faces).T
        )

        labels = np.array(
            labels
        )

        return (
            face_database,
            labels,
            label_map
        )