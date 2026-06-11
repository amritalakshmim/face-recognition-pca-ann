import matplotlib.pyplot as plt
import numpy as np


def show_sample_faces(
    face_database,
    labels,
    label_map,
    image_size=(100, 100),
    samples=5
):
    """
    Display random sample
    face images from dataset.

    This function visualizes
    randomly selected face
    images along with their
    corresponding labels.

    Parameters
    ----------
    face_database :
    numpy.ndarray
        Matrix containing
        flattened face images.

    labels :
    numpy.ndarray
        Class labels for
        each image.

    label_map : dict
        Mapping between
        label IDs and
        person names.

    image_size : tuple
        Face image dimensions.
        Default is (100, 100).

    samples : int
        Number of random
        faces to display.
        Default is 5.
    """

    # Select random images
    random_indices = (
        np.random.choice(
            len(labels),
            samples,
            replace=False
        )
    )

    plt.figure(
        figsize=(15, 5)
    )

    for i, idx in enumerate(
        random_indices
    ):

        # Convert vector
        # back to image
        image = (
            face_database[:, idx]
            .reshape(image_size)
        )

        # Get label name
        label = (
            label_map[
                labels[idx]
            ]
        )

        plt.subplot(
            1,
            samples,
            i + 1
        )

        plt.imshow(
            image,
            cmap="gray"
        )

        plt.title(label)

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_eigenfaces(
    eigenfaces,
    image_size=(100, 100),
    rows=2,
    cols=5
):
    """
    Display generated
    eigenfaces.

    Eigenfaces represent
    the most important
    facial feature directions
    extracted using PCA.

    Parameters
    ----------
    eigenfaces :
    numpy.ndarray
        Matrix containing
        generated eigenfaces.

    image_size : tuple
        Dimensions used
        to reshape images.

    rows : int
        Number of rows
        in visualization.

    cols : int
        Number of columns
        in visualization.
    """

    plt.figure(
        figsize=(12, 6)
    )

    total_faces = (
        rows * cols
    )

    for i in range(
        total_faces
    ):

        # Convert eigenface
        # vector to image
        eigenface = (
            eigenfaces[i, :]
            .reshape(image_size)
        )

        plt.subplot(
            rows,
            cols,
            i + 1
        )

        plt.imshow(
            eigenface,
            cmap="gray"
        )

        plt.title(
            f"Eigenface {i+1}"
        )

        plt.axis("off")

    plt.tight_layout()

    # Save visualization
    plt.savefig(
        "outputs/eigenfaces/"
        "eigenfaces.png"
    )

    plt.show()