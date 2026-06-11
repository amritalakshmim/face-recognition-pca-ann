import os
import cv2
import numpy as np


def test_imposters(
    imposter_folder,
    pca,
    eigenfaces,
    scaler,
    ann_classifier,
    image_size=(100, 100)
):
    """
    Test unknown/imposter faces.

    This function checks whether
    a face belongs to the enrolled
    dataset or should be classified
    as:

    'Not Enrolled'

    Steps:
    ------
    1. Read imposter image
    2. Convert image into vector
    3. Perform mean-zero
       transformation
    4. Project image onto
       eigenfaces
    5. Generate face signature
    6. Scale features
    7. Predict identity
       using ANN

    Parameters
    ----------
    imposter_folder : str
        Path containing
        unknown face images.

    pca : PCAFaceRecognizer
        Trained PCA model.

    eigenfaces :
    numpy.ndarray
        Generated eigenfaces.

    scaler :
    StandardScaler
        Trained scaler used
        during ANN training.

    ann_classifier :
    ANNFaceClassifier
        Trained ANN model.

    image_size : tuple
        Resize dimensions.
        Default is (100, 100).
    """

    print(
        "\nTesting Imposters:\n"
    )

    # Read all imposter images
    for image_name in os.listdir(
        imposter_folder
    ):

        image_path = os.path.join(
            imposter_folder,
            image_name
        )

        # Read image in grayscale
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Skip invalid images
        if image is None:
            continue

        # Resize image
        image = cv2.resize(
            image,
            image_size
        )

        # Convert image
        # to vector
        image_vector = (
            image.flatten()
        )

        image_vector = (
            image_vector.reshape(
                -1,
                1
            )
        )

        # Mean-zero
        # transformation
        mean_zero_face = (
            image_vector
            - pca.mean_face
        )

        # Project onto
        # eigenfaces
        signature = np.dot(
            eigenfaces,
            mean_zero_face
        )

        signature = (
            signature.reshape(
                1,
                -1
            )
        )

        # Normalize signature
        signature = (
            scaler.transform(
                signature
            )
        )

        # Predict identity
        prediction = (
            ann_classifier
            .predict_with_unknown(
                signature,
                threshold=0.75
            )
        )[0]

        # Convert class number
        # to celebrity name
        if (
            prediction
            != "Not Enrolled"
        ):

            prediction = (
                pca.label_map[
                    prediction
                ]
            )

        print(
            f"{image_name}"
            f" → "
            f"{prediction}"
        )