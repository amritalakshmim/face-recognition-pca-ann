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

    print("\nTesting Imposters:\n")

    for image_name in os.listdir(
        imposter_folder
    ):

        image_path = os.path.join(
            imposter_folder,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            image_size
        )

        image_vector = image.flatten()

        image_vector = (
            image_vector.reshape(
                -1,
                1
            )
        )

        mean_zero_face = (
            image_vector -
            pca.mean_face
        )

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

        signature = scaler.transform(
            signature
        )

        prediction = (
            ann_classifier
            .predict_with_unknown(
                signature,
                threshold=0.75
            )
        )[0]

        if prediction != "Not Enrolled":
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