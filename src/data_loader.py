import os
import cv2
import numpy as np


class FaceDataLoader:
    def __init__(self, dataset_path, image_size=(100, 100)):
        self.dataset_path = dataset_path
        self.image_size = image_size

    def load_dataset(self):
        faces = []
        labels = []
        label_map = {}

        label_id = 0

        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(
                self.dataset_path,
                person_name
            )

            if not os.path.isdir(person_path):
                continue

            label_map[label_id] = person_name

            for image_name in os.listdir(person_path):
                image_path = os.path.join(
                    person_path,
                    image_name
                )

                image = cv2.imread(image_path)

                if image is None:
                    continue

                gray_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                resized_image = cv2.resize(
                    gray_image,
                    self.image_size
                )

                flattened_image = resized_image.flatten()

                faces.append(flattened_image)
                labels.append(label_id)

            label_id += 1

        face_database = np.array(faces).T
        labels = np.array(labels)

        return (
            face_database,
            labels,
            label_map
        )