import cv2
import joblib
import numpy as np

from src.ann_model import ANNFaceClassifier


class FacePredictor:
    def __init__(self):

        # Load trained model
        self.ann = ANNFaceClassifier()
        self.ann.load_model("models/ann_model.pkl")

        self.scaler = joblib.load("models/scaler.pkl")
        self.label_map = joblib.load("models/label_map.pkl")

        self.mean_face = np.load("models/mean_face.npy")
        self.eigenfaces = np.load("models/eigenfaces.npy")

    def predict(self, image_path, threshold=0.75):

        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return "Invalid Image", 0.0

        # Resize to training size
        image = cv2.resize(image, (100, 100))

        # Flatten image
        image_vector = image.flatten().reshape(-1, 1)

        # Mean normalization
        mean_zero = image_vector - self.mean_face

        # Project onto eigenfaces
        signature = np.dot(self.eigenfaces, mean_zero)

        # Convert to row vector
        signature = signature.reshape(1, -1)

        # Apply scaler
        signature = self.scaler.transform(signature)

        # Predict
        prediction, confidence = self.ann.predict_with_unknown(
            signature,
            threshold
        )

        confidence = round(confidence * 100, 2)

        if prediction is None:
            return "Not Enrolled", confidence

        return self.label_map[prediction], confidence