from sklearn.neural_network import (
    MLPClassifier
)

import numpy as np
import joblib


class ANNFaceClassifier:
    """
    Artificial Neural Network (ANN)
    classifier for face recognition.

    This class uses a Multi-Layer
    Perceptron (MLP) classifier
    to classify face signatures
    generated using PCA.

    Features:
    --------
    - Backpropagation learning
    - Adaptive learning rate
    - Early stopping
    - Unknown face detection
    - Model saving/loading
    """

    def __init__(self):
        """
        Initialize ANN model.

        Model Architecture:
        ------------------
        Input Layer:
            PCA face signatures

        Hidden Layer 1:
            128 neurons

        Hidden Layer 2:
            64 neurons

        Output Layer:
            Person class label
        """

        self.model = (
            MLPClassifier(

                hidden_layer_sizes=(
                    128,
                    64
                ),

                activation="relu",

                solver="adam",

                learning_rate=
                "adaptive",

                max_iter=1000,

                early_stopping=True,

                random_state=42
            )
        )

    def train(
        self,
        X_train,
        y_train
    ):
        """
        Train ANN model.

        Parameters
        ----------
        X_train :
        numpy.ndarray
            Training features.

        y_train :
        numpy.ndarray
            Training labels.
        """

        self.model.fit(
            X_train,
            y_train
        )

    def predict(
        self,
        X_test
    ):
        """
        Predict face labels.

        Parameters
        ----------
        X_test :
        numpy.ndarray
            Test features.

        Returns
        -------
        numpy.ndarray
            Predicted labels.
        """

        predictions = (
            self.model.predict(
                X_test
            )
        )

        return predictions

    def calculate_accuracy(
        self,
        X_test,
        y_test
    ):
        """
        Calculate model accuracy.

        Parameters
        ----------
        X_test :
        numpy.ndarray

        y_test :
        numpy.ndarray

        Returns
        -------
        float
            Classification accuracy.
        """

        accuracy = (
            self.model.score(
                X_test,
                y_test
            )
        )

        return accuracy

    def predict_with_unknown(
        self,
        X_test,
        threshold=0.75
    ):
        """
        Predict face identity
        with unknown face
        detection.

        If prediction confidence
        is below threshold,
        classify as:

        'Not Enrolled'

        Parameters
        ----------
        X_test :
        numpy.ndarray
            Test features.

        threshold : float
            Confidence threshold.

        Returns
        -------
        list
            Predicted labels
            or 'Not Enrolled'.
        """

        probabilities = (
            self.model
            .predict_proba(
                X_test
            )
        )

        predictions = []

        for prob in probabilities:

            max_prob = np.max(
                prob
            )

            # Reject low confidence
            # predictions
            if (
                max_prob <
                threshold
            ):

                predictions.append(
                    "Not Enrolled"
                )

            else:

                predicted_class = (
                    np.argmax(
                        prob
                    )
                )

                predictions.append(
                    predicted_class
                )

        return predictions

    def save_model(
        self,
        model_path
    ):
        """
        Save trained ANN model.

        Parameters
        ----------
        model_path : str
            Path to save model.
        """

        joblib.dump(
            self.model,
            model_path
        )

    def load_model(
        self,
        model_path
    ):
        """
        Load trained ANN model.

        Parameters
        ----------
        model_path : str
            Saved model path.
        """

        self.model = (
            joblib.load(
                model_path
            )
        )