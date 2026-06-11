from sklearn.neural_network import (
    MLPClassifier
)
import numpy as np


class ANNFaceClassifier:

    def __init__(self):

        self.model = MLPClassifier(

            hidden_layer_sizes=(128, 64),

            activation="relu",

            solver="adam",

            learning_rate="adaptive",

            max_iter=1000,

            early_stopping=True,

            random_state=42
        )

    def train(
        self,
        X_train,
        y_train
    ):

        self.model.fit(
            X_train,
            y_train
        )

    def predict(
        self,
        X_test
    ):

        predictions = self.model.predict(
            X_test
        )

        return predictions

    def calculate_accuracy(
        self,
        X_test,
        y_test
    ):

        accuracy = self.model.score(
            X_test,
            y_test
        )

        return accuracy
    
    def predict_with_unknown(
            self,
            X_test,
            threshold=0.75
    ):
        probabilities = (
            self.model.predict_proba(
                X_test
            )
        )

        predictions = []

        for prob in probabilities:
            max_prob = np.max(prob)
            if max_prob < threshold:
                predictions.append(
                    "Not Enrolled"
                )

            else:
                predicted_class = (
                    np.argmax(prob)
                )

                predictions.append(
                    predicted_class
                )

        return predictions