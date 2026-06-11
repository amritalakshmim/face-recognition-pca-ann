from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler
)


def split_dataset(
    face_signatures,
    labels,
    test_size=0.4,
    random_state=42
):
    """
    Split dataset into
    training and testing sets.

    The dataset is split using:

    60% → Training
    40% → Testing

    Stratified splitting is used
    to maintain equal class
    distribution.

    Feature scaling is also
    applied using StandardScaler
    for better ANN performance.

    Parameters
    ----------
    face_signatures :
    numpy.ndarray
        PCA-generated face
        signatures.

    labels :
    numpy.ndarray
        Face class labels.

    test_size : float, optional
        Testing split ratio.
        Default is 0.4.

    random_state : int, optional
        Random seed for
        reproducibility.

    Returns
    -------
    tuple
        X_train :
        Training features

        X_test :
        Testing features

        y_train :
        Training labels

        y_test :
        Testing labels

        scaler :
        Trained scaler
    """

    # Convert face signatures
    # into row-wise samples
    X = face_signatures.T

    y = labels

    # Split dataset into
    # train and test sets
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y,

        test_size=test_size,

        random_state=
        random_state,

        stratify=y
    )

    # Standardize features
    scaler = StandardScaler()

    X_train = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test = (
        scaler.transform(
            X_test
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )