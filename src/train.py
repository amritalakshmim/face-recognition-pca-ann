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
    
    X = face_signatures.T

    y = labels

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )