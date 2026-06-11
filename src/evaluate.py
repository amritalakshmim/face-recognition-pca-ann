import os
import matplotlib.pyplot as plt


def plot_accuracy_vs_k(
    k_values,
    accuracies
):
    """
    Plot model accuracy
    against different
    PCA k values.

    This function visualizes
    how the number of PCA
    components affects the
    face recognition accuracy.

    The graph helps identify
    the optimal k value
    for best performance.

    Parameters
    ----------
    k_values : list
        List of PCA
        component values.

    accuracies : list
        Accuracy percentages
        corresponding
        to each k value.
    """

    plt.figure(
        figsize=(8, 5)
    )

    # Plot accuracy curve
    plt.plot(
        k_values,
        accuracies,
        marker="o"
    )

    plt.xlabel(
        "K Value"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Accuracy vs K Value"
    )

    plt.grid(True)

    # Create output folder
    # if it does not exist
    os.makedirs(
        "outputs/graphs",
        exist_ok=True
    )

    # Save graph
    plt.savefig(
        "outputs/graphs/"
        "accuracy_vs_k.png"
    )

    plt.show()