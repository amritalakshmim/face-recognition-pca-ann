import matplotlib.pyplot as plt


def plot_accuracy_vs_k(
    k_values,
    accuracies
):

    plt.figure(
        figsize=(8, 5)
    )

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

    plt.savefig(
        "outputs/graphs/accuracy_vs_k.png"
    )

    plt.show()