import matplotlib.pyplot as plt
import numpy as np



def show_sample_faces(
        face_database,
        labels,
        label_map,
        image_size=(100, 100),
        samples=5
):
    
    random_indices = np.random.choice(
        len(labels),
        samples,
        replace=False
    )

    plt.figure(figsize=(15, 5))

    for i, idx in enumerate(random_indices):

        image = face_database[:, idx].reshape(
            image_size
        )

        label = label_map[labels[idx]]

        plt.subplot(1, samples, i + 1)

        plt.imshow(
            image,
            cmap="gray"
        )

        plt.title(label)
        plt.axis("off")

    plt.tight_layout()
    plt.show()