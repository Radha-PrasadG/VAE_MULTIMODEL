import tensorflow as tf
import matplotlib.pyplot as plt
import os

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

DATASET_PATH = "../dataset/bottle/train/good"

# Load image file paths
image_paths = [
    os.path.join(DATASET_PATH, fname)
    for fname in os.listdir(DATASET_PATH)
    if fname.endswith(".png")
]

print(f"Total Images Found: {len(image_paths)}")


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image


dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_image)
dataset = dataset.batch(BATCH_SIZE)

for batch in dataset.take(1):

    print("Batch Shape:", batch.shape)

    plt.figure(figsize=(10,10))

    for i in range(9):

        plt.subplot(3,3,i+1)
        plt.imshow(batch[i])
        plt.axis("off")

    plt.show()