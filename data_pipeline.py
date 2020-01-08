import os
import tensorflow as tf
import numpy as np
from params import patches_root, train_db_path, test_db_path

def decode(filename):
    contents = tf.io.read_file(filename)
    img = tf.io.decode_png(contents)
    # Convert from range [0, 255] to [0, 1]
    img = tf.cast(img, tf.float32) / 255.
    return img

def augment(img):
    # Randomly transform image to obtain more training data
    img = tf.image.random_flip_left_right(img)
    return img

def set_up_data_pipeline(patches_root, db_path):
    db = np.load(db_path, allow_pickle=True).item()
    files = db['path'].tolist()
    imgs = []
    for file in files:
        imgs += [os.path.join(patches_root, file)]
    labels = db['labels']

    images_dataset = tf.data.Dataset.from_tensor_slices(imgs)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

    # Decode images
    images_dataset = images_dataset.map(decode)

    # Randomly decide to flip image
    images_dataset = images_dataset.map(augment)

    # Zip images and labels
    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))

    # Shuffle
    dataset = dataset.shuffle(buffer_size=len(files))
    
    return dataset