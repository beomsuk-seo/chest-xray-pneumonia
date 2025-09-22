import tensorflow as tf
import os

IMG_SIZE = (224, 224) #standard size, e.g. ResNet50
BATCH_SIZE = 32
SEED = 123

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw", "chest_xray")

def get_dataset(subset):
    """
    Create tf.data.Dataset from specified directory.

    Args:
        subset (str): "train", "test", or "val"
    
    Returns:
        tf.data.Dataset
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, subset),
        labels = "inferred", # based on folder structure
        label_mode = "categorical",
        image_size = IMG_SIZE, #resizing image (preprocessing)
        batch_size = BATCH_SIZE,
        shuffle = True if subset == "train" else False, #only shuffling training set
        seed = SEED
    )
    return dataset

# normalization layer
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

# augmentation layers (4)
# flip, rotate, zoom, contrast  
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

def preprocess(dataset, augment = False):
    """
    Preprocessing tf.data.Dataset
    - Normalization of pixel values
    - Data augmentation on training set
    """

    def normalize_and_augment(x, y):
        # normalize
        x = normalization_layer(x)

        #augment (if training set)
        if augment:
            x = data_augmentation(x)
        return x, y
    
    # apply normalization + augmentation
    dataset = dataset.map(normalize_and_augment, num_parallel_calls = tf.data.AUTOTUNE)
    
    # Pipeline performance optimization
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def load_datasets():
    """
    Returns preprocessed datasets (train, test, val).
    Note: data augmentation only performed on training set.
    """
    train_ds = preprocess(get_dataset("train"), augment = True)
    test_ds = preprocess(get_dataset("test"))
    val_ds = preprocess(get_dataset("val"))
    return train_ds, test_ds, val_ds