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

def preprocess(dataset):
    #TODO: DATA AUGMENTATION (only training set )
    #TODO: Caching / Prefetching

    """
    Preprocessing (normalization, augmentation, etc.)
    """
    # normalize
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return dataset

train_ds = preprocess(get_dataset("train"))
test_ds = preprocess(get_dataset("test"))
val_ds = preprocess(get_dataset("val"))

