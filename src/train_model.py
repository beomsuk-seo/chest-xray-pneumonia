import tensorflow as tf
from preprocessing_pipeline import load_datasets

#params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
NUM_CLASSES = 3
EPOCHS = 10
LEARNING_RATE = 1e-4

# load preprocessed datasets (see preprocesing_pipeline.py)
train_ds, test_ds, val_ds = load_datasets()

#model: ResNet50
base_model = tf.keras.applications.ResNet50(
    input_shape = IMG_SIZE + (3,), # (224, 224, 3)
    include_top = False, # exclude original resnet classifier
    weights = 'imagenet' # transfer learning
)

#freeze base model initially
base_model.trainable = False