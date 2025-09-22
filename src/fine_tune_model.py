# fine tuning model (best_model.h5) generated in train_model.py
from preprocessing_pipeline import load_datasets
import tensorflow as tf
import os

# unfreeze base ResNet50 model
# . 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "best_model.h5")

# load datasets (preprocessing_pipeline.py) 
#(224, 224, 3)
train_ds, test_ds, val_ds = load_datasets()
model = tf.keras.models.load_model("best_model.h5")
print("Model loaded successfully.")

# check frozen parts of base model. 177 layers.
# for i, layer in enumerate(model.layers):
#     print(i, layer.name, layer.trainable)

# unfreeze top 20 layers 