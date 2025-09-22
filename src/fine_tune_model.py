# fine tuning model (best_model.h5) generated in train_model.py

import tensorflow as tf
import os

# unfreeze base ResNet50 model
# . 

MODEL_PATH = os.join("..", "best_model.h5")

model = tf.keras.load_model(MODEL_PATH)
print("Model loaded successfully.")