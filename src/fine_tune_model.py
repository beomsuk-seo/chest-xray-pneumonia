# fine tuning model (best_model.h5) generated in train_model.py
# unfreezing 30 layers, smaller learning_rate (1e-5), no ReduceLROnPlateau

from preprocessing_pipeline import load_datasets
import tensorflow as tf
import os

# num of base model layers to unfreeze 
UNFREEZE_LAYERS = 30
LEARNING_RATE = 5e-6
EPOCHS = 15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "best_model.h5")

# load datasets (preprocessing_pipeline.py) 
#(224, 224, 3)
train_ds, test_ds, val_ds = load_datasets()
model = tf.keras.models.load_model(MODEL_DIR)
print("Model loaded successfully.")

# # to see layers
# for i, layer in enumerate(model.layers):
#     print(i, layer.name, type(layer))

# unfreeze layers
for layer in model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

# recompile model w/ lower learning rate
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
    loss = "sparse_categorical_crossentropy",
    metrics = ['accuracy'] 
)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 8,
        restore_best_weights = True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_fine_tuned.h5",
        save_best_only = True,
        monitor = 'val_loss'
    )
]
#retraining model
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = EPOCHS,
    callbacks = callbacks
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"test_loss: {test_loss:.4f}")
print(f"test_accuracy: {test_acc:.4f}")

# Epoch 13/15
# 163/163 [==============================] - 286s 2s/step - loss: 0.4171 - accuracy: 0.8112 - val_loss: 2.1256 - val_accuracy: 0.4375
# # Epoch 14/15
# # 163/163 [==============================] - 284s 2s/step - loss: 0.4216 - accuracy: 0.8081 - val_loss: 1.5729 - val_accuracy: 0.7500
# # Epoch 15/15
# # 163/163 [==============================] - 299s 2s/step - loss: 0.4300 - accuracy: 0.8004 - val_loss: 2.0547 - val_accuracy: 0.5625