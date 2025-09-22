# fine tuning model (best_model.h5) generated in train_model.py
# unfreezing 20 layers, smaller learning_rate (1e-5), no ReduceLROnPlateau


from preprocessing_pipeline import load_datasets
import tensorflow as tf
import os

# num of base model layers to unfreeze 
UNFREEZE_LAYERS = 20

EPOCHS = 10

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

# unfreeze top 20 layers
for layer in model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

# recompile model w/ lower learning rate
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
    loss = "sparse_categorical_crossentropy",
    metrics = ['accuracy']
)
# same callbacks minus ReduceLROnPlateau (already small learning rate)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience = 3,
        restore_best_weights = True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_fine_tuned.h5",
        save_best_only = True
    )
]
#retraining model
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = EPOCHS,
    callbacks = callbacks
)