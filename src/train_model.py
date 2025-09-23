import tensorflow as tf
from preprocessing_pipeline import load_datasets

#params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
NUM_CLASSES = 3
EPOCHS = 25 # 10 -> 25: more epochs for initial training
LEARNING_RATE = 1e-3 #1e-4 -> 1e-3: higher LR for initial training

# load preprocessed datasets (see preprocesing_pipeline.py)
train_ds, test_ds, val_ds = load_datasets()

#base model: ResNet50
base_model = tf.keras.applications.ResNet50(
    input_shape = IMG_SIZE + (3,), # (224, 224, 3)
    include_top = False, # exclude original resnet classifier
    weights = 'imagenet' # transfer learning
)
#freeze base model initially
base_model.trainable = False

#build classifier head
global_avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
# prevent overfitting, drop 30% at random
dropout = tf.keras.layers.Dropout(0.3)(global_avg)

output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dropout)

#connect models
model = tf.keras.Model(inputs = base_model.input, outputs = output)
#print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
    loss = "sparse_categorical_crossentropy",
    metrics=[
        "accuracy"#, extra metrics causing errors, will implement with classification_report
        #tf.keras.metrics.Precision(name = 'precision'),
        #tf.keras.metrics.Recall(name = 'recall'),
    ] # tracking metrics (optional: add F1 Score)
)

# callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 8, # 3 -> 8
        restore_best_weights = True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        save_best_only = True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor = 0.5, # 0.2 -> 0.5, less aggressive reduction
        patience = 4
    )
]

# train model
history = model.fit(
    train_ds, #training data; tf.data.Dataset
    validation_data = val_ds,
    epochs = EPOCHS,
    callbacks = callbacks
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"test_loss: {test_loss:.4f}")

#initial base modeL:
# 10 epochs
# test_loss: 1.0639
# accuracy: 0.4877

#new base model: more patience, epochs, higher LR
# 10 epochs -> 25 epochs
# LR reduction occurred on epoch #16
# final learning rate: 2.5000e-04
# test_loss: 0.8989
# best val_loss: 0.9026
# best val_accuracy: 0.6250
# major improvement: 48.7% -> 62.5% validation accuracy (+13.8%)