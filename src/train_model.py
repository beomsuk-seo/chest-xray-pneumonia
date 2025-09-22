import tensorflow as tf
from preprocessing_pipeline import load_datasets

# load preprocessed datasets (see preprocesing_pipeline.py)
train_ds, test_ds, val_ds = load_datasets()
