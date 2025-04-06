import tensorflow as tf
import matplotlib.pyplot as plt
from unet_baseline import Unet as UnetBaseline
from unet_improved import Unet as UnetImproved
from data_processing import get_train_dataset, get_val_dataset
from config import (
    BATCH_SIZE, 
    EPOCHS, 
    TRAIN_IMG_NUM, 
    VALIDATION_IMG_NUM, 
    PRETRAINED_WEIGHTS, 
    EARLY_STOPPING_PATIENCE, 
    LR_REDUCTION_PATIENCE, 
    LR_REDUCTION_FACTOR,
    USE_IMPROVED_MODEL
)

def Unet_train(load_model=None):
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()

    if USE_IMPROVED_MODEL:
        model = UnetImproved(load_model)
    else:
        model = UnetBaseline(load_model)
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        PRETRAINED_WEIGHTS,
        monitor='loss',
        verbose=1,
        save_best_only=True
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_REDUCTION_FACTOR,
        patience=LR_REDUCTION_PATIENCE,
        min_lr=1e-6
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=TRAIN_IMG_NUM // BATCH_SIZE,
        validation_steps=VALIDATION_IMG_NUM // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )
    
    plot_loss_history(history)
    plot_accuracy_history(history)
    
    return model

def plot_loss_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_history(history):
    """Plot training and validation accuracy."""
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()