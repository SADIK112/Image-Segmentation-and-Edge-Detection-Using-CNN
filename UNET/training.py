import tensorflow as tf
from UNET.model import Unet
from UNET.data_processing import get_train_dataset, get_val_dataset
from UNET.config import BATCH_SIZE, EPOCHS, TRAIN_IMG_NUM, VALIDATION_IMG_NUM, PRETRAINED_WEIGHTS, PATIENCE

def Unet_train(load_model = None):
    """Train the U-Net model."""
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()

    model = Unet(load_model)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        PRETRAINED_WEIGHTS, monitor = 'loss', verbose = 1, save_best_only = True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(patience = PATIENCE, restore_best_weights = True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor = 0.1, patience = 5)

    model.fit(
        train_dataset,
        validation_data = val_dataset,
        steps_per_epoch = TRAIN_IMG_NUM // BATCH_SIZE,
        validation_steps = VALIDATION_IMG_NUM // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    return model