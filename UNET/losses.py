import tensorflow as tf

def focal_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.8, gamma: float = 2.0) -> tf.Tensor:
    # Focal Loss
    focal_loss = alpha * tf.pow(1 - y_pred, gamma) * y_true * tf.math.log(y_pred + 1e-7)
    focal_loss += (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
    focal_loss = -tf.reduce_mean(focal_loss)
    
    # Dice Loss
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_loss = 1 - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
    
    return 0.5 * focal_loss + 0.5 * dice_loss