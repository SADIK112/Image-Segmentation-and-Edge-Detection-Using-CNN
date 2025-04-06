import tensorflow as tf
from config import INPUT_CHANNELS, IMAGE_SIZE, OUTPUT_CHANNELS, LEARNING_RATE, KERNEL_REGULARIZER, WEIGHT_DECAY, USE_RELU
from losses import focal_dice_loss
# from attention import channel_attention

def Unet(pretrained_weights = None, input_shape = (*IMAGE_SIZE, INPUT_CHANNELS)):
    inputs = tf.keras.layers.Input(shape = input_shape, name = 'input_tensor')
    resize_inputs = tf.keras.layers.Lambda(lambda x: x / 255, name = "normalize_input")(inputs)

    def conv_block(x: tf.Tensor, filters, dropout_rate = 0.1):
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation=None, 
                                   kernel_initializer="he_normal", padding="same", 
                                   kernel_regularizer=tf.keras.regularizers.l2(KERNEL_REGULARIZER))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) if USE_RELU else tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Try with attention machanism to see how the model does for better result
        # x = channel_attention(x)

        x = tf.keras.layers.Conv2D(filters, (3, 3), activation=None, 
                                   kernel_initializer="he_normal", padding="same", 
                                   kernel_regularizer=tf.keras.regularizers.l2(KERNEL_REGULARIZER))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) if USE_RELU else tf.keras.layers.LeakyReLU()(x)
        return x

    # (Encoder)
    d1 = conv_block(resize_inputs, 16, 0.1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(d1)
    d2 = conv_block(p1, 32, 0.1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(d2)
    d3 = conv_block(p2, 64, 0.2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(d3)
    d4 = conv_block(p3, 128, 0.2)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(d4)
    d5 = conv_block(p4, 256, 0.3)

    # (Decoder)
    def upconv_block(x: tf.Tensor, skip: tf.Tensor, filters, dropout_rate = 0.1):
        x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip])
        x = conv_block(x, filters, dropout_rate)
        return x

    d6 = upconv_block(d5, d4, 128, 0.2)
    d7 = upconv_block(d6, d3, 64, 0.2)
    d8 = upconv_block(d7, d2, 32, 0.1)
    d9 = upconv_block(d8, d1, 16, 0.1)
    
    outputs = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1, 1), activation = "sigmoid")(d9)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss=focal_dice_loss, metrics=["accuracy"])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model