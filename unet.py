import tensorflow as tf
import numpy as np
from helper import f1_scores


def acti_fct(data):
    """LeakyReLu activation function design"""
    return tf.keras.layers.LeakyReLU(alpha=0.25)(data)


def conv_layers(data, n_filter, filter_size):
    """Two convolutions / batch normalizing and activation function at each layer"""
    conv1 = tf.keras.layers.Conv2D( n_filter,
                                    filter_size,
                                    padding='same',
                                    kernel_initializer='he_normal')(data)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    l1 = acti_fct(conv1)

    conv2 = tf.keras.layers.Conv2D( n_filter,
                                    filter_size,
                                    padding='same',
                                    kernel_initializer='he_normal')(l1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    l2 = acti_fct(conv2)
    return l2


def down_sample(data, n_filter, filter_size, dropout):
    """Contraction phase main function"""
    conv = conv_layers(data, n_filter, filter_size)
    pool = tf.keras.layers.MaxPool2D((2, 2))(conv)
    if dropout is not None:
        pool = tf.keras.layers.Dropout(dropout)(pool)
    return conv, pool


def up_sample(data, n_filter, filter_size, dropout):
    """Expansion phase main function"""
    conv = conv_layers(data, n_filter, filter_size)
    up_ = tf.keras.layers.Conv2DTranspose(n_filter,
                                         filter_size,
                                         (2, 2),
                                         padding='same',
                                         kernel_initializer='he_normal')(conv)
    if dropout is not None:
        up_ = tf.keras.layers.Dropout(dropout)(up_)
    return up_


def unet_arch(data, n_filter, filter_size, dropout):
    """U-Net architecture - cf https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png"""

    conv1, pool1 = down_sample(data, n_filter, filter_size, dropout)
    conv2, pool2 = down_sample(pool1, n_filter * 2, filter_size, dropout)
    conv3, pool3 = down_sample(pool2, n_filter * 4, filter_size, dropout)
    conv4, pool4 = down_sample(pool3, n_filter * 8, filter_size, dropout)
    conv5 = conv_layers(pool4, n_filter * 16, filter_size)

    up6 = tf.keras.layers.Conv2DTranspose(n_filter * 8,
                                          filter_size,
                                          (2, 2),
                                          padding='same',
                                          kernel_initializer='he_normal')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv4], axis=3)
    up7 = up_sample(up6, n_filter * 4, filter_size, dropout)
    up7 = tf.keras.layers.concatenate([up7, conv3], axis=3)
    up8 = up_sample(up7, n_filter * 2, filter_size, dropout)
    up8 = tf.keras.layers.concatenate([up8, conv2], axis=3)
    up9 = up_sample(up8, n_filter, filter_size, dropout)
    up9 = tf.keras.layers.concatenate([up9, conv1], axis=3)

    up9 = conv_layers(up9, n_filter, filter_size)

    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')(up9)

    print("np.shape(output)",np.shape(output))

    return output


def unet_model(img_size, n_channel, n_filter, filter_size, dropout=None):
    """Construct the U-Net model"""
    inputs = tf.keras.layers.Input((img_size, img_size, n_channel))
    outputs = unet_arch(inputs, n_filter, filter_size, dropout)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['Precision', 'Recall'])
    return model


def train_model(model, x_train, y_train, batch_size, n_epochs, valid_split=0.0):
    """Optimize the model and return the model and training F1-Score"""
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=valid_split)
    print('\nhistory dict:', hist.history)

    scores = f1_scores(hist)

    return model, scores
