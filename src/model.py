import copy

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

from networks import lstm, cnn, fully_connected
from utils import prepare_data
from callbacks import EvaluateCallback, LearningRateCallback


def train(hparams):
    input_size = hparams["input_size"]

    if hparams["network_type"] == "lstm":
        inputs, outputs = lstm(input_size, hparams["network_config"]["lstm"])
    elif hparams["network_type"] == "cnn":
        inputs, outputs = cnn(input_size, hparams["network_config"]["cnn"])
    elif hparams["network_type"] == "fully_connected":
        inputs, outputs = fully_connected(
            input_size, hparams["network_config"]["fully_connected"])
    else:
        raise ValueError('Undefined {} network for {} '.format(
            hparams["network_type"], hparams["features_type"]))


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fault_detector")
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=hparams["loss"])

    lr_callback = LearningRateCallback(hparams['learning_rate'])
    learn_rate_callback = LearningRateScheduler(lr_callback.calculate, verbose=1)

    x_train, y_train = prepare_data(hparams['train'])
    x_dev, y_dev = prepare_data(hparams['dev'])
    x_test, y_test = prepare_data(hparams['test'])


    model.fit(x_train, y_train, verbose=1, shuffle=True,
              epochs=hparams["epochs"], batch_size=hparams["hparams"],
              callbacks=learn_rate_callback)
