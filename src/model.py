import copy

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

from networks import lstm, cnn, fully_connected
from utils import prepare_data, evaluate_model


def train(hparams):
    input_size = hparams["input_size"]

    if hparams["network_type"] == "lstm":
        inputs, outputs = lstm(hparams,
                               input_size,
                               hparams["network_config"]["lstm"])
    elif hparams["network_type"] == "cnn":
        inputs, outputs = cnn(hparams,
                              input_size,
                              hparams["network_config"]["cnn"])
    elif hparams["network_type"] == "fully_connected":
        inputs, outputs = fully_connected(hparams,
                                          input_size,
                                          hparams["network_config"]["fully_connected"])
    else:
        raise ValueError('Undefined {} network for {} '.format(
            hparams["network_type"], hparams["features_type"]))


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fault_detector")
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(hparams["learning_rate"]),
                  loss=hparams["loss"])

    x_train, y_train = prepare_data(hparams, 'train')
    x_dev, y_dev = prepare_data(hparams, 'dev')
    x_test, y_test = prepare_data(hparams, 'test')


    model.fit(x_train, y_train, verbose=1, shuffle=True, validation_data=(x_dev, y_dev),
              epochs=hparams["epochs"], batch_size=hparams["batch_size"])

    print("====================== Done training ==============================")

    evaluate_model(model, x_test, y_test)
