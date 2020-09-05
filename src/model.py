import copy

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

from data_generator import DataGenerator
from networks import lstm, attention, cnn, fully_connected, task_specific, tdnn
from utils import log_results
from callbacks import DataGeneratorCallback, EvaluateCallback, LearningRateCallback


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

    dev_data_generator = DataGenerator(hparams, loss_weights, data_type='dev')
    dev_eval_callback = EvaluateCallback(dev_data_generator, "dev", hparams['dev_eval_period'], hparams)

    train_data_generator = DataGenerator(train_params, loss_weights)
    train_eval_callback = EvaluateCallback(
        train_data_generator, "train", train_params['train_eval_period'], hparams)

    model.fit(train_data_generator, verbose=1, shuffle=True,
              epochs=hparams["epochs"], batch_size=hparams["hparams"],
              callbacks=[learn_rate_callback, train_eval_callback, dev_eval_callback])

    test_data_generator = DataGenerator(hparams, loss_weights, data_type='test')
    log_results(dev_eval_callback.best_models[k]["model"], hparams,
                epochs=dev_eval_callback.best_models[k]["epoch"],
                data_generator=test_data_generator, mean_type=mean_type)
