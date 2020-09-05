import tensorflow as tf
import sys

from utils import GlobalSTDPooling1D


def lstm(input_size, network_config):
    inputs = tf.keras.Input(shape=(None, input_size))

    prev = inputs
    layers = network_config["layers"]
    units = [i[0] for i in layers]
    directions = [i[1] for i in layers]

    for i, (size, direction) in enumerate(zip(units, directions)):
        return_sequences = False if i >= len(units) - 1 else True
        if direction == 1:
            prev = tf.keras.layers.LSTM(size, return_sequences=return_sequences)(prev)
        elif direction == 2:
            prev = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                size, return_sequences=return_sequences))(prev)
        else:
            sys.exit("{} is not a valid LSTM direction".format(direction))

    return inputs, prev


def cnn(input_size, network_config):
    inputs = tf.keras.Input(shape=(None, input_size))
    layers = network_config['layers']
    prev = inputs

    for i in layers:
        prev = tf.keras.layers.Conv1D(i[0], i[1],
                                      activation=network_config["activation"])(prev)
    shared = tf.keras.layers.GlobalMaxPooling1D()(prev)
    return inputs, shared


def fully_connected(input_size, network_config):
    inputs = tf.keras.Input(shape=(input_size,), dtype=tf.dtypes.float32)
    outputs = build_dense_layers(inputs, network_config, name="shared")
    return inputs, outputs


def build_dense_layers(inputs, network_config, name):
    hidden_nodes = network_config["hidden_nodes"]
    prev = inputs

    for i, size in enumerate(hidden_nodes):
        layer = tf.keras.layers.Dense(
            size, activation=network_config['activation'], name="{}_{}".format(name, i))(prev)
        prev = tf.keras.layers.Dropout(network_config['dropout'])(layer)

    return prev
