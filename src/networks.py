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


def tdnn(input_size, network_config):
    # Implementation of: Deep Neural Network Embeddings for Text-Independent Speaker Verification
    # Section 3.3

    inputs = tf.keras.Input(shape=(None, input_size))
    temp = tf.keras.layers.Conv1D(network_config["kernels"], 5,
                                  activation=network_config["activation"])(inputs)
    temp = tf.keras.layers.Conv1D(network_config["kernels"], 3,
                                  activation=network_config["activation"], dilation_rate=2)(temp)
    temp = tf.keras.layers.Conv1D(network_config["kernels"], 3,
                                  activation=network_config["activation"], dilation_rate=3)(temp)
    temp = tf.keras.layers.Conv1D(network_config["kernels"], 1,
                                  activation=network_config["activation"])(temp)
    temp = tf.keras.layers.Conv1D(network_config["embed_size"], 1,
                                  activation=network_config["activation"])(temp)

    mean = tf.keras.layers.GlobalAveragePooling1D()(temp)
    std = GlobalSTDPooling1D()(temp)
    pooled = tf.concat([mean, std], axis=1)

    layer = tf.keras.layers.Dense(network_config["hidden_nodes"],
                                  activation=network_config['activation'])(pooled)
    output = tf.keras.layers.Dense(network_config["hidden_nodes"],
                                   activation=network_config['activation'])(layer)
    return inputs, output


def cnn(input_size, network_config):
    inputs = tf.keras.Input(shape=(None, input_size))
    #filters = network_config['filters']
    #kernels = network_config['kernels']
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


def task_specific(network, network_config, tasks, selected_tasks, inputs):
    losses = []
    loss_weights = []
    outputs = []

    for task in tasks:
        if task["name"] in selected_tasks:
            output = build_dense_layers(
                network, network_config, name=task["name"]+"_specific")
            if network_config['concat_feat_vector']:
                output = tf.concat([output, inputs], axis=1)
            outputs.append(tf.keras.layers.Dense(
                len(task["classes"]), name=task["name"] + "_output",
                activation=task['activation'])(output))
            loss_weights.append(tf.keras.backend.variable(1))
            losses.append(task["loss"])

    return outputs, loss_weights, losses


def build_dense_layers(inputs, network_config, name):
    hidden_nodes = network_config["hidden_nodes"]
    prev = inputs

    for i, size in enumerate(hidden_nodes):
        layer = tf.keras.layers.Dense(
            size, activation=network_config['activation'], name="{}_{}".format(name, i))(prev)
        prev = tf.keras.layers.Dropout(network_config['dropout'])(layer)

    return prev
