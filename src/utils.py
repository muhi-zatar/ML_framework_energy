# Utils file
from datetime import datetime
import os
import pandas as pd
from collections import defaultdict

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import tensorflow as tf

def evaluate_model(model, x_test, y_test, train_type):
    answers = model.predict(x_test)
    if train_type == 'DL':
        answers = np.argmax(answers, axis=1)
#    import pdb;pdb.set_trace()
    print(accuracy_score(y_test, answers))
    print(classification_report(y_test, answers))


def build_parameter_combinations(params):
    base = [{}]
    for param, values in params.items():
        frontier = []
        for b in base:
            for value in values:
                c = {k: v for k, v in b.items()}
                c[param] = value
                frontier.append(c)
        base = frontier

    return frontier


def prepare_data(params, data_type):
    input_data = pd.read_excel(params[data_type], 'inputs')
    output_data = pd.read_excel(params[data_type], 'outputs')

    inputs = input_data.to_numpy()
    outputs = output_data.to_numpy()

    return inputs, outputs

def preprocess_images(params, data_type, subs = None, val_split = None):
    ds = tf.keras.preprocessing.image_dataset_from_directory(params[data_type],
        labels = 'inferred',
        validation_split=val_split,
        subset= subs,
        label_mode = "int", 
        batch_size = params['batch_size'], 
        image_size=tuple(params['network_config']['cnn']['input_shape'])[0:2],
        seed = 123
    )
    return ds

