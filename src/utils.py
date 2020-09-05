from datetime import datetime
import csv
import os
from collections import defaultdict

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import tensorflow as tf


def get_feature_input_size(features_type, input_size):
    if features_type in ["mfcc", "mel"]:
        return input_size[features_type]

    vector_types = features_type.replace("vector", "")
    size = sum([input_size[f+"vector"] for f in ["i", "d", "x"] if f in vector_types])
    return size


def evaluate_model(model, data_generator, hparams, mode=None):
    results = {}
    batch_size = data_generator.batch_size
    for i, task in enumerate(data_generator.all_tasks):
        preds = []
        labels = [x[1] for x in data_generator.datasets[task]]
        corpuses = [x[2] for x in data_generator.datasets[task]]
        for batch_start in range(0, len(labels), batch_size):
            batch = data_generator.datasets[task][batch_start:batch_start+batch_size]
            inputs = data_generator.get_inputs(batch)
            if len(data_generator.all_tasks) == 1:
                proba = model.predict(inputs)
            else:
                proba = model.predict(inputs)[i]
            preds += [x for x in np.argmax(proba, axis=1)]
        acc = accuracy_score(labels, preds)*100

        f1 = f1_score(labels, preds, average="macro")*100
        results[task+"_acc"] = acc
        results[task+"_f1"] = f1


        # Compute scores for each corpus
        c_labels = defaultdict(list)
        c_preds = defaultdict(list)
        for c, l, p in zip(corpuses, labels, preds):
            c_labels[c].append(l)
            c_preds[c].append(p)

        for c in c_labels.keys():
            acc = accuracy_score(c_labels[c], c_preds[c])*100
            f1 = f1_score(c_labels[c], c_preds[c], average="macro")*100
            results[c+"_"+task+"_f1"] = f1
            results[c+"_"+task+"_acc"] = acc

    return results


def get_csv_header(hparams):
    measures = ["f1", "acc"]
    tasks = [t["name"] for t in hparams["tasks"]]
    corpus = sorted(hparams["available_datasets"])

    general = ["features", "network_type", "selected_tasks", "mean_type", "epochs", "date", "time"]
    task_acc = [t+"_"+m for t in tasks for m in measures]
    corpus_acc = [c+"_"+t+"_"+m for c in corpus for t in tasks for m in measures]
    config = ["network_config", "config"]

    return general + task_acc + corpus_acc + config


def log_results(model, hparams, epochs=None, model_data=None, data_generator=None, mean_type="sum"):
    headers = get_csv_header(hparams)
    if data_generator:
        model_data = evaluate_model(model, data_generator, hparams, 'test')
    now = datetime.now()
    model_data["mean_type"] = mean_type
    model_data["date"] = now.strftime("%d/%m/%Y")
    model_data["time"] = now.strftime("%H:%M:%S")
    model_data["features"] = hparams["features_type"]
    model_data["selected_tasks"] = hparams["selected_tasks"]
    if hparams["features_type"] in ["mfcc", "mel"]:
        model_data["network_type"] = hparams["network_type"]
        model_data["network_config"] = hparams["network_config"][hparams["network_type"]]
    else:
        model_data["network_type"] = "fully_connected"
        model_data["network_config"] = hparams["network_config"]["fully_connected"]

    model_data['epochs'] = epochs
    if "epochs" not in model_data:
        model_data["epochs"] = hparams["epochs"]
    model_data['config'] = hparams
    if not os.path.isfile(hparams["csv_file"]):
        with open(hparams["csv_file"], 'w+') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(headers)

    with open(hparams["csv_file"], 'a') as myfile:
        values = [model_data[k] if k in model_data else 0 for k in headers]
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(values)


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


class GlobalSTDPooling1D(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(GlobalSTDPooling1D, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        temp = tf.math.reduce_variance(inputs, axis=self.axis)
        return tf.where(temp > 0, tf.sqrt(temp), 0)
