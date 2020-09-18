import pickle
import tensorflow as tf

from sklearn import preprocessing
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from networks import lstm, cnn, fully_connected, NaiiveBayes, KNN, LR, SVM
from utils import prepare_data, evaluate_model


def train(hparams):
    input_size = hparams["input_size"]

    x, y = prepare_data(hparams, 'train')
    x_scaled = preprocessing.scale(x)
    #x_dev, y_dev = prepare_data(hparams, 'dev')
    #x_test, y_test = prepare_data(hparams, 'test')
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.15, random_state=42)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

    if hparams['training_type'] == 'DL':

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
            raise ValueError('Undefined {} network for DL'.format(hparams["network_type"]))

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fault_detector")
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(hparams["learning_rate"]),
                      loss=hparams["loss"])

        model.fit(x_train, y_train, verbose=1, shuffle=True, validation_data=(x_dev, y_dev),
                  epochs=hparams["epochs"], batch_size=hparams["batch_size"])


    elif hparams["training_type"] == 'ML':

        if hparams["network_type"] == 'NB':
            model = NaiiveBayes(x_train, y_train, hparams["network_config"]["NB"]["type"])
        elif hparams["network_type"] == 'KNN':
            model = KNN(x_train, y_train, hparams["network_config"]["KNN"]["k"])
        elif hparams["network_type"] == 'logistic_regression':
            model = LR(x_train, y_train, hparams["network_config"]["logistic_regression"]["c"])
        elif hparams["network_type"] == 'SVM':
            model = SVM(x_train, y_train, hparams["network_config"]["SVM"]["c"],
                                          hparams["network_config"]["SVM"]["kernel"])
        else:
            raise ValueError('Undefined {} network type for ML'.format(hparams["network_type"]))

    evaluate_model(model, x_test, y_test, hparams["training_type"])

    if bool(hparams['save_model']):
        if hparams['training_type']=='DL':
            model.save(hparams['model_path']+'.h5')
        else:
            with open(hparams['model_path']+'.pkl', 'wb') as file:
                pickle.dump(model, file)


