from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import traceback

import numpy as np
from tensorflow_core.python.keras.layers.core import RepeatVector, Dense
from tensorflow_core.python.keras.layers.normalization import LayerNormalization

import tensorflow as tf
import tensorflow_core as tfc

from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.layers.wrappers import TimeDistributed
from tensorflow_core.python.keras.models import Sequential, load_model
from tensorflow_core.python.client import device_lib

MODEL_PATH = 'resources/data/generated_model'
TRAINING_SET_PATH = './resources/model/wide-2.npy'
BATCH_SIZE = 4096
EPOCHS =  50


def get_training_set():
    model_vector = np.load(TRAINING_SET_PATH)
    model_vector = model_vector[:, :, :1].reshape(model_vector.shape[0],model_vector.shape[1])
    model_vector = np.expand_dims(model_vector,axis=1)
    return model_vector

def get_model(reload_model=True):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    if not reload_model:
        return load_model(MODEL_PATH, custom_objects={'LayerNormalization': LayerNormalization})

    training_set = get_training_set()

    timesteps = training_set.shape[1]
    n_features = training_set.shape[2]

    model = Sequential()

    # Encoder
    model.add(LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    model.add(LSTM(32, activation='relu',return_sequences=False))

    # Bottleneck
    model.add(RepeatVector(timesteps))

    # Decoder
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))

    model.add(TimeDistributed(Dense(n_features)))

    return model


try:
    print(device_lib.list_local_devices())
    print(tfc.test.is_gpu_available())

    training_set = get_training_set()

    model = get_model()
    model.summary()
    model.compile(optimizer="adam", loss="mae")
    model.fit(training_set, training_set, epochs=EPOCHS, verbose=2, batch_size=32).history
    print(model.predict(np.expand_dims(training_set[0],axis=1)))
except Exception as e:
    print(e)
    traceback.print_exc(file=sys.stdout)
    sys.exit(-1)
