from dataclasses import dataclass

from tensorflow.keras import layers
from tensorflow.python.keras import Model

import dataset
import tcn


@dataclass
class Hyperparams:
    learning_rate:float = 0.001
    batch_size:int = 32
    epochs:int = 5

    normalize_inputs:bool = False


@dataclass
class TCNHyperparams(Hyperparams):
    depth:int = 5
    kernel_size:int = 4
    filters:int = 64
    dropout_rate:float = 0.2


@dataclass
class RNNHyperparams(Hyperparams):
    units:int = 128
    dropout_rate:float = 0.2


def build_rnn_model(sequence_lenght, hyperparams: RNNHyperparams) -> Model:
    channels = len(dataset.SENSORS)
    num_classes = 3

    inputs = layers.Input(shape=(sequence_lenght, channels))

    x = layers.GRU(hyperparams.units)(inputs)
    x = layers.Dropout(hyperparams.dropout_rate)(x)
    outputs = layers.Dense(num_classes,
                           activation="linear",
                           name="output")(x)

    model = Model(inputs, outputs)
    return model


def build_tcn_model(sequence_lenght: int, hyperparams: TCNHyperparams) -> Model:
    print(f"Input sequence lenght: {sequence_lenght}, "
          f"model receptive field: {tcn.receptive_field_size(hyperparams.kernel_size, hyperparams.depth)}")

    channels = len(dataset.SENSORS)
    block_filters = [hyperparams.filters] * hyperparams.depth
    num_classes = 3

    inputs = layers.Input(shape=(sequence_lenght, channels))
    x = tcn.TCN(block_filters,
                kernel_size=hyperparams.kernel_size,
                dropout_rate=hyperparams.dropout_rate)(inputs)
    x = layers.Dropout(hyperparams.dropout_rate)(x)
    outputs = layers.Dense(num_classes,
                           activation="linear",
                           name="output")(x)

    model = Model(inputs, outputs)

    return model


def build_model(sequence_lenght:int, hyperparams: Hyperparams) -> Model:
    if isinstance(hyperparams, TCNHyperparams):
        return build_tcn_model(sequence_lenght, hyperparams)
    elif isinstance(hyperparams, RNNHyperparams):
        return build_rnn_model(sequence_lenght, hyperparams)

    raise Exception()
