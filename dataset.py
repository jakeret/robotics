import tensorflow as tf
import numpy as np


SENSORS = ["pASP", "pBSP", "DC1", "DC2", "DC3", "DC4"]
OUTPUTS = ["alpha", "pA", "pB"]


def normalize(x, ):
    x = x.copy()
    x[:, 0] = np.log(1 + x[:, 0])
    x[:, 1] = np.log(1 + x[:, 1])
    x[:, 2] -= 0.5
    x[:, 3] -= 0.5
    x[:, 4] -= 0.5
    x[:, 5] -= 0.5
    return x


def load_data(path, sequence_lenght, offset, max_samples, sampling_rate, normalize_inputs):
    print(f"Loading {path}")
    raw_data = np.loadtxt(path)[:max_samples]

    inputs = raw_data[::sampling_rate, :len(SENSORS)]
    outputs = raw_data[::sampling_rate, len(SENSORS):]

    if normalize_inputs:
        inputs = normalize(inputs)

    n, m = inputs.shape
    num_sequences = n - sequence_lenght - offset + 1
    s0, s1 = inputs.strides

    sequences = np.lib.stride_tricks.as_strided(inputs,
                                                shape=(num_sequences, sequence_lenght, m),
                                                strides=(s0, s0, s1))

    targets = outputs[sequence_lenght + offset - 1:]

    assert len(sequences) == len(targets)

    return sequences, targets


def load_datasets(sequence_lenght, offset, train_data_path, test_data_path, max_samples, sampling_rate):
    x_train, y_train = load_data(train_data_path, sequence_lenght, offset, max_samples, sampling_rate)
    x_test, y_test = load_data(test_data_path, sequence_lenght, offset, max_samples, sampling_rate)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000)
    return train_dataset, validation_dataset
