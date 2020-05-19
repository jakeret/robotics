import argparse
from datetime import datetime
from pathlib import Path

import mlflow
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import plotting
import tcn
import dataset

OFFSET = 1
SEQUENCE_LENGHT = 150
MAX_SAMPLES = 1_000_000
SAMPLING_RATE = 5


def create_model(depth = 5, kernel_size = 4, filters = 64, dropout_rate = 0.2, **__):
    print(f"Input sequence lenght: {SEQUENCE_LENGHT}, "
          f"model receptive field: {tcn.receptive_field_size(kernel_size, depth)}")

    channels = len(dataset.SENSORS)
    block_filters = [filters] * depth
    num_classes = 3

    inputs = layers.Input(shape=(SEQUENCE_LENGHT, channels))
    x = tcn.TCN(block_filters,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes,
                           activation="linear",
                           name="output")(x)

    model = Model(inputs, outputs)

    return model


def train(model, train_dataset, test_dataset, log_path, output_path,
          learning_rate=0.001, batch_size=32, epochs=5, **__):

    print("Training samples: ", tf.data.experimental.cardinality(train_dataset).numpy())
    print("Test samples: ", tf.data.experimental.cardinality(test_dataset).numpy())

    model.compile(optimizers.Adam(learning_rate=learning_rate),
                  loss='mae',
                  metrics=[metrics.mean_absolute_error,
                           metrics.mean_squared_error,
                           metrics.RootMeanSquaredError()
                           ])
    model.summary()

    model.fit(train_dataset.batch(batch_size),
              validation_data=test_dataset.batch(batch_size),
              callbacks=build_callbacks(log_path, output_path),
              epochs=epochs)

    return model


def build_callbacks(log_path, output_path):
    return [
        ModelCheckpoint(output_path, save_best_only=True),
        TensorBoard(str(Path(log_path) / datetime.now().strftime("%Y-%m-%dT%H-%M_%S")))
    ]


def run_training(data_path, log_path, output_path, **hyperparams):
    train_dataset, test_dataset = dataset.load_datasets(SEQUENCE_LENGHT, OFFSET,
                                                        train_data_path=Path(data_path) / "train.txt",
                                                        test_data_path=Path(data_path) / "test.txt",
                                                        max_samples=MAX_SAMPLES,
                                                        sampling_rate=SAMPLING_RATE)

    with mlflow.start_run():
        mlflow.log_params(hyperparams)

        model = create_model(**hyperparams)

        train(model, train_dataset, test_dataset, log_path, output_path, **hyperparams)

        log_metrics(model, train_dataset, test_dataset, hyperparams)

        log_predictions(model, output_path, test_dataset, train_dataset)

        log_model(model, output_path)


def log_model(model, output_path):
    model_path = str(Path(output_path) / datetime.now().strftime("%Y-%m-%dT%H-%M_%S"))
    model.save(model_path)
    mlflow.log_artifact(model_path)


def log_predictions(model, output_path, test_dataset, train_dataset):
    plot_path = Path(output_path) / "predictions_train.png"
    plotting.plot_predictions(train_dataset.batch(250), model, plot_path)
    mlflow.log_artifact(plot_path)
    plot_path = Path(output_path) / "predictions_test.png"
    plotting.plot_predictions(test_dataset.batch(250), model, plot_path)
    mlflow.log_artifact(plot_path)


def log_metrics(model, train_dataset, test_dataset, hyperparams):
    mlflow.log_metrics(
        model.evaluate(train_dataset
                       .batch(hyperparams["batch_size"]),
                       return_dict=True))

    validation_metrics = model.evaluate(test_dataset
                                        .batch(hyperparams["batch_size"]),
                                        return_dict=True)
    mlflow.log_metrics({"val_"+key: metric for key, metric in validation_metrics.items()})


def _parse_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default="./")
    parser.add_argument('--output-path', type=str, default="model")
    parser.add_argument('--log-path', type=str, default="logs")

    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.2)

    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    run_training(**_parse_cli_arguments())
