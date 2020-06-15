import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import mlflow
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import dataset
import plotting
import modelling

OFFSET = 1
SEQUENCE_LENGHT = 150
MAX_SAMPLES = 1_000_00
SAMPLING_RATE = 5


def train(model,
          train_dataset,
          validation_dataset,
          log_path: Path,
          output_path: Path,
          hyperparams: modelling.Hyperparams):

    model.compile(optimizers.Adam(learning_rate=hyperparams.learning_rate),
                  loss='mae',
                  metrics=[metrics.mean_absolute_error,
                           metrics.mean_squared_error,
                           metrics.RootMeanSquaredError()
                           ])
    model.summary()

    model.fit(train_dataset.batch(hyperparams.batch_size),
              validation_data=validation_dataset.batch(hyperparams.batch_size),
              callbacks=build_callbacks(log_path, output_path),
              epochs=hyperparams.epochs)

    return model


def build_callbacks(log_path: Path, output_path:Path):
    return [
        ModelCheckpoint(str(output_path), save_best_only=True),
        TensorBoard(str(log_path))
    ]


def run_training(data_path, log_path, output_path, hyperparams: modelling.Hyperparams):
    train_dataset, validation_dataset = dataset.load_datasets(SEQUENCE_LENGHT, OFFSET,
                                                        train_data_path=Path(data_path) / "train.txt",
                                                        test_data_path=Path(data_path) / "test.txt",
                                                        max_samples=MAX_SAMPLES,
                                                        sampling_rate=SAMPLING_RATE,
                                                        normalize_inputs=hyperparams.normalize_inputs)

    start_time = datetime.now().strftime("%Y-%m-%dT%H-%M_%S")
    output_path = Path(output_path) / start_time
    log_path = Path(log_path) / start_time

    with mlflow.start_run():
        mlflow.set_tag("start_time", start_time)
        mlflow.set_tag("output_path", output_path)
        mlflow.set_tag("log_path", log_path)

        mlflow.log_params(asdict(hyperparams))

        model = modelling.build_model(SEQUENCE_LENGHT, hyperparams)

        train(model, train_dataset, validation_dataset, log_path, output_path, hyperparams)

        log_metrics(model, train_dataset, validation_dataset, hyperparams.batch_size)

        log_predictions(model, output_path, validation_dataset, train_dataset)

        log_model(model, output_path)


def log_model(model, output_path):
    model_path = output_path / "model"
    model.save(model_path)
    mlflow.log_artifact(model_path)


def log_predictions(model, output_path: Path, validation_dataset, train_dataset):
    plot_path = output_path / "predictions_train.png"
    plotting.plot_predictions(train_dataset.batch(250), model, plot_path)
    mlflow.log_artifact(plot_path)
    plot_path = output_path / "predictions_validation.png"
    plotting.plot_predictions(validation_dataset.batch(250), model, plot_path)
    mlflow.log_artifact(plot_path)


def log_metrics(model, train_dataset, validation_dataset, batch_size:int):
    mlflow.log_metrics(
        model.evaluate(train_dataset
                       .batch(batch_size),
                       return_dict=True))

    validation_metrics = model.evaluate(validation_dataset
                                        .batch(batch_size),
                                        return_dict=True)
    mlflow.log_metrics({"val_"+key: metric for key, metric in validation_metrics.items()})


def _parse_cli_arguments() -> Dict:
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default="./")
    parser.add_argument('--output-path', type=str, default="model")
    parser.add_argument('--log-path', type=str, default="logs")

    parser.add_argument('--depth', type=int, default=modelling.TCNHyperparams.depth)
    parser.add_argument('--kernel_size', type=int, default=modelling.TCNHyperparams.kernel_size)
    parser.add_argument('--filters', type=int, default=modelling.TCNHyperparams.filters)
    parser.add_argument('--dropout_rate', type=float, default=modelling.TCNHyperparams.dropout_rate)

    parser.add_argument('--normalize_inputs', type=bool, default=modelling.Hyperparams.normalize_inputs)

    parser.add_argument('--learning_rate', type=int, default=modelling.Hyperparams.learning_rate)
    parser.add_argument('--batch_size', type=int, default=modelling.Hyperparams.batch_size)
    parser.add_argument('--epochs', type=int, default=modelling.Hyperparams.epochs)

    return vars(parser.parse_args())


if __name__ == '__main__':
    arguments = _parse_cli_arguments()
    run_training(data_path=arguments.pop("data_path"),
                 output_path=arguments.pop("output_path"),
                 log_path=arguments.pop("log_path"),
                 # hyperparams=modelling.RNNHyperparams()
                 hyperparams=modelling.TCNHyperparams(**arguments)
                 )
