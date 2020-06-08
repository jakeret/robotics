import argparse

import numpy as np
from sklearn.model_selection import ParameterGrid

import robotics

PARAM_GRID = dict(
    depth=[3, 4, 5],
    kernel_size=[3, 4, 5, 6],
    filters=[16, 32, 64, 96, 128, 256],
    dropout_rate=np.arange(0.0, 0.751, 0.25),
    normalize_inputs=[True, False]
)


def tune_model(data_path="", log_path="logs", output_path="model",
               learning_rate=0.001, batch_size=256, epochs=20):

    for i, hyperparams in enumerate(ParameterGrid(PARAM_GRID)):
        print(i, hyperparams)
        robotics.run_training(data_path, log_path, output_path,
                              learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
                              **hyperparams)


def _parse_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default="./")
    parser.add_argument('--output-path', type=str, default="model")
    parser.add_argument('--log-path', type=str, default="logs")

    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    tune_model(**_parse_cli_arguments())
