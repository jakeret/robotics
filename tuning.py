import argparse

import numpy as np
from mlflow import tracking
from sklearn.model_selection import ParameterGrid

import robotics


def tune_model(data_path="", log_path="logs", output_path="model", tracking_uri="mlruns",
               learning_rate=0.001, batch_size=256, epochs=20):
    param_grid = dict(
        depth=[3, 4, 5],
        kernel_size=[3, 4, 5, 6],
        filters=[16, 32, 64, 96, 128, 256],
        dropout_rate=np.arange(0.0, 0.751, 0.25),
        normalize_inputs=[True, False]
    )

    for i, hyperparams in enumerate(ParameterGrid(param_grid)):
        hyperparams = dict(learning_rate=learning_rate,
                           batch_size=batch_size,
                           epochs=epochs,
                           **hyperparams)

        if contained(hyperparams, list_executed_hyperparams(tracking_uri)):
            continue

        print(i, hyperparams)
        robotics.run_training(data_path, log_path, output_path,
                              **hyperparams)


def contained(hyperparams, executed_hp):
    hyperparams = {k: str(v) for k,v in hyperparams.items()}
    for hp in executed_hp:
        if hyperparams == hp:
            return True
    return False


def list_executed_hyperparams(tracking_uri="./output/mlruns"):
    client = tracking.MlflowClient(tracking_uri=tracking_uri)
    run_infos = client.list_run_infos(experiment_id="0")
    executed_hp = []
    for run_info in run_infos:
        run = client.get_run(run_info.run_id)
        if run.info.status == "FINISHED":
            executed_hp.append(run.data.params)
    return executed_hp


def _parse_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default="./")
    parser.add_argument('--output-path', type=str, default="model")
    parser.add_argument('--log-path', type=str, default="logs")
    parser.add_argument('--tracking-uri', type=str, default="output/mlruns")

    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    tune_model(**_parse_cli_arguments())
