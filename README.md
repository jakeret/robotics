Sequence Modelling for Position Tracking of Soft Robotic Arm using Temporal Convolutional Networks
====

Setup
-----
Setup virtualenv and dependencies by running:

```
$ pipenv install --dev 
$ pipenv shell
```

Training
--------

A single model can be trained by using `robotics.py`, either in a scripts using

```
import robotics
robotics.run_training(data_path, log_path, output_path, **hyperparams)
``` 

or on the command line with

```
$ python robotics.py --help
usage: robotics.py [-h] [--data-path DATA_PATH] [--output-path OUTPUT_PATH]
                   [--log-path LOG_PATH] [--depth DEPTH]
                   [--kernel_size KERNEL_SIZE] [--filters FILTERS]
                   [--dropout_rate DROPOUT_RATE]
                   [--normalize_inputs NORMALIZE_INPUTS]
                   [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                   [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
  --output-path OUTPUT_PATH
  --log-path LOG_PATH
  --depth DEPTH
  --kernel_size KERNEL_SIZE
  --filters FILTERS
  --dropout_rate DROPOUT_RATE
  --normalize_inputs NORMALIZE_INPUTS
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --epochs EPOCHS
```

Parameter tuning
----------------

A hyperparameter grid search can be starter using `tuning.py`, either in a scripts using

```
import tuning
tuning.tune_model(data_path="", log_path="logs", output_path="model")
``` 

or on the command line with

```
python tuning.py --help
usage: tuning.py [-h] [--data-path DATA_PATH] [--output-path OUTPUT_PATH]
                 [--log-path LOG_PATH] [--learning_rate LEARNING_RATE]
                 [--batch_size BATCH_SIZE] [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
  --output-path OUTPUT_PATH
  --log-path LOG_PATH
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --epochs EPOCHS
```

Evaluation
----------

All training results are being tracked with mlflow and TensorBoard. To launch the UI's run

```
$ pipenv shell
$ mlflow ui
```

```
$ pipenv shell
$ tensorboard --logdir=./logs
```
