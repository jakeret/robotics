from pathlib import Path

from dataset import OUTPUTS
import matplotlib.pyplot as plt
from sklearn import metrics as skmetrics

def plot_predictions(dataset, model, plot_path: Path):
    x, y = tuple(iter(dataset))[0]
    y = y.numpy()
    predictions = model.predict(x)
    num_predictions = len(predictions)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    for i, output in enumerate(OUTPUTS):
        truths = y[:num_predictions, i]
        preds = predictions[:num_predictions, i]

        axs[i].plot(truths, linestyle="--", label="truth")
        axs[i].plot(preds, alpha=0.75, label="predictions")

        mse = skmetrics.mean_squared_error(truths, preds)
        r2 = skmetrics.r2_score(truths, preds)
        axs[i].set_title(f"{output} (MSE: {mse:.3f}, R2: {r2:.3f})")
        axs[i].legend(loc="best")

    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path)
    return fig
