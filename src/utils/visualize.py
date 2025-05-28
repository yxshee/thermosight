import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def plot_metrics(writer: SummaryWriter, loss: float, accuracy: float, epoch: int):
    writer.add_scalar('Training Loss', loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)


def plot_confusion_matrix(writer: SummaryWriter, cm: np.ndarray, epoch: int):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", va='center', ha='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    writer.add_figure('Confusion Matrix', fig, epoch)
