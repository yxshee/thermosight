import matplotlib.pyplot as plt
import seaborn as sns
import io
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

def plot_metrics(writer, train_loss, test_acc, epoch):
    """
    Logs training loss and test accuracy to TensorBoard.
    """
    writer.add_scalar('Loss/train_epoch', train_loss, epoch)
    writer.add_scalar('Accuracy/test_epoch', test_acc, epoch)
    # Example of adding combined metrics if desired
    # writer.add_scalars('Performance', {'train_loss': train_loss, 'test_accuracy': test_acc}, epoch)

def plot_confusion_matrix(writer, cm, epoch, class_names=None):
    """
    Plots a confusion matrix and logs it to TensorBoard as an image.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    # Convert PIL Image to tensor
    image_tensor = transforms.ToTensor()(image)
    writer.add_image(f'ConfusionMatrix/Epoch_{epoch+1}', image_tensor, epoch)
    plt.close(fig)
    buf.close()
