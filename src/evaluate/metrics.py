import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(labels, preds, num_classes):
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    return cm


def per_class_accuracy(cm):
    diag = np.diag(cm)
    return diag / cm.sum(axis=1)
