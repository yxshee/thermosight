import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

def compute_confusion_matrix(true_labels, pred_labels, num_classes):
    """
    Computes the confusion matrix.
    """
    return sk_confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))

def per_class_accuracy(cm):
    """
    Computes per-class accuracy from a confusion matrix.
    Returns a numpy array where each element is the accuracy for a class.
    Handles cases where a class might have zero true instances to avoid division by zero.
    """
    class_sum = cm.sum(axis=1)
    # Replace 0 with 1 in class_sum for classes with no true samples to avoid division by zero
    # Their accuracy will be 0 anyway if diagonal is 0, or NaN if diagonal is also 0 (which we then convert to 0).
    class_acc = np.diag(cm) / np.where(class_sum == 0, 1, class_sum) 
    class_acc = np.nan_to_num(class_acc) # Convert NaNs (0/0) to 0
    return class_acc
