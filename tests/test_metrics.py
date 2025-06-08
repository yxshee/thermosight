import numpy as np
from src.evaluate.metrics import compute_confusion_matrix, per_class_accuracy


def test_compute_confusion_matrix_basic():
    true_labels = [0, 1, 2, 1, 0, 2]
    pred_labels = [0, 2, 1, 1, 0, 2]
    cm = compute_confusion_matrix(true_labels, pred_labels, num_classes=3)
    expected = np.array([
        [2, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ])
    assert np.array_equal(cm, expected)


def test_per_class_accuracy_basic():
    cm = np.array([
        [2, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ])
    acc = per_class_accuracy(cm)
    expected = np.array([1.0, 0.5, 0.5])
    assert np.allclose(acc, expected)


def test_per_class_accuracy_with_zero_true_class():
    true_labels = [0, 0, 1, 1, 1]
    pred_labels = [0, 0, 1, 0, 1]
    cm = compute_confusion_matrix(true_labels, pred_labels, num_classes=3)
    expected_cm = np.array([
        [2, 0, 0],
        [1, 2, 0],
        [0, 0, 0],
    ])
    assert np.array_equal(cm, expected_cm)

    acc = per_class_accuracy(cm)
    expected_acc = np.array([1.0, 2/3, 0.0])
    assert np.allclose(acc, expected_acc)
