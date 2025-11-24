"""
Evaluation metrics for classification tasks.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    return accuracy_score(y_true, y_pred)


def confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: list = None) -> dict:
    """
    Generate confusion matrix and classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary containing:
        - confusion_matrix: numpy array
        - accuracy: float
        - classification_report: dict
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # Get classification report as dict
    if class_names:
        report = classification_report(y_true, y_pred,
                                       target_names=class_names,
                                       output_dict=True,
                                       zero_division=0)
    else:
        report = classification_report(y_true, y_pred,
                                       output_dict=True,
                                       zero_division=0)

    return {
        'confusion_matrix': cm,
        'accuracy': acc,
        'classification_report': report
    }


def print_evaluation_report(results: dict, class_names: list = None):
    """
    Print formatted evaluation report.

    Args:
        results: Output from confusion_matrix_report
        class_names: Optional list of class names
    """
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']

    # Print header
    if class_names:
        header = "    " + " ".join([f"{name:>6}" for name in class_names])
        print(header)
        for i, row in enumerate(cm):
            row_str = " ".join([f"{val:>6}" for val in row])
            print(f"{class_names[i]:>3} {row_str}")
    else:
        print(cm)

    print("\nPer-class Performance:")
    report = results['classification_report']
    for key, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"  {key}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")

    print("=" * 50)
