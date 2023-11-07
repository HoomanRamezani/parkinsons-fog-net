from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the predictions.
    """
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='binary'):
    """
    Calculate the precision of the predictions.
    """
    return precision_score(y_true, y_pred, average=average)

def calculate_recall(y_true, y_pred, average='binary'):
    """
    Calculate the recall of the predictions.
    """
    return recall_score(y_true, y_pred, average=average)

def calculate_f1_score(y_true, y_pred, average='binary'):
    """
    Calculate the F1 score of the predictions.
    """
    return f1_score(y_true, y_pred, average=average)

def calculate_all_metrics(y_true, y_pred, average='binary'):
    """
    Calculate all metrics.
    """
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred, average=average)
    recall = calculate_recall(y_true, y_pred, average=average)
    f1 = calculate_f1_score(y_true, y_pred, average=average)
    return accuracy, precision, recall, f1
