import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_accuracy(history, title='Model Accuracy'):
    """
    Plots the accuracy of the model for each epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(history, title='Model Loss'):
    """
    Plots the loss of the model for each epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

def visualize_predictions(x_test, y_test, y_pred, class_names):
    """
    Visualizes the predictions along with the ground truth for comparison.
    """
    plt.figure(figsize=(15, 10))
    for i in range(10):  # Just plotting 10 samples for brevity
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.plot(x_test[i])
        plt.xlabel(f"Actual: {class_names[y_test[i]]}\nPredicted: {class_names[y_pred[i]]}")
    plt.show()
