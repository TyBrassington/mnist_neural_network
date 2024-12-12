import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from config import IMAGE_SHAPE, INPUT_SIZE, labels
from model_training import forward_prop
from activation_and_regularization import apply_alpha_decay


def get_predicted_classes(A3): return np.argmax(A3, 0)

def get_accuracy(predictions, Y): return np.sum(predictions == Y) / Y.size



def make_predictions(X, params):
    cache = forward_prop(params, X, training=False)
    predictions = get_predicted_classes(cache["A3"])
    return predictions

def evaluate_model(
    i, params, cache, X, Y, X_test, Y_test, initial_lr, decay_rate,
    train_accuracies, val_accuracies, iterations_list
):
    alpha = apply_alpha_decay(initial_lr, decay_rate, i)
    predictions = get_predicted_classes(cache["A3"])
    accuracy = get_accuracy(predictions, Y)

    val_cache = forward_prop(params, X_test, training=False)
    val_predictions = get_predicted_classes(val_cache["A3"])
    val_accuracy = get_accuracy(val_predictions, Y_test)

    iterations_list.append(i)
    train_accuracies.append(accuracy)
    val_accuracies.append(val_accuracy)

    return alpha, accuracy, val_accuracy

def plot_convergence_curve(iterations_list, train_accuracies, val_accuracies):
    plt.figure()
    plt.title("Convergence Curves of Training and Validation Accuracy")
    plt.plot(iterations_list, train_accuracies, label="Train Accuracy")
    plt.plot(iterations_list, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def plot_image(image_vector, prediction, label):
    plt.figure()
    plt.title("Predicted and Real Class of Image")
    plt.imshow(image_vector.reshape(IMAGE_SHAPE) * 255, cmap='binary')
    plt.title(f"Prediction: {labels[prediction[0]]}, Label: {labels[label]}")
    plt.show()


def evaluate_single_prediction(index, X_train, Y_train, params):
    current_image = X_train[:, index].reshape(INPUT_SIZE, 1)
    prediction = make_predictions(current_image, params)
    label_index = Y_train[index]
    print(f"Prediction: {prediction[0]}, Label: {label_index}")
    plot_image(current_image, prediction, label_index)


def compute_confusion_matrix(y_true, y_pred, num_classes):
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_labels):
    num_classes = len(class_labels)

    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes)

    # Plot the confusion matrix
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, linewidths=0.1, cmap="gray", # bone or gray
                linecolor="white", fmt='.0f', ax=ax)

    # Add labels and titles

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(num_classes) + 0.5, class_labels, rotation=90)
    plt.yticks(np.arange(num_classes) + 0.5, class_labels, rotation=0)
    plt.show()


def read_generalized_accuracy(filepath):
    try:
        df = pd.read_csv(filepath)
        return df.loc[0, 'generalized_accuracy']
    except FileNotFoundError:
        return 0.0


def save_generalized_accuracy(filepath, accuracy):
    df = pd.DataFrame({'generalized_accuracy': [accuracy]})
    df.to_csv(filepath, index=False)
