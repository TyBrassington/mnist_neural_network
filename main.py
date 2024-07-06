import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

INPUT_SIZE = 784  # 28x28 images
HIDDEN_LAYER_1_SIZE = 312
HIDDEN_LAYER_2_SIZE = 55
OUTPUT_CLASSES = 10
IMAGE_SHAPE = (28, 28)
NUM_ITERATIONS = 50000
INITIAL_LEARNING_RATE = 0.1
DROPOUT_RATE = 0.25
DECAY_RATE = 0.0001

DATASET_IN_USE = "Handwritten Digits"
ENABLE_DROPOUT = False

# True - Trains Model | False - Loads existing parameters and tests model
train_model = False
resume_autosave_state = False # Set to True to resume from a saved state

labels = []
param_dir = ""
data = pd.DataFrame()
# Datasets, Paths, and Labels
if DATASET_IN_USE == "Handwritten Digits":
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    param_dir = "digit_params"
    data = pd.read_csv('data/train.csv')
elif DATASET_IN_USE == "Fashion":
    labels = ["T-shirt", "Pants", "Sweatshirt", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    param_dir = "fashion_params"
    data = pd.read_csv('data_fashion/fashion-mnist_train.csv')


generalized_accuracy_filepath = f"{param_dir}/generalized_accuracy.csv"

data = data.sample(frac=1).to_numpy()  # Shuffle and convert to numpy array
m, n = data.shape
print(f"Data: {m} x {n} matrix")

val_set_size = m // 5 # 20% of dataset
data_test = data[0:val_set_size].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.0

data_train = data[val_set_size:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape


# Initialize Parameters
def init_params():
    W1 = np.random.randn(HIDDEN_LAYER_1_SIZE, INPUT_SIZE) * 0.01
    b1 = np.zeros((HIDDEN_LAYER_1_SIZE, 1))
    W2 = np.random.randn(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE) * 0.01
    b2 = np.zeros((HIDDEN_LAYER_2_SIZE, 1))
    W3 = np.random.randn(OUTPUT_CLASSES, HIDDEN_LAYER_2_SIZE) * 0.01
    b3 = np.zeros((OUTPUT_CLASSES, 1))
    return W1, b1, W2, b2, W3, b3


# Activation Functions
def ReLU(Z): return np.maximum(0, Z)
def deriv_ReLU(Z): return Z > 0
def softmax(Z): return np.exp(Z) / np.sum(np.exp(Z), axis=0)
def one_hot_encode(Y, num_classes=OUTPUT_CLASSES):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


# Regularization and Decay
def apply_dropout(A, dropout_rate):
    dropout_mask = (np.random.rand(*A.shape) > dropout_rate).astype(int)
    return A * dropout_mask / (1 - dropout_rate)
def apply_alpha_decay(initial_lr, decay_rate, iteration):
    return initial_lr / (1 + decay_rate * iteration)


# Forward and Backward Propagation
def forward_prop(W1, b1, W2, b2, W3, b3, X, training=True):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    if training:
        A1 = apply_dropout(A1, DROPOUT_RATE)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    #if training:
        # A2 = apply_dropout(A2, DROPOUT_RATE)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def back_prop(Z1, A1, Z2, A2, W2, A3, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot_encode(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1

    W2 -= alpha * dW2
    b2 -= alpha * db2

    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3


# Prediction and Accuracy
def get_predicted_classes(A3): return np.argmax(A3, 0)
def get_accuracy(predictions, Y): return np.sum(predictions == Y) / Y.size


# Training
def gradient_descent(X, Y, iterations, initial_lr, decay_rate, patience, resume=False):
    if resume:
        W1, b1, W2, b2, W3, b3, start_iteration = load_params("autosave_state", autosave_state=True)
        iterations -= start_iteration
    else:
        W1, b1, W2, b2, W3, b3 = init_params()
        start_iteration = 0

    alpha = initial_lr
    best_accuracy = 0
    patience_counter = 0

    train_accuracies = []
    val_accuracies = []
    iterations_list = []

    for i in range(start_iteration, start_iteration + iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X, training=ENABLE_DROPOUT)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, W2, A3, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 50 == 0:
            alpha = apply_alpha_decay(initial_lr, decay_rate, i)
            predictions = get_predicted_classes(A3)
            accuracy = get_accuracy(predictions, Y)
            Z1_val, A1_val, Z2_val, A2_val, Z3_val, A3_val = forward_prop(W1, b1, W2, b2, W3, b3, X_test, training=False)
            val_predictions = get_predicted_classes(A3_val)
            val_accuracy = get_accuracy(val_predictions, Y_test)

            iterations_list.append(i)
            train_accuracies.append(accuracy)
            val_accuracies.append(val_accuracy)

            print(f"Iteration {i}: Train Accuracy {accuracy:.4f}, Validation Accuracy {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at iteration {i} with best validation accuracy {best_accuracy:.4f}")
                break

            # Autosave the state every 200 iterations
            if i % 200 == 0 and i != 0: save_params(W1, b1, W2, b2, W3, b3, "autosave_state", i, autosave_state=True)


    # Plot at the end
    plt.figure()
    plt.title("Convergence Curves of Training and Validation Accuracy")
    plt.plot(iterations_list, train_accuracies, label='Train Accuracy')
    plt.plot(iterations_list, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return W1, b1, W2, b2, W3, b3


# Predictions and Evaluation
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X, training=False)
    predictions = get_predicted_classes(A3)
    return predictions


def plot_image(image_vector, prediction, label):
    plt.figure()
    plt.title("Predicted and Real Class of Image")
    plt.imshow(image_vector.reshape(IMAGE_SHAPE) * 255, cmap='binary')
    plt.title(f"Prediction: {labels[prediction[0]]}, Label: {labels[label]}")
    plt.show()


def evaluate_single_prediction(index, X_train, Y_train, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index].reshape(INPUT_SIZE, 1)
    prediction = make_predictions(current_image, W1, b1, W2, b2, W3, b3)
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


# Model Saving and Loading
def save_params(W1, b1, W2, b2, W3, b3, filepath, current_iteration=0,autosave_state=False):
    # Convert each parameter to DataFrame and save as CSV
    pd.DataFrame(W1).to_csv(f"{filepath}/W1.csv", index=False)
    pd.DataFrame(b1).to_csv(f"{filepath}/b1.csv", index=False)
    pd.DataFrame(W2).to_csv(f"{filepath}/W2.csv", index=False)
    pd.DataFrame(b2).to_csv(f"{filepath}/b2.csv", index=False)
    pd.DataFrame(W3).to_csv(f"{filepath}/W3.csv", index=False)
    pd.DataFrame(b3).to_csv(f"{filepath}/b3.csv", index=False)
    if autosave_state:
        pd.DataFrame({'current_iteration': [current_iteration]}).to_csv(f"{filepath}/state.csv", index=False)
        print(f"Training state saved to {filepath}")
    else: print(f"Weights and biases saved to {filepath}")

def load_params(filepath, autosave_state=False):
    # Read each CSV file into a DataFrame and convert to numpy array
    W1 = pd.read_csv(f"{filepath}/W1.csv").values
    b1 = pd.read_csv(f"{filepath}/b1.csv").values
    W2 = pd.read_csv(f"{filepath}/W2.csv").values
    b2 = pd.read_csv(f"{filepath}/b2.csv").values
    W3 = pd.read_csv(f"{filepath}/W3.csv").values
    b3 = pd.read_csv(f"{filepath}/b3.csv").values
    if autosave_state:
        current_iteration = int(pd.read_csv(f"{filepath}/state.csv")['current_iteration'][0])
        print(f"Training state loaded from {filepath}")
    else:
        current_iteration = 0
        print(f"Weights and biases loaded from {filepath}")
    return W1, b1, W2, b2, W3, b3, current_iteration


def read_generalized_accuracy(filepath):
    try:
        df = pd.read_csv(filepath)
        return df.loc[0, 'generalized_accuracy']
    except FileNotFoundError:
        return 0.0


def save_generalized_accuracy(filepath, accuracy):
    df = pd.DataFrame({'generalized_accuracy': [accuracy]})
    df.to_csv(filepath, index=False)


# Main Execution
if train_model:
    patience = 50
    if resume_autosave_state:
        W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, NUM_ITERATIONS, INITIAL_LEARNING_RATE, DECAY_RATE,
                                                  patience, resume=True)
    else:
        W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, NUM_ITERATIONS, INITIAL_LEARNING_RATE, DECAY_RATE,
                                                  patience)

    test_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3)
    generalized_accuracy = get_accuracy(test_predictions, Y_test)
    print(f"\nGeneralized Accuracy after training: {generalized_accuracy:.4f}")
    previous_generalized_accuracy = read_generalized_accuracy(generalized_accuracy_filepath)

    if generalized_accuracy > previous_generalized_accuracy + 0.003:
        save_params(W1, b1, W2, b2, W3, b3, param_dir)
        save_generalized_accuracy(generalized_accuracy_filepath, generalized_accuracy)
        print(f"New parameters saved with generalized accuracy: {generalized_accuracy:.4f}")
    else:
        print(f"Parameters not saved. Improvement not sufficient: {generalized_accuracy - previous_generalized_accuracy:.4f}")
else:
    W1, b1, W2, b2, W3, b3, _ = load_params(param_dir)



# Test the model
test_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3)
generalized_accuracy = get_accuracy(test_predictions, Y_test)

print(f"\nGeneralized Accuracy: {generalized_accuracy:.4f}")

for i in range(5):
    evaluate_single_prediction(np.random.randint(0, val_set_size), X_test, Y_test, W1, b1, W2, b2, W3, b3)

plot_confusion_matrix(Y_test, test_predictions, labels)