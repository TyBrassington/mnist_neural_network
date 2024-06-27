import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

INPUT_SIZE = 784  # 28x28 images
HIDDEN_LAYER_1_SIZE = 55
HIDDEN_LAYER_2_SIZE = 35
OUTPUT_CLASSES = 10
IMAGE_SHAPE = (28, 28)
NUM_ITERATIONS = 10000
INITIAL_LEARNING_RATE = 0.1
DROPOUT_RATE = 0.25
DECAY_RATE = 0.00045

data = pd.read_csv('data/train.csv')
data = data.sample(frac=1).to_numpy()  # Shuffle and convert to numpy array
m, n = data.shape
print(f"Data: {m} x {n} matrix")

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape

def init_params():
    W1 = np.random.randn(HIDDEN_LAYER_1_SIZE, INPUT_SIZE) * 0.01
    b1 = np.zeros((HIDDEN_LAYER_1_SIZE, 1))
    W2 = np.random.randn(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE) * 0.01
    b2 = np.zeros((HIDDEN_LAYER_2_SIZE, 1))
    W3 = np.random.randn(OUTPUT_CLASSES, HIDDEN_LAYER_2_SIZE) * 0.01
    b3 = np.zeros((OUTPUT_CLASSES, 1))
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def one_hot_encode(Y, num_classes=OUTPUT_CLASSES):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def apply_dropout(A, dropout_rate):
    dropout_mask = (np.random.rand(*A.shape) > dropout_rate).astype(int)
    return A * dropout_mask / (1 - dropout_rate)

def apply_alpha_decay(initial_lr, decay_rate, iteration):
    new_alpha = initial_lr / (1 + decay_rate * iteration)
    return new_alpha

def forward_prop(W1, b1, W2, b2, W3, b3, X, training=True):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    if training:
        A1 = apply_dropout(A1, DROPOUT_RATE)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    if training:
        A2 = apply_dropout(A2, DROPOUT_RATE)
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

def get_predicted_classes(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, initial_lr, decay_rate, patience):
    W1, b1, W2, b2, W3, b3 = init_params()
    alpha = initial_lr
    best_accuracy = 0
    patience_counter = 0

    train_accuracies = []
    val_accuracies = []
    iterations_list = []

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X, training=False)
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

    # Plot at the end
    plt.plot(iterations_list, train_accuracies, label='Train Accuracy')
    plt.plot(iterations_list, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return W1, b1, W2, b2, W3, b3

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X, training=False)
    predictions = get_predicted_classes(A3)
    return predictions

def plot_image(image_vector, prediction, label):
    plt.imshow(image_vector.reshape(IMAGE_SHAPE) * 255, cmap='gray')
    plt.title(f"Prediction: {prediction[0]}, Label: {label}")
    plt.show()

def evaluate_single_prediction(index, X_train, Y_train, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index].reshape(INPUT_SIZE, 1)
    prediction = make_predictions(current_image, W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print(f"Prediction: {prediction[0]}, Label: {label}")
    plot_image(current_image, prediction, label)

# Train the model
patience = 20
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, NUM_ITERATIONS, INITIAL_LEARNING_RATE, DECAY_RATE, patience)

# Test the model
test_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3)
generalized_accuracy = get_accuracy(test_predictions, Y_test)

print(f"\nGeneralized Accuracy: {generalized_accuracy}")
