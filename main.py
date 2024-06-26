import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


IMAGE_SIZE = 784  # 28x28 images
HIDDEN_LAYER_1_SIZE = 10 # Adjust these through empirical testing
HIDDEN_LAYER_2_SIZE = 532 # Adjust these through empirical testing
NUM_CLASSES = 10

data = pd.read_csv('data/train.csv')
data = data.sample(frac=1).to_numpy()  # Shuffle and convert to numpy array
m, n = data.shape

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_,m_train = X_train.shape

def init_params():
    W1 = np.random.randn(HIDDEN_LAYER_1_SIZE, IMAGE_SIZE) * 0.01
    b1 = np.zeros((HIDDEN_LAYER_1_SIZE, 1))
    W2 = np.random.randn(NUM_CLASSES, HIDDEN_LAYER_1_SIZE) * 0.01
    b2 = np.zeros((NUM_CLASSES, 1))
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def deriv_ReLU(Z):
    return Z > 0


def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


def one_hot(Y, num_classes = NUM_CLASSES):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0: # Update progress every 10th iteration
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy {accuracy:.4f}")
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index].reshape(IMAGE_SIZE, 1)
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print(f"Prediction: {prediction[0]}, Label: {label}")
    plt.imshow(current_image.reshape(28, 28) * 255, cmap='gray')
    plt.title(f"Prediction: {prediction[0]}, Label: {label}")
    plt.show()


# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

# Test the model
#test_prediction(0, W1, b1, W2, b2)


test_predictions = make_predictions(X_test, W1, b1, W2, b2)
get_accuracy(test_predictions, Y_test)
