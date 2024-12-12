from activation_and_regularization import *
from config import *
from param_utils import *

def initialize_training(resume, iterations):
    if resume:
        params, start_iteration = load_params("autosave_state", autosave_state=True)
        iterations -= start_iteration
    else:
        params = init_params()
        start_iteration = 0
    return params, start_iteration

def perform_training_step(params, X, Y, alpha):
    cache = forward_prop(params, X, training=ENABLE_DROPOUT)
    grads = back_prop(params, cache, X, Y)
    params = update_params(params, grads, alpha)
    return params, cache

# Forward and Backward Propagation
def forward_prop(params, X, training=True):
    Z1 = params["W1"].dot(X) + params["b1"]
    A1 = ReLU(Z1)
    if training:
        A1 = apply_dropout(A1, DROPOUT_RATE)

    Z2 = params["W2"].dot(A1) + params["b2"]
    A2 = ReLU(Z2)
    # if training:
    #     A2 = apply_dropout(A2, DROPOUT_RATE)

    Z3 = params["W3"].dot(A2) + params["b3"]
    A3 = softmax(Z3)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3,
    }
    return cache


def back_prop(params, cache, X, Y):
    m = Y.size
    one_hot_Y = one_hot_encode(Y)

    dZ3 = cache["A3"] - one_hot_Y
    dW3 = 1 / m * dZ3.dot(cache["A2"].T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = params["W3"].T.dot(dZ3) * deriv_ReLU(cache["Z2"])
    dW2 = 1 / m * dZ2.dot(cache["A1"].T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = params["W2"].T.dot(dZ2) * deriv_ReLU(cache["Z1"])
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3,
    }
    return grads

def update_params(params, grads, alpha):
    for key in params.keys():
        if key.startswith('W') or key.startswith('b'):
            params[key] -= alpha * grads["d" + key]
    return params

def handle_early_stopping(i, val_accuracy, best_accuracy, patience_counter, patience, params):
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at iteration {i} with best validation accuracy {best_accuracy:.4f}")
        return best_accuracy, patience_counter

    if i % 200 == 0 and i != 0:
        save_params(params, "autosave_state", i, autosave_state=True)

    return best_accuracy, patience_counter
