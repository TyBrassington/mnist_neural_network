import numpy as np
from config import OUTPUT_CLASSES

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