"""tinynet - a small, readable neural network library written from scratch in NumPy."""

from .layers import Dense, Dropout, Layer, LeakyReLU, ReLU, Sigmoid, Tanh, sigmoid, softmax
from .losses import MSE, BinaryCrossEntropy, Loss, SoftmaxCrossEntropy
from .model import History, Sequential
from .optimizers import SGD, Adam, CosineDecay, Optimizer, RMSProp, StepDecay

__version__ = "1.0.0"

__all__ = [
    "Layer", "Dense", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Dropout", "sigmoid", "softmax",
    "Loss", "SoftmaxCrossEntropy", "BinaryCrossEntropy", "MSE",
    "Sequential", "History",
    "Optimizer", "SGD", "RMSProp", "Adam", "StepDecay", "CosineDecay",
]
