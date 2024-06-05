import numpy as np
import matplotlib.pyplot as plt
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)


def step(x):
    return np.where(x<0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

x = np.linspace(-1.5, 1.5, 100000)

#plt.figure(figsize=(6.5,4))
plt.plot(x, step(x), 'k-', linewidth=0.5, label='Step function')
plt.plot(x, sigmoid(x), 'k--', alpha=0.7, label='Sigmoid')
plt.plot(x, tanh(x), 'k-.', alpha=0.7, label='Tanh')
plt.plot(x, relu(x), 'k:', linewidth=2, label='ReLU')
plt.plot(x, leaky_relu(x), 'k-', alpha=0.7, label='Leaky ReLU')

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xlabel('Input')
plt.ylabel('Output')

save()
