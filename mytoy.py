import numpy as np
import math
# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
# X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
      #parameters
      self.inputSize = 2
      self.outputSize = 1
      self.hiddenSize = 3

      self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
      self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

  def sigmoid(self, layer):
      return 1/(1 + np.exp(-layer))
  def sigmoid_derivative(self, layerout):
      return layerout * (1 - layerout)

  def forward(self, data):
      self.hidden = self.sigmoid(data @ self.W1)
      self.result = self.hidden @ self.W2
      return self.sigmoid(self.result)

  def backward(self, X, target, oracle):
      error = (target - oracle)
      dlastin = error * self.sigmoid_derivative(target)

      dhiddenout = dlastin @ self.W2.T
      dhiddenin = dhiddenout * self.sigmoid_derivative(self.hidden)
      dW1 = X.T @ dhiddenin
      self.W1 -= dW1

      dW2 = self.hidden.T @ dlastin
      self.W2 -= dW2

nn = Neural_Network()

for i in range(10000):
    target = nn.forward(X)
    nn.backward(X, target, y)

print(target)




