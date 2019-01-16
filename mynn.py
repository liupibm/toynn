import numpy as np
import math


class Layer:
    def __init__(self, num_node, activation_func=None):
        self._num_node = num_node
        self._activation_func = activation_func
        self._input = np.zeros((1, self._num_node))

        self._output = np.zeros((1, self._num_node))

        self._weights_for_layer = None
        self._bias_for_layer = None

        self._next_layer = None
        self._pre_layer = None

        # these are for backward analysis
        self._d_output = None
        self._d_input = None
        self._d_weights = None
        self._d_bias = None

    def connect(self, next_layer):
        self._weights_for_layer = np.random.normal(0, 0.001, size=(self._num_node, next_layer._num_node))
        self._bias_for_layer = np.random.normal(0, 0.001, size=(1, next_layer._num_node))
        self._next_layer = next_layer
        next_layer._pre_layer = self

    def sigmoid(self, layer):
        return 1/(1 + np.exp(-layer))

    def sigmoid_derivative(self, layerout):
        return layerout * (1 - layerout)

    def forward(self): # get the values ready
        # precondition: input is available.
        # from my input to next's input.
        if self._activation_func: #
            if self._activation_func == "sigmoid":
                self._output = self.sigmoid(self._input)
            else:
                raise Exception("not supported yet")
        else:
            self._output = np.copy(self._input)

        if self._next_layer: # if we still next
            self._next_layer._input = self._output @ self._weights_for_layer + self._bias_for_layer

    def backward(self): # get the derivatives ready
        # precondition: d_output is available
        # task 1: from my d_output to pre's d_output
        # task 2: d_W and d_b used for gradient descent

        # task 1:
        if self._activation_func: #
            if self._activation_func == "sigmoid":
                self._d_input = self._d_output * self.sigmoid_derivative(self._output)
            else:
                raise Exception("not supported yet")
        else:
            self._d_input = np.copy(self._d_output)

        if self._pre_layer:
            self._pre_layer._d_output = self._d_input @ self._pre_layer._weights_for_layer.T
        # task 2:
            self._pre_layer._d_weights = self._pre_layer._output.T @ self._d_input
            self._pre_layer._d_bias = self._d_input

    def update_parameters(self, learning_rate):
        if self._weights_for_layer is not None:
            self._weights_for_layer -= learning_rate * self._d_weights
            self._bias_for_layer -= learning_rate * self._d_bias

class Neural_Network(object):
    def __init__(self):
      #parameters
        self.layers = []
        self.layers.append(Layer(2))
        self.layers.append(Layer(3, activation_func="sigmoid"))
        self.layers.append(Layer(1, activation_func="sigmoid"))
        self.lineup()

        self.learning_rate = 0.5

    def lineup(self):
        l = len(self.layers)
        for i in range(l-1):
            cur = self.layers[i]
            next = self.layers[i+1]
            cur.connect(next)



    def forward(self, data):
        self.layers[0]._input = data
        ret = None
        for layer in self.layers:
            layer.forward()
            ret = layer._output
        return ret

    def gradient_descent(self):
        for layer in self.layers:
            layer.update_parameters(self.learning_rate)

    def backward(self, X, target, oracle):
        d_target = (target - oracle) # derivative w.r.t the target/output
        self.layers[-1]._d_output = d_target
        for layer in reversed(self.layers):
            layer.backward()
        self.gradient_descent()

    def train(self, data, oracle, epoch=1): # batch size is fixed as 1
        totalpoints = len(data)
        dataarray = np.split(data, len(data),axis=0)
        oraclearray = np.split(oracle, len(oracle),axis=0)
        for i in range(epoch):
            target = nn.forward(dataarray[i%totalpoints])
            nn.backward(dataarray[i%totalpoints], target, oraclearray[i%totalpoints])

    def predict(self, data):
        target = nn.forward(data)
        return target


# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) #
y = np.array(([92], [86], [89]), dtype=float) #

# scale units
# X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

nn = Neural_Network()
nn.train(X, y, epoch=100000)
print(nn.predict(X))




