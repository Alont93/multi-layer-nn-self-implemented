import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 2  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001  # Learning rate of gradient descent algorithm

RAISING_VALIDATION_ERROR_THRESHOLD = 5


def softmax(x):
    result = np.exp(x - np.max(x, axis=1, keepdims=True))
    result = result/np.sum(result, axis=1, keepdims=True)
    return result


def load_data(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    images = data[:, :-1]
    labels = encode_onehot_labels(data[:, -1])
    return images, labels


# parse the lables to sparse matrix
def encode_onehot_labels(labels):
    data_table = pd.DataFrame({'number': labels})
    data_table = pd.concat((data_table, pd.get_dummies(data_table.number)), 1)
    data_table = data_table.drop(columns=['number'])
    relevant_labels = data_table.values
    return relevant_labels


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type
        self.x = None
        # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        self.x = x
        return np.maximum(x, 0)

    def grad_sigmoid(self):
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        return 1 - np.square(self.tanh(self.x))

    def grad_ReLU(self):
        grad = self.x.copy()
        grad[grad <= 0] = 0
        grad[grad > 0] = 1
        return grad


class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer. Do not apply activation function here.
        """
        self.a = (self.w.T @ x.T).T + self.b  # Weighted sum of x with weight matrix(augmented with bias)
        self.x = x
        return self.a

    def backward_pass(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        self.d_x = np.dot(delta, self.w.T)
        self.d_b = np.matmul(np.ones((1, delta.shape[0])), delta)
        self.d_w = np.dot(self.x.T, delta)
        return self.d_x


class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def forward_pass(self, x, targets):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        self.targets = targets

        processed_a = x
        for layer in self.layers:
            processed_a = layer.forward_pass(processed_a)
        self.y = softmax(processed_a)

        if targets is not None:
            loss = self.loss_func(self.y, self.targets)
        return loss, self.y

    def loss_func(self, logits, targets):
        return -np.sum(targets * np.log(logits))/logits.shape[0]


    def backward_pass(self):
        # the gradient of cross-entropy on top of softmax is (t-y)
        back_output = (self.targets - self.y)/self.y.shape[0]

        for layer in reversed(self.layers):
            back_output = layer.backward_pass(back_output)


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    batch_size = int(config['batch_size'])
    alpha = config['learning_rate']
    L2_pen = config['L2_penalty']
    number_of_training_samples = X_train.shape[0]
    num_of_batches = int(np.ceil(number_of_training_samples / config['batch_size']))

    for n in range(int(config['epochs'])):
        all_indices = np.arange(X_train.shape[0])
        np.random.shuffle(all_indices)
        X_train = X_train[all_indices]
        y_train = y_train[all_indices]
        print("epoch ", n)

        # For loop over mini batches
        for i in range(num_of_batches):

            # Get train set and target sets for the batch training
            batch_train = X_train[i * batch_size: (i+1) * batch_size]
            batch_train_y = y_train[i * batch_size: (i+1) * batch_size]

            # Forward pass
            loss_train, output_train = model.forward_pass(batch_train, batch_train_y)

            # Backward pass
            model.backward_pass()

            # Update weights and bias
            for layer in model.layers:
                if isinstance(layer, Layer):
                    # TODO: add momentum and regularization
                    layer.w = layer.w + alpha * layer.d_w
                    layer.b = alpha * layer.d_b

            # Calculate validation loss and accuracy
            valid_loss, valid_pred = model.forward_pass(X_valid, y_valid)


def caclulate_accuracy_of_predictions(predictions, labels):
    decisions = (predictions == predictions.max(axis=1)[:, None]).astype(np.int)

    diff = labels + decisions
    AGREEMENT_VALUE = 2
    diff[diff < AGREEMENT_VALUE] = 0
    diff[diff == AGREEMENT_VALUE] = 1

    number_of_images = labels.shape[0]
    return np.sum(diff) / number_of_images


def test(model, X_test, y_test, config):
    loss, predictions = model.forward_pass(X_test, y_test)
    return caclulate_accuracy_of_predictions(predictions, y_test)


def numerical_comparison(model, i, j, m, n, k, epsilon, X_train, y_train):
    layer_names = ['input_to_hidden_layer', 'hidden_to_output_layer']

    # Use to store numerical approximation and backward gradient for w1, w2
    num_approx_w1 = []
    back_grad_w1 = []
    num_approx_w2 = []
    back_grad_w2 = []

    # Use to store numerical approximation and backward gradient for d
    num_approx_d = []
    back_grad_d = []

    for layer in model.layers:
        if isinstance(layer, Layer):
            # Numerical operation on layer for w1
            layer.w[i][j] -= epsilon
            minus_loss, _ = model.forward_pass(X_train, y_train)
            layer.w[i][j] += 2 * epsilon
            add_loss, _ = model.forward_pass(X_train, y_train)
            num_approx_w1.append((add_loss-minus_loss)/(epsilon * 2))
            layer.w[i][j] -= epsilon

            # Numerical operation on layer for w2
            layer.w[m][n] -= epsilon
            minus_loss, _ = model.forward_pass(X_train, y_train)
            layer.w[m][n] += 2 * epsilon
            add_loss, _ = model.forward_pass(X_train, y_train)
            num_approx_w2.append((add_loss-minus_loss)/(epsilon * 2))
            layer.w[m][n] -= epsilon

            # Numerical operation on layer for d
            layer.b[0][k] -= epsilon
            minus_loss, _ = model.forward_pass(X_train, y_train)
            layer.b[0][k] += 2 * epsilon
            add_loss, _ = model.forward_pass(X_train, y_train)
            num_approx_d.append((add_loss-minus_loss)/(epsilon * 2))

            # Gradient obtained by backprop
            layer.b[0][k] -= epsilon
            model.forward_pass(X_train, y_train)
            model.backward_pass()
            back_grad_w1.append(-layer.d_w[i][j])
            back_grad_w2.append(-layer.d_w[m][n])
            back_grad_d.append(-layer.d_b[0][k])

    data_table = pd.DataFrame({'layer': layer_names,
                               'num_ap_w1': num_approx_w1,
                               'bp_grad_w1': back_grad_w1,
                               'num_ap_w2': num_approx_w2,
                               'bp_grad_w2': back_grad_w2,
                               'num_ap_b': num_approx_d,
                               'bp_grad_b': back_grad_d})
    data_table.to_csv('./part2b.csv')
    print(data_table)



if __name__ == "__main__":
    train_data_fname = 'MNIST_train.pkl'
    valid_data_fname = 'MNIST_valid.pkl'
    test_data_fname = 'MNIST_test.pkl'

    ### Train the network ###
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    numerical_comparison(model, 0, 0, 0,1,1, 0.01, X_train, y_train)
    test_acc = test(model, X_test, y_test, config)
    print("test accuracy is", test_acc)

