import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

config = {}

"""
# Original config
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm
"""

# 2c) Configuration
config['layer_specs'] = [784, 50,
                         10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 300  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.08  # Learning rate of gradient descent algorithm

"""
# 2d) Configuration
config['L2_penalty'] = 0.001
config['epochs'] = 140

# 2e) Configuration
config['activation'] = 'sigmoid'
config['activation'] = 'ReLU'

# 2f) Configuration
config['layer_specs'] = [784, 25, 10]  # 2fa
config['layer_specs'] = [784, 100, 10]  # 2fb
config['layer_specs'] = [784, 45, 45, 10]  #2fc
"""


def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    output = np.exp(x)
    return output / np.sum(output, axis=1, keepdims=True)


def load_data(fname):
    """
      Write code to read the data and return it as 2 numpy arrays.
      Make sure to convert labels to one hot encoded format.
    """
    # Open file with pickle
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    # Split the images and labels
    images = data[:, :-1]
    labels = data[:, -1].astype(int)
    # Encode one hot labels
    one_hot_labels = np.zeros((len(labels), len(np.unique(labels))))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return images, one_hot_labels


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type
        self.x = None  # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

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
        """
        Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        """
        Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return np.maximum(x, 0)

    def grad_sigmoid(self):
        """
        Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
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
        loss = None

        processed_a = x
        # Process input in each layer
        for layer in self.layers:
            processed_a = layer.forward_pass(processed_a)
        self.y = softmax(processed_a)

        # Calculate Losses
        if targets is not None:
            loss = self.loss_func(self.y, self.targets)
        return loss, self.y

    def loss_func(self, logits, targets):
        """
        find cross entropy loss between logits and targets
        """
        return -np.sum(targets * np.log(logits)) / logits.shape[0]

    def backward_pass(self):
        """
        implement the backward pass for the whole network.
        hint - use previously built functions.
        """
        # the gradient of cross-entropy on top of softmax is (t-y)
        back_output = (self.targets - self.y) / self.y.shape[0]

        for layer in reversed(self.layers):
            back_output = layer.backward_pass(back_output)


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
    Write the code to train the network. Use values from config to set parameters
    such as L2 penalty, number of epochs, momentum, etc.
    """

    batch_size = config['batch_size']
    alpha = config['learning_rate']
    penalty = config['L2_penalty']
    momentum = config['momentum_gamma']

    num_train = X_train.shape[0]
    num_batches = int(np.ceil(num_train / config['batch_size']))

    train_accu = []
    valid_accu = []

    valid_losses = []
    train_losses = []

    check_early = []
    best_test_accu = []

    for n in range(config['epochs']):
        print('Current Epoch is ' + str(n + 1))

        if (n + 1) % 10 == 0:
            print(max(best_test_accu))

        for i in range(num_batches):

            # Get train and target sets for the batch training
            batch_train_X = X_train[i * batch_size: (i + 1) * batch_size]
            batch_train_y = y_train[i * batch_size: (i + 1) * batch_size]

            # Forward Pass
            train_loss, train_pred = model.forward_pass(batch_train_X, batch_train_y)

            # Backward Pass
            model.backward_pass()

            # Update weights and bias
            for layer in model.layers:
                if isinstance(layer, Layer):
                    # With momentum
                    if config['momentum']:
                        if not hasattr(layer, 'prev_w'):
                            layer.prev_w = np.zeros_like(layer.w)

                        # Weights and bias update
                        d_w = layer.d_w * alpha + momentum * layer.prev_w
                        layer.w = layer.w * (1 - alpha * penalty / batch_size) + d_w
                        layer.prev_w = d_w
                        layer.b = layer.d_b * alpha
                    # Without momentum
                    else:
                        layer.w = layer.w * (1 - alpha * penalty / batch_size) + layer.d_w * alpha

        # Calculate validation loss and accuracy
        valid_loss, valid_pred = model.forward_pass(X_valid, y_valid)

        train_accu.append(test(model, X_train, y_train, config))
        valid_accu.append(test(model, X_valid, y_valid, config))

        valid_losses.append(valid_loss)
        train_losses.append(train_loss)
        best_test_accu.append(test(model, X_test, y_test, config))

        if config["early_stop"]:
            if not check_early:
                check_early.append(valid_loss)
            else:
                if len(check_early) == config['early_stop_epoch']:
                    break
                elif valid_loss >= check_early[-1]:
                    check_early.append(valid_loss)
                else:
                    check_early = []

    print('The best test accuracy is ' + str(max(best_test_accu) * 100) + '%.')
    print('The best epoch is epoch ' + str(best_test_accu.index(max(best_test_accu))) + '.')
    plot_train_vad_accu_and_losses('2c', train_accu, valid_accu, train_losses, valid_losses, config)  # 2c)
    # plot_train_vad_accu_and_losses('2d', train_accu, valid_accu, train_losses, valid_losses, config)  # 2d)
    # plot_train_vad_accu_and_losses('2ea', train_accu, valid_accu, train_losses, valid_losses, config)  # 2e)a
    # plot_train_vad_accu_and_losses('2eb', train_accu, valid_accu, train_losses, valid_losses, config)  # 2e)b
    # plot_train_vad_accu_and_losses('2fa', train_accu, valid_accu, train_losses, valid_losses, config)  # 2f)a
    # plot_train_vad_accu_and_losses('2fb', train_accu, valid_accu, train_losses, valid_losses, config)  # 2f)b
    # plot_train_vad_accu_and_losses('2fc', train_accu, valid_accu, train_losses, valid_losses, config)  # 2f)c


def plot_train_vad_accu_and_losses(question, train_accu, valid_accu, train_losses, valid_losses, config):
    graph_title = 'Training vs Validation Accuracies for ' + str(config['epochs']) \
                  + ' Epochs with Early Stopping with Hidden Layer of Activation Function ' + config['activation']
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    ax.plot(np.arange(1, len(train_accu) + 1), np.array(train_accu) * 100, label='Training Accuracies(%)')
    ax.plot(np.arange(1, len(valid_accu) + 1), np.array(valid_accu) * 100, label='Validation Accuracies(%)')
    ax.set(xlabel='Number of Epochs', ylabel='Accuracy(%)',
           title=graph_title)
    leg = ax.legend(loc=4)
    fig.savefig('graphs/' + question + 'train_vad_accu.png')

    graph_title = 'Training vs Validation Losses for ' + str(config['epochs']) \
                  + ' Epochs with Early Stopping with Hidden Layer of Activation Function ' + config['activation']
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label='Training Losses')
    ax.plot(np.arange(1, len(valid_losses) + 1), valid_losses, label='Validation Losses')
    ax.set(xlabel='Number of Epochs', ylabel='Losses',
           title=graph_title)
    leg = ax.legend(loc=4)
    fig.savefig('graphs/' + question + 'train_vad_loss.png')


def caclulate_accuracy_of_predictions(predictions, labels):
    """
    :param predictions: predictions using current model
    :param labels: the actual labels of each item
    :return: the accuracy of the current model
    """
    decisions = (predictions == predictions.max(axis=1)[:, None]).astype(np.int)

    diff = labels + decisions
    AGREEMENT_VALUE = 2
    diff[diff < AGREEMENT_VALUE] = 0
    diff[diff == AGREEMENT_VALUE] = 1

    number_of_images = labels.shape[0]
    return np.sum(diff) / number_of_images


def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    loss, predictions = model.forward_pass(X_test, y_test)
    return caclulate_accuracy_of_predictions(predictions, y_test)


def numerical_comparison(model, i, j, k, epsilon, X_train, y_train):
    layer_names = ['input_to_hidden_layer', 'hidden_to_output_layer']

    # Use to store numerical approximation and backward gradient for w
    num_approx_w = []
    back_grad_w = []

    # Use to store numerical approximation and backward gradient for d
    num_approx_d = []
    back_grad_d = []

    for layer in model.layers:
        if isinstance(layer, Layer):
            # Numerical operation on layer for w
            layer.w[i][j] -= epsilon
            minus_loss, _ = model.forward_pass(X_train, y_train)
            layer.w[i][j] += 2 * epsilon
            add_loss, _ = model.forward_pass(X_train, y_train)
            num_approx_w.append((add_loss - minus_loss) / (epsilon * 2))

            # Numerical operation on layer for d
            layer.w[i][j] -= epsilon
            layer.b[0][k] -= epsilon
            minus_loss, _ = model.forward_pass(X_train, y_train)
            layer.b[0][k] += 2 * epsilon
            add_loss, _ = model.forward_pass(X_train, y_train)
            num_approx_d.append((add_loss - minus_loss) / (epsilon * 2))

            # Gradient obtained by backprop
            layer.b[0][k] -= epsilon
            model.forward_pass(X_train, y_train)
            model.backward_pass()
            back_grad_w.append(-layer.d_w[i][j])
            back_grad_d.append(-layer.d_b[0][k])

    data_table = pd.DataFrame({'layer': layer_names,
                               'num_ap_w': num_approx_w,
                               'bp_grad_w': back_grad_w,
                               'num_ap_b': num_approx_d,
                               'bp_grad_b': back_grad_d})
    data_table.to_csv('./part2b.csv')


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
    test_acc = test(model, X_test, y_test, config)
