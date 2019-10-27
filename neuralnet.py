import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 10  # Number of epochs to train the model
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
        return -np.sum(targets * np.log(logits))


    def backward_pass(self):
        # the gradient of cross-entropy on top of softmax is (t-y)
        back_output = self.targets - self.y

        for layer in reversed(self.layers):
            back_output = layer.backward_pass(back_output)

    def get_layers_weights_and_biases(self):
        weights = []
        biases = []

        for layer in self.layers:
            if isinstance(layer, Layer):
                weights.append(layer.w)
                biases.append(layer.b)

        return weights, biases

    def apply_weights_and_biases_on_layers(self, weights, biases):
        layer_index = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.w = weights[layer_index]
                layer.b = biases[layer_index]
                layer_index += 1


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    batch_size = int(config['batch_size'])
    alpha = config['learning_rate']
    L2_pen = config['L2_penalty']
    number_of_training_samples = X_train.shape[0]
    num_of_batches = int(np.ceil(number_of_training_samples / config['batch_size']))

    for n in range(int(config['epochs'])):
        #all_indices = np.arange(X_train.shape[0])
        #np.random.shuffle(all_indices)
        #X_train = X_train[all_indices]
        #y_train = y_train[all_indices]
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
            #print("validation accur is", caclulate_accuracy_of_predictions(valid_pred, y_valid))

   #training_errors, validation_errors, training_accuracies, validation_accuracies \
    #    = batch_stochastic_gradient_decent(model, X_train, y_train, X_valid, y_valid)

    #plot_train_and_validation_loss(training_errors, validation_errors)
    #plot_train_and_validation_accuracy(training_accuracies, validation_accuracies)


def plot_train_and_validation_loss(training_errors, validation_errors):
    plt.plot(training_errors, label="training loss")
    plt.plot(validation_errors, label="validation loss")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.title("Loss as a function of the number of epochs")
    plt.legend()
    plt.show()


# batch and stochastic gradient decent implementation for two classes
def plot_train_and_validation_accuracy(training_accuracies, validation_accuracies):
    plt.plot(training_accuracies, label="training accuracy")
    plt.plot(validation_accuracies, label="validation accuracy")
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.title("Accuracy as a function of the number of epochs")
    plt.legend()
    plt.show()


def update_single_batch_loss_and_accuracy(batch_loss, predictions, batch_labels, epoch_train_accuracies,
                                          epoch_train_errors):
    batch_accuracy = caclulate_accuracy_of_predictions(predictions, batch_labels)
    epoch_train_errors.append(batch_loss)
    epoch_train_accuracies.append(batch_accuracy)


def update_validation_loss_and_accuracy(nn, validation_accuracies, validation_errors, validation_labels,
                                        validation_samples):
    validation_loss, predictions = nn.forward_pass(validation_samples, validation_labels)
    validation_errors.append(validation_loss)
    validation_accuracies.append(caclulate_accuracy_of_predictions(predictions, validation_labels))

    return validation_loss


def update_train_loss_and_acuuracy(epoch_train_accuracies, epoch_train_errors, training_accuracies, training_errors):
    average_train_epoch_loss = np.average(np.array(epoch_train_errors))
    training_errors.append(average_train_epoch_loss)
    average_train_epoch_accuracy = np.average(np.array(epoch_train_accuracies))
    training_accuracies.append(average_train_epoch_accuracy)


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
    print("test accuracy is", test_acc)

