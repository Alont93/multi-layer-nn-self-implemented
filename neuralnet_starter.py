import numpy as np
import pandas as pd
import pickle


config = {}
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

def softmax(x):
  e_x = np.exp(x)
  return e_x / e_x.sum(axis=0)


def load_data(fname):
  with open(fname, 'rb') as f:
    data = pickle.load(f)

  images = data[:,:-1]
  labels = encode_onehot_labels(data[:,-1])
  return images, labels


# parse the lables to sparse matrix
def encode_onehot_labels(labels):
    data_table = pd.DataFrame({'number': labels})
    data_table = pd.concat((data_table, pd.get_dummies(data_table.number)), 1)
    data_table = data_table.drop(columns=['number'])
    relevant_labels = data_table.values
    return relevant_labels



class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.relu(a)
  
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
    e_x = np.exp(x)
    return e_x / (e_x + 1)

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
    self.a = np.matmul(np.c_[self.b,self.w],x)  # Weighted sum of x with weight matrix(augmented with bias)
    self.x = x
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    self.d_x = np.matmul(delta,np.transpose(np.c_[self.b,self.w]))
    grad = np.matmul(delta,np.transpose(self.x))
    self.d_b,self.d_w = np.split(grad,[1])
    return self.d_x

      
class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))  
    
  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    self.targets = targets

    processed_x = x
    for layer in self.layers:
      processed_x = layer.forward_pass(processed_x)

    self.y = softmax(processed_x)
    if targets != None:
      loss = self.loss_func(self.y, self.targets)

    return loss, self.y

  def loss_func(self, logits, targets):
    amount_of_data = targets.shape[0]
    return -np.sum(targets * np.log(logits)) / amount_of_data

  def backward_pass(self):
    # the gradient of cross-entropy on top of softmax is (t-y)
    back_output = self.targets - self.y
    weights_gradients = []

    for layer in reversed(self.layers):
      back_output = layer.backward_pass(back_output)
      if isinstance(layer, Layer):
        weights_gradients.append(layer.d_w)

    # update weights with learning rule
    for layer in self.layers:
      if isinstance(layer, Layer):
        alpha = config['learning_rate']
        layer.w = layer.w - alpha * layer.d_w
        layer.b = layer.b - alpha * layer.d_b

    loss = self.loss_func(self.y, self.targets)
    return loss, self.y
      

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  
  
def test(model, X_test, y_test, config):
  nn = Neuralnetwork(config)

  predictions = nn.forward_pass(X_test, y_test)
  decisions = (predictions == predictions.max(axis=1)[:,None]).astype(np.int)

  diff = y_test + decisions
  AGREEMENT_VALUE = 2
  diff[diff < AGREEMENT_VALUE] = 0
  diff[diff == AGREEMENT_VALUE] = 1

  number_of_images = X_test.shape[0]
  return np.sum(diff) / number_of_images


# calculate the specific weights accuracy
def get_softmax_weights_accuracy(pca_images, labels, weights):

      

if __name__ == "__main__":
  train_data_fname = 'data/MNIST_train.pkl'
  valid_data_fname = 'data/MNIST_valid.pkl'
  test_data_fname = 'data/MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)

