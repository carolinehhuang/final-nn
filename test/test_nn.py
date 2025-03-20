# TODO: import dependencies and write unit tests below

from nn import NeuralNetwork
from nn import preprocess
import numpy as np
from sklearn import metrics

nn_arch = [
    {
        'input_dim': 64,
        'output_dim': 16,
        'activation': 'relu'
    },
    {
        'input_dim': 16,
        'output_dim': 64,
        'activation': 'relu'
    }
]

lr = 0.001
seed = 42
batch_size = 4
epochs = 100
loss_function = 'binary_cross_entropy'

nn = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

#generate matrix of normally distributed values to assign to weights, biases and assertions
W_curr = np.random.normal(size=(10, 10))
b_curr = np.random.normal(size=10)
A_prev = np.random.normal(size=(10,1))
Z_curr = np.random.random(size=(10, 1))
dA_curr = np.random.random(size=(10, 1))

#create random input dataset
X = np.random.random((10, 64))

def test_single_forward():
    #run single forward for both relu and sigmoid
    A_curr_relu, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, 'relu')
    A_curr_sig, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, 'sigmoid')

    assert np.all(Z_curr == np.dot(W_curr, A_prev) + b_curr)
    assert np.all(A_curr_relu == np.maximum(Z_curr, 0))
    assert np.all(A_curr_sig == 1 / (1 + np.exp(-Z_curr)))

def test_forward():
    output, cache = nn.forward(X)

    # Check cache matches expected values; cache keys should be up to number of layers + A0
    assert set(cache.keys()) == {'A0', 'Z1', 'A1', 'Z2', 'A2'}
    assert np.all(cache['A0'] == X.T)

def test_single_backprop():
    dZ_relu = nn._relu_backprop(dA_curr, Z_curr)
    dZ_sig = nn._sigmoid_backprop(dA_curr, Z_curr)

    #manually calculate all the backprop functions
    dW_curr_test_relu = np.dot(dZ_relu, A_prev.T)
    dW_curr_test_sigmoid = np.dot(dZ_sig, A_prev.T)
    db_curr_test_relu = np.sum(dZ_relu, axis=1, keepdims=True)
    db_curr_test_sigmoid = np.sum(dZ_sig, axis=1, keepdims=True)
    dA_prev_test_relu = np.dot(W_curr.T, dZ_relu)
    dA_prev_test_sigmoid = np.dot(W_curr.T, dZ_sig)

    #use built in function to calculate
    dA_prev_relu, dW_curr_relu, db_curr_relu = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr,
                                                                            'relu')
    dA_prev_sig, dW_curr_sig, db_curr_sig = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr,
                                                                         'sigmoid')

    assert np.all(dW_curr_relu == dW_curr_test_relu)
    assert np.all(dW_curr_sig == dW_curr_test_sigmoid)

    assert np.all(db_curr_relu == db_curr_test_relu)
    assert np.all(db_curr_sig == db_curr_test_sigmoid)

    assert np.all(dA_prev_relu == dA_prev_test_relu)
    assert np.all(dA_prev_sig == dA_prev_test_sigmoid)

def test_predict():
    output, cache = nn.forward(X)
    pred = nn.predict(X)

    # Predicted labels should be the same as the forward labels
    assert np.all(output == pred)

def test_binary_cross_entropy():
    y_hat = np.array([1,0,1,1])
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    y = np.array([1,0,1,0])

    loss_test = float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
    loss = nn._binary_cross_entropy(y, y_hat)

    assert loss == loss_test
    assert loss > 0

def test_binary_cross_entropy_backprop():
    y_hat = np.array([1, 0, 1, 1])
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    y = np.array([1, 0, 1, 0])

    loss_backprop_test = (1/batch_size) * (-y/y_hat + (1-y)/(1-y_hat))
    loss_backprop = nn._binary_cross_entropy_backprop(y, y_hat)

    assert np.all(loss_backprop == loss_backprop_test)

def test_mean_squared_error():
    y_hat = np.array([1, 0, 1, 1])
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    y = np.array([1, 0, 1, 0])

    loss_test = 0.25
    loss = nn._mean_squared_error(y, y_hat)

    assert np.allclose(loss,loss_test)

def test_mean_squared_error_backprop():
    y_hat = np.array([1, 0, 1, 1])
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    y = np.array([1, 0, 1, 0])

    loss_test = [0, 0, 0, 0.5]
    loss = nn._mean_squared_error_backprop(y, y_hat)

    assert np.allclose(loss,loss_test)

def test_sample_seqs():
    seqs = ["CC", "CG", "CA", "AA", "GA", "GG", "GC", "GT", "AT","AC"]
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1]).astype(np.bool)

    true_g = [seq for seq, label in zip(seqs, labels) if label]
    false_g = [seq for seq, label in zip(seqs, labels) if not label]
    balanced_class_n = max(len(true_g), len(false_g))

    seqs_bal, labels_bal = preprocess.sample_seqs(seqs, labels)
    labels_bal = np.array(labels_bal)

    #assert that the length of the balanced sequence is double the amount of the maximum count between the two classes
    assert(len(seqs_bal) == 2 * balanced_class_n)

    #assert that the number of each label is the same
    assert(len(np.where(labels_bal != 0)) == len(np.where(labels_bal == 0)))

def test_one_hot_encode_seqs():
    seqs = ["CC", "CG", "CA", "AA", "GA", "GG", "GC", "GT", "AT", "AC"]
    seq_one_hot = preprocess.one_hot_encode_seqs(seqs)

    #since each sequence is made up of two basepairs, the length of the one hot encoded sequence should be 2 * 4 = 8
    assert(len(seq_one_hot[0]) == 8)

    #check to see if the function is encoding correctly
    assert np.all(seq_one_hot[1] == [0, 0, 1, 0, 0, 0, 0, 1])