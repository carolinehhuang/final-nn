# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        #calculate the weighted outputs based on the activation values of the previous layer, the weights, and the biases
        z_curr = np.dot(W_curr, A_prev) + b_curr

        #caclulate the activation values based on the layer's activation function
        if activation == "relu":
            a_curr = self._relu(z_curr)
        elif activation == "sigmoid":
            a_curr = self._sigmoid(z_curr)

        return a_curr, z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        #initialize cache
        a_prev = X.T
        cache = {'A0' : a_prev}

        for layer, feat in enumerate(self.arch):
            #dictionary starts at 1
            layer_idx = layer + 1
            # get the weights and biases of the current layer by calling it from the dictionary
            w_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]

            #compute for each layer
            a_curr, z_curr = self._single_forward(w_curr, b_curr, a_prev, feat['activation'])

            #store layer name and activation values for the layer in cache
            cache[f"A{layer_idx}"] = a_curr

            #assign total weighted sums of inputs to layer in cache
            cache[f"Z{layer_idx}"] = z_curr

            #update the new a_prev to the activations calculated from this layer for the next iteration
            a_prev = a_curr

        return a_prev, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        #calculate the derivative of weighted outputs based on the activation function of its previous later
        if activation_curr == "relu":
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == "sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)

        #calculate the partial derivatives of weight, bias and previous activation layer to determine how much it influenced the output
        dW_curr = np.dot(dZ_curr, A_prev.T)
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # check loss function
        if self._loss_func == "binary_cross_entropy":
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            dA_curr = self._mean_squared_error_backprop(y, y_hat)

        #initialize the gradient dictionary to store the values of the partial derivatives
        gradient_dict = {}

        for layer in range(len(self.arch)-1, -1, -1):
            #get original layer index, no need to -1 because the dictionary indices start at 1
            layer_idx = layer + 1

            gradient_dict[f"dA{layer_idx}"] = dA_curr

            #retrive the values needed for backpropagation from the cache and the dictionary
            a_prev = cache[f"A{layer}"]
            z_curr = cache[f"Z{layer_idx}"]
            w_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = self.arch[layer]['activation']

            #compute backpropagation for each layer
            dA_prev, dW_curr, db_curr = self._single_backprop(w_curr, b_curr, z_curr, a_prev, dA_curr, activation)

            #update gradient dictionary for the partial derivatives
            gradient_dict[f"dW{layer_idx}"] = dW_curr
            gradient_dict[f"db{layer_idx}"] = db_curr

            dA_curr = dA_prev

        return gradient_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        #update the parameters to improve the model
        for layer, feat in enumerate(self.arch):
            layer_idx = layer + 1
            self._param_dict['W' + str(layer_idx)] -= (self._lr * grad_dict[f"dW{layer_idx}"])
            self._param_dict['b' + str(layer_idx)] -= (self._lr * np.mean(grad_dict[f"db{layer_idx}"]))

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        #initialize lists for loss of training and validation sets
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        m = len(X_train)

        for epoch in range(self._epochs):
            rand_ind = np.random.permutation(m)
            X_train = X_train[rand_ind]
            y_train = y_train[rand_ind]
            loss_sum = 0

            #break the training set into mini training sets by batch size and run for each batch set
            for i in range (0, m, self._batch_size):
                x_mini = X_train[i:i+self._batch_size]
                y_mini = y_train[i:i+self._batch_size].T

                #forward pass
                pred, cache = self.forward(x_mini)

                #loss
                if self._loss_func == "binary_cross_entropy":
                    loss_batch = self._binary_cross_entropy(y_mini, pred)
                else:
                    loss_batch = self._mean_squared_error(y_mini, pred)

                #backpropagation
                gradient = self.backprop(y_mini, pred, cache)

                #update parameters for each pass of the training set
                self._update_params(gradient)
                loss_sum += loss_batch

            per_epoch_loss_train.append(loss_sum/(m/self._batch_size))

            #get a prediction for the validation matrix based on the model trained from the training values for this epoch
            pred_val = self.predict(X_val)

            #store results of loss function calculated based on the model generated predicted values from each epoch
            if self._loss_func == "binary_cross_entropy":
                per_epoch_loss_val.append(self._binary_cross_entropy(y_val.T, pred_val))
            else:
                per_epoch_loss_val.append(self._mean_squared_error(y_val.T, pred_val))

        return per_epoch_loss_train, per_epoch_loss_val


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat


    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1/(1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        #calculate the derivative of the sigmoid
        sigmoid = self._sigmoid(Z)
        return dA * sigmoid * (1 - sigmoid)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * (Z>0).astype(float)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return float(loss)

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        derivative = (1/self._batch_size) * (-y/y_hat + (1-y)/(1-y_hat))
        return derivative

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean((y - y_hat)**2)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        derivative = - (2/self._batch_size) * (y-y_hat)
        return derivative