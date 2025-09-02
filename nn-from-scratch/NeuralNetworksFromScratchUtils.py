"""
NeuralNetworksFromScratch.py

This file contains various functions
for neural network implementation using
only Python and NumPy:
- Forward propagation
- Backpropagation
- Activations (Sigmoid, ReLU, softmax)
- Binary and multiclass model training
- Prediction and accuracy evaluation

WARNING: 
This project is focused on learning and demonstrating the internal mechanics
of neural networks - not on robust software engineering.
Exception handling is not built into all of these functions.

Author: James Milgram

"""
import numpy as np
import idx2numpy

def sigmoid(z):
    """
    REQUIRES NUMPY AS NP
    
    Takes a scalar, nd.array
    and transforms it via
    the sigmoid function
    (applied element-wise)

    Args:
        z : scalar, nd.array

    Returns:
        An array
    """
    return 1/(1+np.exp(-z))

def dense_layer_forward(inpt, W, B):
    """
    REQUIRES NUMPY AS NP
    
    Takes the activation/input
    of the previous layer and calculates
    the current activation

    Args:
        inpt : Previous activation/input (2D Array - matrix)
        W : Weights (2D Array - matrix)
        B : Bias (2D Array - row vector)
    
    Returns:
        A : Activation array (2D)
    """
    if not inpt.shape[1] == W.shape[0] or not W.shape[1] == B.shape[1]:
        raise ValueError(f"Array shape mismatch! - inpt: {inpt.shape}, W: {W.shape}, B: {B.shape}")
    z = np.matmul(inpt, W) + B 
    A = sigmoid(z)
    
    return A

def forward_propagation(X, W, B, num_layers):
    """
    REQUIRES NUMPY AS NP
    
    Executes forward propagation
    on a dense neural network

    Args:
        X : Input values (2D Array - matrix)
        W : Weights (Python list of 2D Arrays - matrices)
        B : Bias (Python list of 2D Arrays - row vectors)
        num_layers : # of layers (scalar)
    
    Returns:
        A : Final activation array (2D)
        A_list : Python list of activation arrays (2D)
    """
    A_list = []
    
    for i in range(num_layers): 
        w = W[i]
        b = B[i]
        A = dense_layer_forward(X, w, b)
        X = A
        A_list.append(A)

    return A, A_list

def backward_propagation(X, y, W, B, num_layers, A_list, alpha):
    """
    REQUIRES NUMPY AS NP
    
    Runs through each layer of the
    dense neural network, computes
    the gradients for W and B, and updates
    W and B accordingly
    
    Args:
        X : Input values (2D Array - matrix)
        y : Output values (2D Array - column vector)
        W : Weights (Python list of 2D Arrays - matrices)
        B : Bias (Python list of 2D Arrays - row vectors)
        num_layers : # of layers (scalar)
        A_list : Python list of activations/inputs (2D Arrays - matrices)
        alpha : Learning rate (scalar)
    
    Returns:
        W, B : Python lists of 2D NumPy arrays
    """
    prev_delta = None
    
    for i in range((num_layers-1), -1, -1): 
        if i == num_layers - 1:
            delta = A_list[i] - y 
            prev_A = A_list[i-1]
        
            dL_dW = np.matmul(prev_A.T, delta)
        
            dL_dB = np.sum(delta, axis = 0, keepdims = True)
            
            W[i] = W[i] - alpha * dL_dW
            B[i] = B[i] - alpha * dL_dB

            prev_delta = delta

        elif i != 0:
            delta = np.matmul(prev_delta, W[i+1].T) 
            delta = delta * (A_list[i] * (1 - A_list[i]))
            dL_dW = np.matmul(A_list[i-1].T, delta)
            
            dL_dB = np.sum(delta, axis = 0, keepdims = True)
            
            W[i] = W[i] - alpha * dL_dW
            B[i] = B[i] - alpha * dL_dB
            
            prev_delta = delta

        else:
            delta = np.matmul(prev_delta, W[i+1].T) 
            delta = delta * (A_list[i] * (1 - A_list[i]))
            dL_dW = np.matmul(X.T, delta)
            
            dL_dB = np.sum(delta, axis = 0, keepdims = True)
            
            W[i] = W[i] - alpha * dL_dW
            B[i] = B[i] - alpha * dL_dB

    
    return W, B

def train_neural_network(X, y, layer_list, epochs, alpha, seed):
    """
    REQUIRES NUMPY AS NP
    
    Takes in a set of input data
    and returns the parameters
    for each layer

    Args:
        X : Input values (2D Array - matrix)
        y : Output values (2D Array - column vector)
        layer_list : Python list of # units/layer
        epochs : Number of epochs (scalar)
        alpha : Learning rate (scalar)
        seed : Random seed (scalar)
    
    Returns:
        W, B : Python lists of 2D NumPy arrays
    """
    if not len(layer_list) > 0:
        raise ValueError("Neural network is not defined!")
    if not type(alpha) == float or type(alpha) == int or not alpha > 0.0:
        raise ValueError("Learning Rate is invalid")
    
    np.random.seed(seed)

    num_layers = len(layer_list)
    
    W = []
    B = []

    prev_unit = None
    for idx, unit in enumerate(layer_list):
        if idx == 0:
            W.append(np.random.rand(X.shape[1], unit))
            B.append(np.random.rand(1, unit))
            prev_unit = unit
        else:
            W.append(np.random.rand(prev_unit,unit))
            B.append(np.random.rand(1, unit))
            prev_unit = unit
    
    if not X.shape[0] == y.shape[0] or not W[0].shape[0] == X.shape[1] or not B[0].shape[1] == W[0].shape[1]:
        raise ValueError(f"Array shape mismatch! - X: {X.shape}, y: {y.shape}, W_initial: {W[0].shape}, B_initial: {B[0].shape}")
    
    for i in range(epochs):
        A, A_list = forward_propagation(X, W, B, num_layers)
        W, B = backward_propagation(X, y, W, B, num_layers, A_list, alpha)
        if (i + 1) % 10 == 0:
            print(f"Status: Epoch {i+1} out of {epochs}")


    
    return W, B

def predict_neural_network(X, y, W, B, layer_list):
    """
    REQUIRES NUMPY AS NP
    
    Uses input data
    to classify the data (binary
    classification)

    Args:
        X : Input values (2D Array - matrix)
        y : Output values (2D Array - column vector)
        W : Weights (Python list of 2D Arrays - matrices)
        B : Bias (Python list of 2D Arrays - row vectors)
        layer_list : Python list of # units/layer
    
    Returns:
        A : Prediction array (2D)
    """
    num_layers = len(layer_list) 
    
    A, history = forward_propagation(X, W, B, num_layers)
    
    A = (A>= .5).astype(int)

    A = A.reshape(1,-1)

    # print(f"The predictions for the neural network are (output vector was reshaped for formatting): {A}")

    A = A.reshape(-1,1)
    
    accuracy = np.mean(A == y)

    print(f"The accuracy (%) of the neural network is: {accuracy}")

    return A

def ReLU(z):
    """
    REQUIRES NUMPY AS NP
    
    Takes a NumPy array
    and transforms it via
    the Rectified Linear Unit
    activation function

    Args:
        z : pre-activation (2D NumPy array)

    Returns:
        2D NumPy array
    """
    return np.maximum(0,z)

def safe_softmax(z):
    """
    REQUIRES NUMPY AS NP
    
    Takes a NumPy array
    and transforms it via the
    softmax function; incorporates
    numerical stability
    
    Args:
        z : 2D NumPy array

    Returns:
        2D NumPy array
    """
    shifted_z = z - np.max(z, keepdims = True, axis = 1)
    e_z = np.exp(shifted_z)
    sm = e_z/np.sum(e_z, keepdims = True, axis = 1)
    return sm

def forward_prop_multiclass(X,W,B):
    """
    REQUIRES NUMPY AS NP
    
    Executes forward propagation
    for a dense neural network,
    activation functions are ReLU,
    linear
    
    Args:
        X : Training examples (2D NumPy array)
        W : Weights list (List of 2D NumPy arrays)
        B : Bias list (List of 2D NumPy arrays)

    Returns:
        z : Pre-activation list (List of 2D NumPy arrays)
        a : Activation list (List of 2D NumPy arrays)
    """
    z = []
    a = []

    num_layers = len(W)

    curr_a_i = X

    for i in range(num_layers): 
        z_i = np.matmul(curr_a_i, W[i]) + B[i]
        z.append(z_i)
        
        if i != num_layers - 1:
            a_i = ReLU(z_i)
            curr_a_i = a_i
            a.append(a_i)
            
        else:
            a_i = safe_softmax(z_i)
            a.append(a_i)

    return z, a

def backward_prop_multiclass(X,y,W,B,a,z,alpha):
    """
    REQUIRES NUMPY AS NP

    Executes backward propagation
    for a dense neural network,
    updating W,B in-place
    (output activation is linear,
    hidden activations are ReLU)

    Args:
        X : Training examples (2D NumPy array)
        y : Training values (2D NumPy array)
        W : Weights list (List of 2D NumPy arrays)
        B : Bias list (List of 2D NumPy arrays)
        a : Activation list (List of 2D NumPy arrays)
        z : Pre-activation list (List of 2D NumPy arrays)
        alpha : Learning rate (int/float)

    Returns:
        W : Weights list (List of 2D NumPy arrays)
        B : Bias list (List of 2D NumPy arrays) 
    """
    num_layers = len(W)

    prev_delta = None

    m = y.shape[0]
    for i in range(num_layers - 1, -1, -1):
        if (i == num_layers - 1):
            delta = a[i] - y
            dJ_dW = np.matmul(a[i-1].T, delta * (1/m))
            dJ_dB = np.mean(delta, keepdims = True, axis = 0)

            W[i] -= dJ_dW * alpha
            B[i] -= dJ_dB * alpha

            prev_delta = delta

        elif (i != 0):
            delta = np.matmul(prev_delta, W[i+1].T) * (z[i] > 0).astype(float)
            dJ_dW = np.matmul(a[i-1].T, delta * (1/m))
            dJ_dB = np.mean(delta, keepdims = True, axis = 0)

            W[i] -= dJ_dW * alpha
            B[i] -= dJ_dB * alpha

            prev_delta = delta

        else:
            delta = np.matmul(prev_delta, W[i+1].T) * (z[i] > 0).astype(float)
            dJ_dW = np.matmul(X.T, delta * (1/m))
            dJ_dB = np.mean(delta, keepdims = True, axis = 0)

            W[i] -= dJ_dW * alpha
            B[i] -= dJ_dB * alpha

    return W, B

def train_neural_network_multiclass(X,y,layer_list,epochs,alpha,batch_size):
    """
    REQUIRES NUMPY AS NP
    
    Trains a neural network via categorical
    cross-entropy loss

    The parameters W, B are initialized using
    He initialization

    Args:
        X : Training examples (2D NumPy array)
        y : Training values (2D NumPy array)
        layer_list : 
        epochs : Number of epochs desired (int)
        alpha : Learning rate (int/float)
        batch_size : Desired batch size (int)

    Returns:
        W : Weights list (List of 2D NumPy arrays)
        B : Bias list (List of 2D NumPy arrays)
    """
    W = []
    B = []

    input_dim = X.shape[1]
    full_layer_list = [input_dim] + layer_list
  
    for i in range(len(layer_list)):
        w = np.random.randn(full_layer_list[i], full_layer_list[i + 1]) * np.sqrt(2. / full_layer_list[i])
        b = np.zeros((1, full_layer_list[i + 1]))
        W.append(w)
        B.append(b)

    m = X.shape[0]
    num_batches = int(np.ceil(m / batch_size))

    for epoch in range(epochs):
        permutation_array = np.random.permutation(m)
        X_shuffled = X[permutation_array]
        y_shuffled = y[permutation_array]

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, m)
            
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            z, a = forward_prop_multiclass(X_batch,W,B)
            W, B = backward_prop_multiclass(X_batch,y_batch,W,B,a,z,alpha)

        if (epoch % 10 == 0):
            print(f"Status: Epoch {epoch} of {epochs}")
        if (epoch == epochs - 1):
            print("Model training is complete.")

    return W, B

def multi_class_predict(X,W,B):
    """
    REQUIRES NUMPY AS NP

    Classifies input data based on
    trained weight and bias matrices

    Args:
        X : Training examples (2D NumPy array)
        W : Weights list (List of 2D NumPy arrays)
        B : Bias list (List of 2D NumPy arrays)

    Returns:
        yhat : Predicted training values (2D NumPy array)
    """
    z, a = forward_prop_multiclass(X,W,B)
    yhat = a[-1]

    return yhat

def multi_class_accuracy(y,yhat):
    """
    REQUIRES NUMPY AS NP
    
    Calculates the accuracy (%)
    of a multiclass NN

    Args:
        y : Training values (2D NumPy array)
        yhat : Predicted training values (2D NumPy array)
    
    Returns:
        per_correct : Accuracy of the model (float, 0.0-1.0 inclusive)
    """
    yhat_classes = np.argmax(yhat, axis=1)
    y_true_classes = np.argmax(y, axis=1)

    correct_predictions = np.sum(yhat_classes == y_true_classes)

    percent_correct = correct_predictions / y.shape[0]

    return percent_correct

def train_neural_network_multiclass_w_tuning(X,y,layer_list,epochs,alpha,batch_size,cv_data,cv_target):
    """
    REQUIRES NUMPY AS NP
    
    Trains a neural network via categorical
    cross-entropy loss

    The parameters W, B are initialized using
    He initialization

    Args:
        X : Training examples (2D NumPy array)
        y : Training values (2D NumPy array)
        layer_list : 
        epochs : Number of epochs desired (int)
        alpha : Learning rate (int/float)
        batch_size : Desired batch size (int)

    Returns:
        W : Weights list (List of 2D NumPy arrays)
        B : Bias list (List of 2D NumPy arrays)
    """
    W = []
    B = []

    input_dim = X.shape[1]
    full_layer_list = [input_dim] + layer_list
  
    for i in range(len(layer_list)):
        w = np.random.randn(full_layer_list[i], full_layer_list[i + 1]) * np.sqrt(2. / full_layer_list[i])
        b = np.zeros((1, full_layer_list[i + 1]))
        W.append(w)
        B.append(b)

    m = X.shape[0]
    num_batches = int(np.ceil(m / batch_size))

    initial_alpha = alpha

    for epoch in range(epochs):
        permutation_array = np.random.permutation(m)
        X_shuffled = X[permutation_array]
        y_shuffled = y[permutation_array]

        alpha = initial_alpha * (0.97 ** (epoch // 8))

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, m)
            
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            z, a = forward_prop_multiclass(X_batch,W,B)
            W, B = backward_prop_multiclass(X_batch,y_batch,W,B,a,z,alpha)

        if (epoch % 20 == 0):
            print(f"Status: Epoch {epoch} of {epochs}")
            val_preds = multi_class_predict(cv_data, W, B)
            val_acc = multi_class_accuracy(cv_target, val_preds)
            print(f"CV Accuracy at Epoch {epoch}: {val_acc:.4f}")
        if (epoch == epochs - 1):
            print("Model training is complete.")

    return W, B

def get_image_idx(y, yhat, correct=True):
    """
    REQUIRES NUMPY AS NP
    
    Obtains the index of a
    random image

    Args:
        y - Actual values (2D NumPy array)
        yhat - Predicted values (2D NumPy array)
        correct - Whether a correct index is desired; default is True
        
    Returns:
        rand_idx - Random index
    """
    if type(correct) != bool:
        raise ValueError("\"correct\" must be a boolean.")
    if len(y.shape) != 2 or len(yhat.shape) != 2:
        raise ValueError("\"y\" and \"yhat\" must be 2D NumPy arrays.")
    if y.shape != yhat.shape:
        raise ValueError("\"y\" and \"yhat\" must have the same shape.")

    m = y.shape[0]

    while True:
        rand_idx = np.random.randint(0, m)

        prediction = np.argmax(yhat[rand_idx])
        actual = np.argmax(y[rand_idx])

        if correct:
            if prediction == actual:
                return rand_idx, prediction, actual
        else:
            if prediction != actual:
                return rand_idx, prediction, actual
