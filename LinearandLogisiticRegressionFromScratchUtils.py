"""
LinearandLogisiticRegressionFromScratchUtils.py

This file contains various functions relating linear
and logistic regression:
- Cost functions
- Gradient descent
- Feature scaling
- Prediction functions

Author: James Milgram
"""
import numpy as np
import matplotlib.pyplot as plt
from math import log, exp

def get_j_multiple_regression(X,y,w,b):
    """
    Calculates the cost for
    a given w vector, b 
    X : 2D array (matrix),
    y : 1D array (vector),
    w : 1D array (vector),
    b : scalar
    RETURNS A FLOAT 
    """
    f = np.dot(X,w) + b
    cost = 0.
    for f_val, y_val in zip(f, y):
        cost += (f_val-y_val) ** 2
    cost /= (2*X.shape[0])
    return cost

def sigmoid(z):
    """
    Takes a scalar, nd.array
    and transforms it via
    the sigmoid function
    z : scalar, nd.array
    REQUIRES NUMPY AS NP
    RETURNS AN ARRAY
    """
    return 1/(1+np.exp(-z))

def logistic_cost(w, b, x, y):
    """
    Computes the logistic cost for
    a given model v.s. a training example
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    REQUIRES NUMPY AS NP,
    FROM MATH IMPORT LOG, EXP
    RETURNS A FLOAT
    """
    if not w.shape[0] == x.shape[1] or not x.shape[0] == y.shape[0]:
        raise ValueError("Data set size error!")
    l = 0
    m = y.shape[0]
    for i in range(m):
        z = np.dot(w, x[i]) + b
        g = 1/(1+exp(-z))
        g = max(min(g, 1 - 1e-15), 1e-15)
        y_i = y[i]

        l += -y_i * log(g) - (1-y_i) * log(1-g)
    
    l /= m
    
    return l

def find_dj_db_logistic(w, b, x, y):
    """
    Finds dj_db
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    REQUIRES NUMPY AS NP
    RETURNS A FLOAT
    """
    m = x.shape[0]
    dj_db = 0
    
    for row in range(m):
        z = np.dot(w,x[row]) + b
        g = 1/(1+exp(-z))
        g -= y[row]
        dj_db += g
    
    dj_db /= m
    
    return dj_db

def find_dj_dw_logistic(w, b, x, y):
    """
    Finds dj_dw
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    REQUIRES NUMPY AS NP
    RETURNS AN ND ARRAY
    """
    m = x.shape[0]
    dj_dw = np.zeros_like(w)

    for row in range(m):
        z = np.dot(w,x[row]) + b
        g = 1/(1+exp(-z))
        g -= y[row]
        g *= x[row]
        dj_dw += g

    dj_dw /= m

    return dj_dw

def gradient_descent_logistic(w, b, x, y, iters, a):
    """
    Uses gradient descent to determine
    parameters w, b that yield the lowest
    cost J(w,b); rescales via the feature array
    for improved model training using Z-Score
    Normalization
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    iters : # of iterations (int)
    a : Learning Rate (float)
    REQUIRES NUMPY AS NP,
    FROM MATH IMPORT EXP
    RETURNS w (ND Array), b (float)
    """
    if not w.shape[0] == x.shape[1] or not x.shape[0] == y.shape[0]:
        raise ValueError("Data set size error!")
    if not type(iters) == int or not iters > 0:
        raise ValueError("Number of iterations is invalid!")
    if not type(a) == float or not a > 0:
        raise ValueError("Learning rate is invalid!")
    
    x = (x-x.mean(axis=0))/x.std(axis=0)
    
    for i in range(iters):
        dj_db = find_dj_db_logistic(w, b, x, y)
        dj_dw = find_dj_dw_logistic(w, b, x, y)
        b -= a * dj_db
        w -= a * dj_dw
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {logistic_cost(w, b, x, y):.4f}")
    
    print(f"The set of parameters (x was rescaled by Z-Score Normalization) are: (w: {list(w)}, b:{b})")
    
    return w, b

def predict(x, w, b):
    """
    Predicts the category
    for binary classifcation.
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    RETURNS A 1D ARRAY
    """ 
    preds = []
    x = (x-x.mean(axis=0))/x.std(axis=0)
    for i in range(x.shape[0]):
        z = np.dot(w, x[i]) + b
        g = 1 / (1 + exp(-z))
        preds.append(1 if g >= 0.5 else 0)
    return np.array(preds)

def find_dj_db_linear_regression(w, b, x, y):
    """
    Finds dj_db
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    iters : # of iterations (int)
    a : Learning Rate (float)
    REQUIRES NUMPY AS NP
    RETURNS A FLOAT
    """
    m = x.shape[0]
    dj_db = 0
    
    for row in range(m):
        f = np.dot(w,x[row]) + b
        dj_db += (f - y[row])

    dj_db /= m

    return dj_db

def find_dj_dw_linear_regression(w, b, x, y, reg_p):
    """
    Finds dj_dw
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    reg_p : Regularization Parameter (float)
    REQUIRES NUMPY AS NP
    RETURNS A FLOAT
    """
    m = x.shape[0]
    dj_dw = np.zeros_like(w)

    for row in range(m):
        f = np.dot(w, x[row]) + b
        f -= y[row]
        f *= x[row]
        dj_dw += f
        
    dj_dw /= m
    dj_dw += (reg_p * w)

    return dj_dw

def gradient_descent_linear_regression_with_regularization(w, b, x, y, iters, a, reg_p):
    """
    Uses gradient descent to determine
    parameters w, b that yield the lowest
    cost J(w,b); rescales via the feature array
    for improved model training using Z-Score
    Normalization
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    iters : # of iterations (int)
    a : Learning Rate (float)
    reg_p : Regularization Parameter (float)
    REQUIRES NUMPY AS NP,
    FROM MATH IMPORT EXP
    RETURNS w (ND Array), b (float)
    """
    if not w.shape[0] == x.shape[1] or not x.shape[0] == y.shape[0]:
        raise ValueError("Data set size error!")
    if not type(iters) == int or not iters > 0:
        raise ValueError("Number of iterations is invalid!")
    if not type(a) == float or not a > 0:
        raise ValueError("Learning rate is invalid!")
    if not type(reg_p) == float or not reg_p >= 0:
        raise ValueError("Regularization parameter is invalid!")
        
    x = (x-x.mean(axis=0))/x.std(axis=0)

    for i in range(iters):
        dj_db = find_dj_db_linear_regression(w, b, x, y)
        dj_dw = find_dj_dw_linear_regression(w, b, x, y, reg_p)
        b -= a * dj_db
        w -= a * dj_dw
    
    print(f"The set of parameters are (scaled using Z-Score Normalization): (w: {list(w)}, b:{b})")
    
    return w, b

def predict_linear_regression_with_regularization(w, b, x, y):
    """
    Computes the predicted
    value (y-hat) for given input
    values (x)
    w : Weights (ND Array)
    b : Bias (float)
    x : Inputs (ND Array)
    y : Outputs (1D Array)
    REQUIRES NUMPY AS NP
    RETURNS A 2D ARRAY
    """
    x = (x-x.mean(axis=0))/x.std(axis=0)
    m = x.shape[0]
    matrix = np.empty((0,2))
    
    for i in range(m):
        y_hat = np.dot(w, x[i]) + b
        y_actual = y[i]
        array = np.array([[y_hat, y_actual]])
        matrix = np.vstack((matrix, array))

    print("Predicted v.s. Actual")
    print(matrix)
    
    return matrix

