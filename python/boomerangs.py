import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import random
import time


def relu(Z):
    return np.maximum(0, Z)


def relu_deriv(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def expit_deriv(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    return dA * s * (1 - s)


def initialize(dims, method="none"):
    factors = {"none": 1}
    params = {}
    for l in range(1, len(dims)):
        factors["he"] = np.sqrt(2 / dims[l-1])
        factors["xavier"] = np.sqrt(1 / dims[l-1])
        
        W = np.random.randn(dims[l], dims[l-1])
        params["W" + str(l)] = np.multiply(W, factors[method])
        params["b" + str(l)] = np.zeros((dims[l], 1))
        
    return params


def forward_prop(X, params, activation="relu"):
    activations = {
        "relu": relu,
        "sigmoid": expit
    }
    layers = len(params) // 2
    caches = []
    A = X
    for l in range(1, layers):
        W = params["W" + str(l)]
        b = params["b" + str(l)]
        Z = np.dot(W, A) + b
        
        caches.append({
            "W": W,
            "A": A,
            "Z": Z
        })
        A = activations[activation](Z)

    W = params["W" + str(l+1)]
    b = params["b" + str(l+1)]
    Z = np.dot(W, A) + b
    caches.append({
        "W": W,
        "A": A,
        "Z": Z
    })
    A = expit(Z)
    return A, caches


def compute_cost(Y, A, params=None, method="none", lambd=0):
    extra_terms = {
        "none": 0,
        "l2": 0 
    }
    m = Y.shape[1]
    if method == "l2":
        layers = len(params) // 2
        W = [params["W" + str(l)] for l in range(1, layers + 1)]
        W = [np.sum(np.square(w)) for w in W]
        W = np.sum(W)
        extra_terms["l2"] = (lambd / (2 * m)) * W
    first_term = np.multiply(np.log(A), Y)
    second_term = np.multiply(np.log(1 - A), 1 - Y)
    cost = ((-1 / m) * np.nansum(first_term + second_term)) + extra_terms[method]
    return cost


def back_prop(A, caches, Y, activation="relu", method="none", lambd=0):
    extra_terms = {
        "none": 0,
        "l2": lambd / Y.shape[1]
    }
    derivs = {
        "relu": relu_deriv,
        "sigmoid": expit_deriv
    }
    m = Y.shape[1]
    layers = len(caches)
    activ = "sigmoid"
    grads = {}
    dA = np.divide(1 - Y,  1 - A) - np.divide(Y, A)
    for l in reversed(range(1, layers + 1)):
        cache = caches.pop()
        W = cache["W"]
        Z = cache["Z"]
        A = cache["A"]
        dZ = derivs[activ](dA, Z)
        activ = activation
        
        dW = (1 / m) * np.dot(dZ, A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dW += extra_terms[method] * W
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
        
        dA = np.dot(W.T, dZ)
    
    return grads


def optimize(params, grads, learn_rate):
    layers = len(params) // 2
    for l in range(1, layers + 1):
        params["W" + str(l)] -= learn_rate * grads["dW" + str(l)]
        params["b" + str(l)] -= learn_rate * grads["db" + str(l)]
    return params


def predict(X, params):
    A, cache = forward_prop(X, params)
    return (A > 0.5).astype(int)


def model(X, Y, dims, num_iterations=15000,
          learn_rate=0.00001, method="none", lambd=0):
    params = initialize(dims, "he")
    costs = []
    for i in range(num_iterations):
        A, caches = forward_prop(X, params)
        cost = compute_cost(Y, A, params=params, method=method, lambd=lambd)
        if i % 100 == 0:
            costs.append(cost)
        grads = back_prop(A, caches, Y, method=method, lambd=lambd)
        params = optimize(params, grads, learn_rate)
        
        print(f"Iteration {i} completed", end="\r")
        
    return params, costs

def count_boomerangs_validate(lst):
    return lst[0]==lst[-1]!=lst[1]


x_train = np.random.randint(low=0, high=10, size=(3, 10_000))
x_test = np.random.randint(low=0, high=10, size=(3, 2_000))

x_train = x_train / 10
x_test = x_test / 10

y_train = np.array([
    count_boomerangs_validate(x_train.T[i]) for i in range(x_train.shape[1])
    ]).reshape((1, x_train.shape[1]))
y_test = np.array([
    count_boomerangs_validate(x_test.T[i]) for i in range(x_test.shape[1])
    ]).reshape((1, x_test.shape[1]))

dims = [3, 15, 12, 1]

params, costs = model(x_train,
                      y_train,
                      dims,
                      10000,
                      0.2,
                      "none",
                      .2)

#plt.figure()
#plt.xlabel("Iteration in 100s")
#plt.ylabel("Cost")
#plt.plot(costs)
#plt.show()

train_predictions = predict(x_train, params)
test_predictions = predict(x_test, params)

train_correct = (train_predictions == y_train)
test_correct = (test_predictions == y_test)

train_acc = (np.sum(train_correct) / x_train.shape[1]) * 100
test_acc = (np.sum(test_correct) / x_test.shape[1]) * 100

print(f"Training data accuracy: {train_acc}%")
print(f"Testing data accuracy : {test_acc}%")

def count_boomerangs(lst):
    global params
    lst = (np.array(lst) / 10).reshape(1, len(lst))
    potential_booms = []
    for i in range(2, len(lst[0])):
        potential_booms.append(lst[0][i-2:i+1].reshape(3, 1))
    booms = sum(predict(potential_boom, params) for potential_boom in potential_booms)
    return int(booms)
