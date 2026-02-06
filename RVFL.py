import numpy as np
import time


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    def y(x):
        return (scale * x) * (x > 0) + (scale * alpha * (np.exp(x) - 1)) * (x <= 0)
    
    result = y(x)
    return result

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def hardlim(x):
    return (np.sign(x) + 1) / 2

def tribas(x):
    return np.maximum(1 - np.abs(x), 0)

def radbas(x):
    return np.exp(-(x**2))

def leaky_relu(x):
    x[x >= 0] = x[x >= 0]
    x[x < 0] = x[x < 0] / 10.0
    return x

def Activation_fun(X1,activation):
    if activation == 1:
        Z = selu(X1)
    elif activation == 2:
        Z = relu(X1)
    elif activation == 3:
        Z = sigmoid(X1)
    elif activation == 4:
        Z = np.sin(X1)
    elif activation == 5:
        Z = hardlim(X1)
    elif activation == 6:
        Z = tribas(X1)
    elif activation == 7:
        Z = radbas(X1)
    elif activation == 8:
        Z = np.sign(X1)
    elif activation == 9:
        Z = leaky_relu(X1)
    elif activation == 10:
        Z = np.tanh(X1)
    return Z

def Evaluate(ACTUAL, PREDICTED):
    idx = (ACTUAL == 1)
    p = np.sum(idx)
    n = np.sum(~idx)
    N = p + n
    tp = np.sum(np.logical_and(ACTUAL[idx] == 1, PREDICTED[idx] == 1))
    tn = np.sum(np.logical_and(ACTUAL[~idx] == 0, PREDICTED[~idx] == 0))
    fp = n - tn
    fn = p - tp
    tp_rate = tp / p if p != 0 else 0
    tn_rate = tn / n if n != 0 else 0
    accuracy = 100 * (tp + tn) / N
    sensitivity = 100 * tp_rate
    specificity = 100 * tn_rate
    precision = 100 * tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = sensitivity
    f_measure = 2 * ((precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) != 0 else 0
    gmean = 100 * np.sqrt(tp_rate * tn_rate)
    
    EVAL = [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean, tp, tn, fp, fn]
    return EVAL


def RVFL_train(train_data1,train_data2, test_data1,test_data2, c1,c2,c3, rho,N,activation):
    # np.random.seed(2)
    

    start = time.time()
    trainX=train_data1[:,:-1]
    dataY_train_temp=train_data1[:,-1]

    trainX2=train_data2[:,:-1]


    s = 0.1
    
    Nsample, Nfea = trainX.shape
    Nsample2, Nfea2 = trainX2.shape

    #View 1
    W = np.random.rand(Nfea, N) * 2 - 1
    b = s * np.random.rand(1, N)
    X1 = np.dot(trainX, W) + np.tile(b, (Nsample, 1))
    X1=Activation_fun(X1,activation)
    X = np.concatenate((trainX, X1), axis=1)
    X_A = np.hstack((X, np.ones((Nsample, 1))))  # Bias in the output layer
    
    
    #View 2
        
    Nsample2, Nfea2 = trainX2.shape  
    W2 = np.random.rand(Nfea2, N) * 2 - 1
    b2 = s * np.random.rand(1, N)
    X1 = np.dot(trainX2, W2) + np.tile(b2, (Nsample2, 1))
    X1=Activation_fun(X1,activation)
    X = np.concatenate((trainX2, X1), axis=1)
    X_B = np.hstack((X, np.ones((Nsample2, 1))))  # Bias in the output layer
    
    X_mat_A = np.concatenate((np.eye(X_A.shape[1])+c1*np.dot(X_A.T, X_A), rho*np.dot(X_A.T, X_B)), axis=1)
    X_mat_B = np.concatenate((rho*np.dot(X_B.T, X_A), c3*np.eye(X_B.shape[1])+c2*np.dot(X_B.T, X_B)), axis=1)
    X_mat = np.vstack((X_mat_A, X_mat_B))

    X_mat_Rhs= np.dot(np.vstack((X_A.T*(c1+rho), X_B.T*(c2+rho))), dataY_train_temp)

    beta = np.dot(np.linalg.inv(X_mat), X_mat_Rhs)
    hl=Nfea+N+1
    beta1 = beta[:hl]
    beta2 = beta[hl:]

    end = time.time()
    # Test_A

    T_A=test_data1[:,:-1]
    Y=test_data1[:,-1]
    
    Nsample = T_A.shape[0]

    # Test Data

    X1 = np.dot(T_A, W) + np.tile(b, (Nsample, 1))
    X1=Activation_fun(X1,activation)

    X1 = np.hstack((X1, np.ones((Nsample, 1))))
    XZ1 = np.hstack((T_A, X1))
    rawScore1 = np.dot(XZ1, beta1)

    #Test_B
    T_B=test_data2[:,:-1]
    
    Nsample = T_B.shape[0]

    # Test Data

    X1 = np.dot(T_B, W2) + np.tile(b2, (Nsample, 1))
    X1=Activation_fun(X1,activation)

    X1 = np.hstack((X1, np.ones((Nsample, 1))))
    XZ2 = np.hstack((T_B, X1))

    rawScore2 = np.dot(XZ2, beta2)

    rawScore3 = (rawScore1 + rawScore2) / 2

    final_pred = (np.sign(rawScore3) + 1) / 2

    Y = Y.reshape(-1, 1)
    if np.any(Y == -1):
        Y = (Y + 1) / 2
        
    EVAL_Validation = Evaluate(Y, final_pred.reshape(-1, 1))
    
    Time=end - start
    return EVAL_Validation,Time
