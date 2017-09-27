import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import mixture

def readFile(filename):
    vectors = []
    data = open(filename).read().splitlines()
    for line in data:
        line = line.split(',')
        vectors += [line[0:len(line)]]
    return vectors

def loss_function(w,x,y,C,lambda_):
    m = x.shape[1]-1;
    loss_1 = 0;
    loss_2 = 0;

    for j in range(C):
        loss_1 += np.dot(w.transpose()[j],w.transpose()[j])
    loss_1 *= 1/2

    for i in range(len(x)):
        for j in range(C):
            if y[i] == j+1:
                loss_2 += max(0,1-(np.dot(w.transpose()[j], x[i][0:m])+x[i][-1]))
            else:
                loss_2 += max(0,1+(np.dot(w.transpose()[j], x[i][0:m])+x[i][-1]))
    loss_2 *= lambda_/m


    return loss_1 + loss_2

def evaluate_gradient(w,x,y,C,lambda_):
    m = x.shape[1]-1;
    grad_1 = np.zeros(shape=(m))
    grad_2 = np.zeros(shape=(m))
    zeros = np.zeros(shape=(m)).tolist()
    ones = np.ones(shape=(m)).tolist()
    for j in range(C):
        grad_1 += w[j]

    for i in range(len(x)):
        for j in range(C):
            xi_list = x[i][0:m].tolist()
            if y[i]==j+1:
                grad_2+=np.multiply(np.asarray(max( np.zeros(shape=(m)).tolist() , [a_i-b_i for a_i, b_i in zip(ones,xi_list)] ))
                                              ,max(0,1-(np.dot(w[j], x[i][0:m])+x[i][-1])))
            else:
                grad_2+=np.multiply(np.asarray(max( np.zeros(shape=(m)).tolist() , [a_i+b_i for a_i, b_i in zip(ones,xi_list)] ))
                                              ,max(0,1+(np.dot(w[j], x[i][0:m])+x[i][-1])))

    grad_2 *= 2*lambda_/m

    return grad_1+grad_2

def main():
    temp_X = readFile("train_features.csv");
    temp_Y = readFile("train_labels.csv");
    train_X = np.array(temp_X).astype('double');
    train_Y = np.array(temp_Y).astype('double');
    C = max(train_Y);

    n = len(train_X)        # number of data
    m = train_X.shape[1]-1; # size of feature
    learning_rate = 0.005;
    lambda_ = 0.1;
    epoch = 20;
    w = np.zeros(shape=(m,C));
    print loss_function(w,train_X,train_Y,C,lambda_)/m

    batch_X = train_X[0:10];
    batch_Y = train_Y[0:10];

    for k in range(epoch):
        for j in range(C):
            param_grad = evaluate_gradient(w.transpose(),batch_X,batch_Y,C, lambda_)
            w.transpose()[j] = w.transpose()[j] - learning_rate * param_grad

        print loss_function(w,train_X,train_Y,C,lambda_)/m

if __name__ == '__main__':
    main()
    



