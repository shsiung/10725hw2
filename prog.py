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

def loss(w,x,b,y,j):
    if y[0]==j:
        return max(0,1-(np.dot(w,x)+b))
    else:
        return max(0,1+(np.dot(w,x)+b))

def f(w,x,y,C,lambda_):
    loss_1 = 0;
    loss_2 = 0;
    m = x.shape[1];

    for j in range(C):
        loss_1 += np.dot(w[j],w[j])
    for i in range(len(x)):
        for j in range(C):
            loss_2 += loss((w.transpose())[j],(x[i])[0:m-1],x[i][-1],y[i],j+1)

    return 1/2*loss_1 + lambda_/m*loss_2 

def grad_f(w,xi,y,C,lambda_):
    n = len(xi);
    grad_1 = np.zeros(shape=(n))
    grad_2 = np.zeros(shape=(n))
    zeros = np.zeros(shape=(n)).tolist()
    ones = np.ones(shape=(n)).tolist()
    for j in range(C):
        grad_1 += w[j]
        xi_list = xi.tolist()
        if y==j+1:
            grad_2+=np.asarray(max( np.zeros(shape=(n)).tolist() ,  [a_i-b_i for a_i, b_i in zip(ones,xi_list)]))
        else:
            grad_2+=np.asarray(max( np.zeros(shape=(n)).tolist() ,  [a_i+b_i for a_i, b_i in zip(ones,xi_list)]))

    grad_2 *= lambda_/n

    return grad_1+grad_2


def main():
    temp_X = readFile("hw2_q4_dataset/train_features.csv");
    temp_Y = readFile("hw2_q4_dataset/train_labels.csv");
    X = np.array(temp_X).astype('double');
    Y = np.array(temp_Y).astype('double');
    C = max(Y);
    n = X.shape[1]-1;
    learning_rate = 0.005
    lambda_ = 0.1;

    w = np.zeros(shape=(n,C));
    print f(w,X,Y,C,lambda_)/n

    print w.shape

    for k in range(20):
        for j in range(C):
            for i in range(500):
                grad = grad_f(w.transpose(),X[i][0:n],Y[i],C, lambda_)
                w.transpose()[j] -= learning_rate*grad

if __name__ == '__main__':
    main()
    



