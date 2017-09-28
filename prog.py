import numpy as np
import matplotlib.pyplot as plt
import csv

def readFile(filename):
    vectors = []
    data = open(filename).read().splitlines()
    for line in data:
        line = line.split(',')
        vectors += [line[0:len(line)]]
    return vectors

def objective_value(w,x,y,C,lambda_):
    m = x.shape[1];
    n = x.shape[0];
    loss_1 = 1.0/2*np.sum(w*w)
    loss_2 = 0;

    for i in range(n):
        for j in range(C):
            loss_2 += pow(loss_function_wj(w[j],x[i],y[i],j+1),2)
    loss_2 *= lambda_/n

    return loss_1 + loss_2

def loss_function_wj(w,x,y,j):
    if y == j:
        return np.maximum(0,1-np.dot(w,x))
    else:
        return np.maximum(0,1+np.dot(w,x))

def evaluate_gradient(w,x,y,C,lambda_,j_):
    m = x.shape[1];
    n = x.shape[0];
    zeros = np.zeros(shape=(m)).tolist()

    grad_1 = w[j_]
    grad_2 = np.zeros(shape=(m))
    grad_2_scale = 0;
    for i in range(n):
        grad_2_scale = loss_function_wj(w[j_],x[i],y[i],j_+1)

        if y[i] == j_+1:
            condition = np.dot(w[j_],x[i])
        else:
            condition = -np.dot(w[j_],x[i])
        if condition<=1:
            if y[i]==j_+1:
                grad_2 += grad_2_scale * np.array([ -x_i for x_i in x[i].tolist()])
            else:
                grad_2 += grad_2_scale * x[i]

    grad_2 *= 2.0*lambda_/n

    return grad_1+grad_2

def pred(w,x):
    return map(lambda y: np.argmax(w.transpose().dot(y))+1, x) 

def acc(y,y_true): 
    return np.sum(y==y_true.transpose())*1.0/len(y_true)*100.0

def main():
    temp_X = readFile("train_features.csv");
    temp_Y = readFile("train_labels.csv");
    train_X = np.array(temp_X).astype('double');
    train_Y = np.array(temp_Y).astype('int');

    C = max(train_Y);
    n = len(train_X);       # number of data
    m = train_X.shape[1];   # size of feature
    print "n: " + str(n);
    print "m: " + str(m);

    learning_rate = 0.005;
    lambda_ = 0.1;
    epoch = 200;
    batch_size = 5000;

    w = np.zeros(shape=(m,C));
    print "== Initial loss: " + str(objective_value(w.transpose(),train_X,train_Y,C,lambda_))
    mini_train_X = train_X.copy()
    mini_train_Y = train_Y.copy()

    for k in range(epoch):
        random_ind = np.random.permutation(batch_size)
        batch_X = mini_train_X[random_ind]
        batch_Y = mini_train_Y[random_ind]
        for j in range(C):
            param_grad = evaluate_gradient(w.transpose(),batch_X,batch_Y,C,lambda_,j)
            w.transpose()[j] = w.transpose()[j] - learning_rate * param_grad

        print " -> Epoch " + str(k) + " with loss " + str(objective_value(w.transpose(),train_X,train_Y,C,lambda_))
        pred_y = np.array(pred(w,train_X))
        print acc(pred_y,train_Y)

if __name__ == '__main__':
    main()
    



