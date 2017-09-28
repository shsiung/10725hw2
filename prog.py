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
    
def plotting(x,y,lambdas,title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Epoch')
    for i in range(len(lambdas)):
        plt.plot(x,y[i], linewidth = 2, label = 'Lambda = ' + str(lambdas[i]))
    plt.legend()

def main():

    print "Reading Files ..."
    train_X = np.array(readFile("train_features.csv")).astype('double');
    train_Y = np.array(readFile("train_labels.csv")).astype('double');
    test_X = np.array(readFile("test_features.csv")).astype('double');
    test_Y = np.array(readFile("test_labels.csv")).astype('double');

    C = max(train_Y);
    n = len(train_X);       # number of data
    m = train_X.shape[1];   # size of feature

    learning_rate = 0.001;
    lambda_all = [0.1, 1.0, 30.0, 50.0];
    epoch = 200;
    batch_size = 5000;

    obj_value = []
    train_acc = []
    test_acc = []

    for lambda_ in lambda_all:
        w = np.zeros(shape=(m,C));
        mini_train_X = train_X.copy()
        mini_train_Y = train_Y.copy()

        obj_value_lambda = []
        train_acc_lambda = []
        test_acc_lambda = []

        for k in range(epoch):
            random_ind_total = np.random.permutation(n)
            g = 0
            for batch in np.reshape(random_ind_total,(n/batch_size,batch_size)):
                batch_X = mini_train_X[batch]
                batch_Y = mini_train_Y[batch]
                for j in range(C):
                    param_grad = evaluate_gradient(w.transpose(),batch_X,batch_Y,C,lambda_,j)
                    w.transpose()[j] = w.transpose()[j] - learning_rate * param_grad

            obj_value_lambda += [objective_value(w.transpose(), train_X, train_Y, C, lambda_)]
            test_acc_lambda += [acc(pred(w,test_X),test_Y)]
            train_acc_lambda += [acc(pred(w,train_X),train_Y)]
            print "-> Lambda:" + str(lambda_) + " Epoch: " + str(k+1) + " ======== Obj value: " + str(obj_value_lambda[-1])+\
                " Train acc: " + str(train_acc_lambda[-1]) + " Test acc: " + str(test_acc_lambda[-1])

        obj_value += [obj_value_lambda]
        test_acc += [test_acc_lambda]
        train_acc += [train_acc_lambda]
        
    print "Plotting ..."
    epoch_plt = [x+1 for x in range(epoch)]
    for i in range(len(obj_value)):
        plt.figure()
        testacc = plt.plot(epoch_plt, test_acc[i],linewidth = 2,label='Test Accuracy')
        trainacc = plt.plot(epoch_plt, train_acc[i],linewidth = 2,label='Train Accuracy')
        plt.legend()
        plt.title('Lambda '+str(lambda_all[i]))

    plotting(epoch_plt, obj_value, lambda_all, 'Objective value vs. lambda')
    plotting(epoch_plt, test_acc, lambda_all, 'Test accuracy vs. lambda')
    plotting(epoch_plt, train_acc, lambda_all, 'Train accuracy vs. lambda')
    plotting(epoch_plt, obj_value[0:-1], lambda_all[0:-1], 'Objective value vs. lambda')
    plt.show()

if __name__ == '__main__':
    main()

