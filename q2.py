import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import svm
import math

def show_confusionMatrix(confusionMatrix):
    plt.imshow(confusionMatrix)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.set_cmap("Reds")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    plt.show()


def read_file(path, d1, d2, multi):
    data = np.array(pd.read_csv(path, header=None, dtype=float).values)
    # print(data.shape) (22500, 785)
    columns = data.shape[1]  # 785
    rows = len(data)
    if(multi == "0"):
        return1 = data[np.ix_((data[:, columns-1] == d1)
                              | (data[:, columns-1] == d2))]
        return2 = return1[:, columns-1:columns]
        for i in range(len(return1)):
            if(return2[i, 0] == d1):
                return2[i, 0] = 1
            else:
                return2[i, 0] = -1

        return1 = return1/255
        return (np.asmatrix(return1[:, 0:columns-1]), np.asmatrix(return2))

        # for i in range(rows):
        #         np.append(return1, data[i], axis=0)
        #         np.append(return2, [1], axis=0)
        #     # if(data[i,columns-1]==d1):
        #     #     np.append(return1, data[i], axis=0)
        #     #     np.append(return2, [1], axis=0)
        #     # elif(data[i,columns-1]==d2):
        #     #     np.append(return1, data[i], axis=0)
        #     #     np.append(return2, [-1], axis=0)

        # return1 = return1/255
        # print(return1)
        # print(return2)
        # # return (np.asmatrix(return1[:,0:columns-1]),np.asmatrix(return2))

    else:
        return1 = data
        return2 = np.array(data[:, columns-1:columns])
        return1 = return1/255
        return (np.asmatrix(return1[:, 0:columns-1]), np.asmatrix(return2))


def gaussianSolver(trIn, trOut, teIn, teOut, gamma, C):
    m = trIn.shape[0]
    n = trIn.shape[1]
    # making the kernel matrix as per gaussian function
    k = np.asmatrix(np.zeros((m, m), dtype=float))
    x = np.dot(trIn, trIn.transpose())
    for a in range(m):
        for b in range(m):
            k[a, b] = float(x[a, a]+x[b, b] - 2*x[a, b])
    k = np.exp(gamma*k*(-1))

    temp = np.dot(trOut, trOut.transpose())
    temp2 = np.multiply(k, temp)
    P = matrix(temp2)
    temp3 = -1*np.ones((m, 1))
    q = matrix(temp3)
    temp4 = trOut.transpose()
    A = matrix(temp4)
    b = matrix(0.0)
    temp5 = np.identity(m)
    temp6 = -1*np.identity(m)
    G = matrix(np.vstack((temp5, temp6)))
    temp7 = C*np.ones((m, 1))
    temp8 = np.zeros((m, 1))
    h = matrix(np.vstack((temp7, temp8)))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    simplified = np.ravel(sol['x'])
    supportV = 0
    epsilon = 1e-4
    gamma_lb = np.asmatrix(np.zeros((len(simplified), 1), dtype=float))
    for i in range(len(simplified)):
        if simplified[i] > epsilon:
            gamma_lb[i, 0] = trOut[i, 0]*simplified[i]
            supportV = supportV + 1
    print("Number of support vectors = "+str(supportV))
    l_params = np.arange(len(simplified))[simplified > epsilon]
    num_params = len(l_params)
    prediction = np.zeros((len(teIn), 1), dtype=int)

    if(num_params > 0):
        b = 0
        for index in l_params:
            b = b + (trOut[index, 0] - np.sum(np.multiply(gamma_lb, np.exp(-1*gamma *
                                                                           np.sum(np.multiply(trIn-trIn[index, :], trIn-trIn[index, :]), axis=1)))))
        b = b / (float(num_params))

        # making the prediction when numparams >0
        temp_tr = np.sum(np.multiply(trIn, trIn), axis=1)
        temp_te = np.sum(np.multiply(teIn, teIn), axis=1)
        temp_tt = np.dot(trIn, teIn.transpose())

        num_test = len(teIn)
        for i in range(num_test):
            prediction[i] = np.sign(np.sum(np.multiply(
                gamma_lb, np.exp(-1*gamma*(temp_tr - 2*temp_tt[:, i] + temp_te[i, 0]))))+b)

    return prediction

    
def partb1(tr_path,te_path,multi):
    C = 1
    gamma = 0.05
    (testI, testO) = read_file(te_path, 6, 7, multi)
    m = len(testI)
    #  here we make kC2 classes and make predictions for each.
    # and keep incrementing the value for that class in the prediction array...
    prediction_dict = {}
    for i in range(m):
        prediction_dict[i] = [0]*10
    prediction = np.zeros((m, 1), dtype=int)
    for i in range(10):
        for j in range(i):
            (trIn, trOut) = read_file(tr_path, i, j, "0")
            local_prediction = gaussianSolver(trIn, trOut, testI, testO, gamma, C)
            for z in range(m):
                if(local_prediction[z, 0] == 1):
                    prediction_dict[z][i] = prediction_dict[z][i]+1
                else:
                    prediction_dict[z][j] = prediction_dict[z][j]+1
            print("Done for "+str(i)+" "+str(j))
    for i in range(m):
        ans = 0
        maxval =  prediction_dict[0][0]
        for i2 in range(10):
            if(prediction_dict[0][i2]>maxval):
                maxval = prediction_dict[0][i2]
                ans = i2
        prediction[i] = 1 + ans

    # making the prediction here
    ans = 0
    for i in range(m):
        if(prediction[i] == testO[i]):
            ans = ans+1

    print(float(ans)/float(m))
    confusionMatrix = confusion_matrix(testO, prediction)
    print(confusionMatrix)
    show_confusionMatrix(confusionMatrix)

def partb2(tr_path,te_path,multi):
    # using sklearn
    (trIn, trOut) = read_file(tr_path, 6, 7, multi)
    (teIn, teOut) = read_file(te_path, 6, 7, multi)
    # clf = OneVsRestClassifier(SVC()).fit(trIn, trOut)
    # prediction = clf.predict(teIn)
    # m = len(teIn)
    # ans = 0
    # for i in range(m):
    #     if(prediction[i]== teOut[i]):
    #         ans= ans+1
    # print("accuracy")
    # print(float(ans)/float(m))
    #             accuracy
    #                0.8668
    C = 1
    gamma = 0.05
    clf = svm.SVC(decision_function_shape='ovr',kernel='rbf', C=C, gamma=gamma)
    clf.fit(trIn, trOut)
    prediction = clf.predict(teIn)
    m = len(teIn)
    ans = 0
    for i in range(m):
        if(prediction[i] == teOut[i]):
            ans = ans+1
    print("accuracy")
    print(float(ans)/float(m))
    confusionMatrix = confusion_matrix(teOut, prediction)
    print  ( confusionMatrix )
    print("ok")
    show_confusionMatrix(confusionMatrix)
    # accuracy
    # 0.8676

def partb4(tr_path,te_path,multi,output):
    (trIn, trOut) = read_file(tr_path, 6, 7, multi)
    (teIn, teOut) = read_file(te_path, 6, 7, multi)
    m = len(trIn)
    n = int(9*m/10)
    train_X = trIn[0:n, :]
    train_Y = trOut[0:n, :]
    valid_X = trIn[n:m, :]
    valid_Y = trOut[n:m, :]
    gamma = 0.05
    C_arr = [0.001, 1, 10]
    num_C = len(C_arr)
    v_acc = [0.0]*num_C
    t_acc = [0.0]*num_C
    store = {}
    for i in range(num_C):
        C = C_arr[i]
        clf = svm.SVC(decision_function_shape='ovr',kernel='rbf', C=C, gamma=gamma)
        clf.fit(train_X, train_Y)
        valid_predict = clf.predict(valid_X)
        m = len(valid_X)
        vp = 0
        for j in range(m):
            if(valid_predict[j] == valid_Y[j]):
                vp = vp+1
        v_acc[i] = float(vp)/float(m)
        test_predict = clf.predict(teIn)
        store[i] = test_predict
        m = len(teIn)
        tp = 0
        for j in range(m):
            if(test_predict[j] == teOut[j]):
                tp = tp + 1
        t_acc[i] = float(tp)/float(m)

    print("Validation Set Accuracy")
    print(v_acc)
    print("Test set Accuracy")
    print(t_acc)
    # plotting
    plot_x = []
    plot_valid = []
    plot_test = []
    for i in range(num_C):
        plot_x.append(math.log(C_arr[i],10))
        plot_valid.append(v_acc[i]*100)
        plot_test.append(t_acc[i]*100)

    maxacc = t_acc[0]
    ans = 0
    for i in range(num_C):
        if(t_acc[i]>maxacc):
            maxacc =t_acc[i]
            ans =i
    print(ans)
    output_array = store[ans]
    np.savetxt(output, output_array, fmt="%d", delimiter="\n")
    print("Final accuracies are:")
    mark = 0 
    for i in range(len(teIn)):
        if(output_array[i]==teOut[i]):
            mark = mark+1
    print("accuracy is : "+str(mark/len(teIn)))
    # plt.plot(plot_x, plot_valid, label='valid accuracy', marker='.')
    # plt.plot(plot_x, plot_test, label='test accuracy', marker='.')
    # plt.legend()
    # plt.xticks([-3,-2,-1,0,1])
    # plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
    # plt.title('Test and Validation Accuracy vs C')
    # plt.ylabel('Accuracy')
    # plt.xlabel('log(C)')
    # plt.show()


def parta1(tr_path,te_path,multi):
    d1 = 6
    d2 = 7
    (trIn, trOut) = read_file(tr_path, d1, d2, multi)
    (teIn, teOut) = read_file(te_path, d1, d2, multi)
    C = 1
    #  number of training data
    m = len(trIn)
    x = np.multiply(trIn, trOut)
    temp1 = np.dot(x, x.transpose())
    P = matrix(temp1)
    temp2 = -1*np.ones((m, 1))
    q = matrix(temp2)
    temp3 = trOut.transpose()
    A = matrix(temp3)
    b = matrix(0.0)
    temp4 =-1*np.identity(m)
    temp5 = np.identity(m)
    G = matrix(np.vstack((temp4,temp5)))
    temp6=np.zeros((m, 1))
    temp7=C*np.ones((m, 1))
    h = matrix(np.vstack((temp6,temp7)))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    # weights for linear kernel
    supportV = 0
    m = trIn.shape[0]
    n = trIn.shape[1]
    # getting the values
    raveled = np.ravel(sol['x'])
    # print(raveled)
    epsilon = 1e-4
    l_params = np.arange(len(raveled))[raveled > epsilon]
    # print(l_params)
    w_mat = np.asmatrix(np.zeros((1, n), dtype=float))
    for i in l_params:
        for j in range(n):
            cal = raveled[i]*trIn[i, j]*trOut[i, 0]
            w_mat[0, j] = w_mat[0, j] + (cal)
        supportV = supportV + 1

    if (supportV>0):
        for i in l_params:
            t1 = trOut[i, 0]-np.dot(trIn[i, :], w_mat.transpose())[0, 0]
            b = b + t1
        b = b/(float(len(l_params)))
    # print(str(b) + " is the value of b")

    # print(str(supportV) + " support vectors")

    prediction = np.asmatrix(np.ones((len(teIn), 1), dtype=int))
    a = np.dot(teIn, w_mat.transpose())+b
    t2 = np.ones((len(teIn), 1))
    prediction = 2*np.multiply((a > 0), t2) - 1
    print("prediction accuracy")
    ans = 0
    for i in range(len(teIn)):
        if(teOut[i] == prediction[i]):
            ans += 1
    print(float(ans)/float(len(teIn)))
    confusionMatrix = confusion_matrix(teOut, prediction)
    print("Confusion Matrix")
    print(confusionMatrix)
    show_confusionMatrix(confusionMatrix)

def parta2(tr_path,te_path,multi):
    d1 = 6
    d2 = 7
    (trIn, trOut) = read_file(tr_path, d1, d2, multi)
    (teIn, teOut) = read_file(te_path, d1, d2, multi)
    C = 1
    gamma = 0.05
    prediction = gaussianSolver(trIn, trOut, teIn, teOut, gamma, C)
    print("prediction accuracy")
    ans = 0
    for i in range(len(teIn)):
        if(teOut[i] == prediction[i]):
            ans += 1
    print(float(ans)/float(len(teIn)))
    confusionMatrix = confusion_matrix(teOut, prediction)
    print("Confusioin Matrix:")
    print(confusionMatrix)
    show_confusionMatrix(confusionMatrix)

    # this part is over.

def main():
    tr_path = sys.argv[1]
    te_path = sys.argv[2]
    output = sys.argv[3]

    # un-comment which ever part you want to run

    partb4(tr_path,te_path,"1",output)
    
    # partb2(tr_path,te_path,"1")
    
    # partb1(tr_path,te_path,"1")
    
    # parta1(tr_path,te_path,"0")
    
    # parta2(tr_path,te_path,"0")


if __name__ == "__main__":
    main()
