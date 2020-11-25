#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
data = load_breast_cancer()
X, t = load_breast_cancer(return_X_y=True)

t = np.matrix(t).T
X_train, X_test = train_test_split(X, train_size=3/4, test_size=1/4, random_state = 6048)
t_train, t_test = train_test_split(t, train_size=3/4, test_size=1/4, random_state = 6048)

sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])
print(X_train.shape, X_test.shape)


# In[2]:


#Separate the classes to 0 and 1
def seperateClass(x, t):    
    i0 = np.where(t == 0)[0]
    n = len(i0)
    x_0 = np.zeros((n,x.shape[1]))
    for i in range(n):
        x_0[i,:] = x[i0[i],:]
    
    i1 = np.where(t == 1)[0]
    n = len(i1)
    x_1 = np.zeros((n,x.shape[1]))
    for i in range(n):
        x_1[i,:] = x[i1[i],:] 
        
#     i2 = np.where(t == 2)[1]
#     n = len(i2)
#     x_2 = np.zeros((n,x.shape[1]))
#     for i in range(n):
#         x_2[i,:] = x[i2[i],:]
    return x_0, x_1
X_train_0,X_train_1 = seperateClass(X_train, t_train)


# In[3]:


#Implement logistic regression with batch gradient descent

import math
def BGD(x, t):
    N = x.shape[1]
    iterations = 300
    w = np.full((30,1),-10)
    alpha = 0.1
    cost = np.zeros(iterations)
    gr_norms = np.zeros(iterations)
    for i in range(iterations):
        z = np.dot(x,w)
        y = 1/(1 + np.exp(-z))
        diff = y-t
        gr = np.dot(x.T, np.transpose(diff.T))/N
        gr_norm_sq = np.dot(gr.T,gr)
        gr_norms[i] = gr_norm_sq
        
        w = w - alpha * gr
        cost[i] = 0
        for j in range(N):
            #print(t[j]*math.log10(1 + math.exp(-z[j])))
            cost[i] += t[j]*math.log10(1 + math.exp(-z[j]) + (1-t[j])*math.log10(math.exp(z[j])))
        cost[i] /= N
    return w,cost
w,cost = BGD(X_train, t_train)


# In[4]:


#Functions for adding dummy, find the model of the estimation and the misclassification rate
def dummy(x):
    dummy = np.ones(x.shape[0])
    x = np.insert(x, 0, dummy, axis = 1)
    return x

#Find the result of multiplication and its estimated classes (0 or 1) based on the results
def findEst(x, w):
    x1 = dummy(x)
    n = x1.shape[0]
    decision = np.zeros(n)
    decision = np.matrix(decision).T
    
    res = np.dot(x, w)
    for i in range(n):
        if(res[i][0] >= 0):
            decision[i][0] = 1
        else:
            decision[i][0] = 0
    return decision

#Find misclassficiation rate (error)
def misclass(res, t):
    n = t.shape[0]
    missed = 0
    for i in range(n):
        if(res[i] != t[i]):
            missed += 1
    return missed/n

result = findEst(X_train, w)
print("Training misclassification rate:",misclass(result,t_train))


# In[5]:


#Error, and the result of testing
def findError(y, t): #error = misclassification rate
    if(y.shape != t.shape):
        return "Dimensions of labels and estimations do not match"
        
    else:
        z = y - t
        error = np.count_nonzero(z)/y.shape[0]
        return error


def testing(x, w, t):
    x1 = dummy(x)
    n = x1.shape[0]
    decision = np.zeros(n)
    decision = np.matrix(decision).T
    
    res = np.dot(x, w)
    ans = 1/(1+np.exp(-res))

    for i in range(n):
        if(res[i][0] >= 0):
            decision[i][0] = 1
        else:
            decision[i][0] = 0
    num_1 = np.sum(decision!=0)
    print(num_1)
    return decision, ans, num_1 #the estimation, the result of logistic regression, and the number of ones


#Find the true pos, false pos and false neg given each threshold
def findPR(x, w, t):
    z = np.dot(x, w)
    z = np.asarray(z).reshape(-1) #turn a matrix to an array
    thres = z.copy()
    thres.sort()
    tplist = []
    fplist = []
    fnlist = []
    tnlist = []
    for i in range(z.shape[0]): #outer loop is each result obtained from X * w
        threshold = thres[i]
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for j in range(t.shape[0]): #Innter loop is to determine the true pos, false pos, fals neg given the theta
            if(z[j] >= threshold):
                if(t[j,0] == 1):
                    tp+=1
                elif(t[j,0] == 0):
                    fp += 1
            else:
                if(t[j,0] == 1):
                    fn += 1
                else:
                    tn += 1

        tplist.append(tp)
        fplist.append(fp)
        fnlist.append(fn)
    return tplist, fplist, fnlist

#Find f1 at threshold of 0
def findf1(x, w, t):
    z = np.dot(x, w)
    z = np.asarray(z).reshape(-1)
    thres = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for j in range(t.shape[0]):
        if(z[j] >= thres):
            if(t[j,0] == 1):
                tp+=1
            elif(t[j,0] == 0):
                fp += 1
        else:
            if(t[j,0] == 1):
                fn += 1
            else:
                tn += 1

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*precision * recall / (precision + recall)
    return f1


tp, fp, fn = findPR(X_test, w, t_test)

#Find f1 given the list of tp, fp, fn
def PR(tp, fp, fn):
    precision = []
    recall = []
    f1score = []
    for i in range(len(tp)):
        p = tp[i]/(tp[i]+fp[i])
        r = tp[i]/(tp[i] +fn[i])
        precision.append(p)
        recall.append(r)
        f1score.append(2*p*r/(p+r))
    return precision, recall, f1score


dec1, res, num1 = testing(X_test, w, t_test)
error1 = findError(dec1, t_test)
print("The experimental logistic regression error:", error1)
p, r, f = PR(tp, fp, fn)
f1 = findf1(X_test, w, t_test)
print("The experimental f1 score is:", f1)


# In[32]:


plt.plot(r, p, color = 'green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim(0.9,)
plt.show()

w_vector = w.T
threshold_vector = np.dot(X_test, w)
threshold_vector = threshold_vector.T

#The vector of parameters are w and theta
print("The vectors of parameters are: w vector:", np.asarray(w_vector).reshape(-1), "\n theta vector:",np.asarray(threshold_vector).reshape(-1))


# In[7]:


#Compute logistic regression with sklearn and find its error and F1 data / plots

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
logreg = LogisticRegression()
logreg.fit(X_train, np.asarray(t_train).reshape(-1))
res2 = logreg.predict(X_test)
#res2 = np.matrix(res)
error2 = findError(res2, np.asarray(t_test).reshape(-1))
print("The sklearn logistic regression error is:", error2)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(logreg, X_test, t_test)

skf1 = f1_score(res2, t_test)
print("The sklearn f1 score is:",skf1)


# In[10]:


#3,4:
#find distances
def finddistance(x1, x2):
    distance = np.linalg.norm(x1 - x2)
    return distance

#implement knn with a given test data and label, k is the number of neighbours
def kneighbour(x1, x2, t, k):#x1: train, x2: test
    pred = []
    for i in range(x2.shape[0]):
        dist = []
        sum1 = 0
        for j in range(x1.shape[0]):
            dist.append(finddistance(x2[i,:], x1[j, :]))
        
        distance = np.argsort(dist)
        distance = distance[:k]

        for j in range(k):
            sum1 += t[distance[j],0]
        if(sum1 > k/2):
            pred.append(1)
        else:
            pred.append(0)
    result = np.matrix(pred).T
    return result


def findError(y, t):
    if(y.shape != t.shape):
        return "Dimensions of labels and estimations do not match"
        
    else:
        z = y - t
        error = np.count_nonzero(z)/y.shape[0]
        return error
    
    
# a = kneighbour(X_train, X_test, t_train, 4)
# print(findError(a, t_test))

    
#using kfold to find the neighbour with the lowest error    
def kfold(x,t):
    errorlist = []
    kf = model_selection.KFold(n_splits = 5)    
    for k in range(1,6):
        errorfirst = 0
        for train_index, test_index in kf.split(x):
            kfx_train, kfx_test = x[train_index], x[test_index]
            kfy_train, kfy_test = t[train_index], t[test_index]
            result = kneighbour(kfx_train, kfx_test, kfy_train, k)
            error = findError(result, kfy_test)
            errorfirst += error
        errorlist.append(errorfirst/5)   
    order = np.argsort(errorlist)
    print("The error for each k value of its knn is:",errorlist)
    return errorlist, order[0]+1


def validation(xtrain, xtest, ytrain, ytest):
    errorlist, k = kfold(xtrain, ytrain)
    pred = kneighbour(xtrain, xtest, ytrain, k)
    error = findError(pred, ytest)
    return error

errorlist, order = kfold(X_train, t_train)
print("The desired k value is:",order)
print("The experimental cv error for each k value of its knn is:",errorlist)
error3 = validation(X_train, X_test, t_train, t_test)
print("The experimental test error is:", error3)


# In[45]:


from sklearn.neighbors import KNeighborsClassifier

#using kfold to find the one with lowest error
def kfoldSk(x,t):
    kf = model_selection.KFold(n_splits = 5)
    errorlist = []
    
        
    for k in range(1,6):
        errorin = 0
        for train_index, test_index in kf.split(x):
            kfx_train, kfx_test = x[train_index], x[test_index]
            kfy_train, kfy_test = t[train_index], t[test_index]
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(kfx_train, np.asarray(kfy_train).reshape(-1))
            y_pred = neigh.predict(kfx_test)
            error = findError(y_pred, np.asarray(kfy_test).reshape(-1))
            errorin += error
        errorlist.append(errorin/5)
    order = np.argsort(errorlist)
    
    return errorlist, order[0]+1

#find its validation error
def validation1(xtrain, xtest, ytrain, ytest):
    errorlist, k = kfoldSk(xtrain, ytrain)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain, np.asarray(ytrain).reshape(-1))
    pred = knn.predict(xtest)
    error = findError(pred, np.asarray(ytest).reshape(-1))
    return error

errorlist, k = kfoldSk(X_train, t_train)
print("The sklearn cv errorlist for each k value is:", errorlist)
print("The sklearn k value for its desired neighbour is:", k)
res = validation1(X_train, X_test, t_train, t_test)
print("The sklearn testing error for knn is:",res)


# In[12]:


#part 2


# In[33]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import model_selection
import pandas as pd

#spambase is the folder where i stored the data file
dataset = pd.read_csv('spambase\spambase.data')
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values


# In[34]:


X_train, X_test = train_test_split(X, train_size=2/3, test_size=1/3, random_state = 6048)
t_train, t_test = train_test_split(t, train_size=2/3, test_size=1/3, random_state = 6048)
print(X_train.shape[0])


# In[35]:


#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Define a tree
def findtree(x1, x2, t1, t2, n):
    tree = DecisionTreeClassifier(max_leaf_nodes=n)
    tree.fit(x1, t1)
    res = tree.predict(x2)
    return tree, res


def findError(y, t):
    if(y.shape != t.shape):
        return "Dimensions of labels and estimations do not match"
        
    else:
        z = y - t
        error = np.count_nonzero(z)/y.shape[0]
        return error
    
    
def kfold(x,t,k):
    kf = model_selection.KFold(n_splits = 5)
    errorlist = []
    for n in range(2, k): 
        error = 0
        for train_index, test_index in kf.split(x):
            kfx_train, kfx_test = x[train_index], x[test_index]
            kfy_train, kfy_test = t[train_index], t[test_index]
            estimator, res = findtree(kfx_train, kfx_test, kfy_train, kfy_test, n)
            error += findError(res, kfy_test)
        errorlist.append(error/5)   
    order = np.argsort(errorlist)
    return errorlist, order[0]+2

    
def findLeaves(tree):
    n_nodes = tree.tree_.node_count
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, -1)] 
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    numleaf = 0
    for i in range(is_leaves.shape[0]):
        if (is_leaves[i] == True):
            numleaf+=1
    print(numleaf)

tree = findtree(X_train, X_test, t_train, t_test, 401)[0]
errortree, minerror = kfold(X_train, t_train, 401)
print(minerror)
print(errortree[minerror])



# In[36]:


#10 bagging classifier
print(errortree[minerror-2])

from sklearn.ensemble import BaggingClassifier

errorbag = []
for i in range(1,11):
    bag = BaggingClassifier(n_estimators = 50*i)
    bag.fit(X_train, t_train)
    prediction = bag.predict(X_test)
    errorbag.append(findError(prediction, t_test))
print(errorbag)
    


# In[37]:


#10 random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
errorrf = []
for i in range(1,11):
    rf = RandomForestClassifier(n_estimators = 50*i)
    rf.fit(X_train, t_train)
    prediction = rf.predict(X_test)
    errorrf.append(findError(prediction, t_test))
print(errorrf)


# In[38]:


#10 adaboost classifier with decision stumps
from sklearn.ensemble import AdaBoostClassifier
errorada1 = []
for i in range(1,11):
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = 1), n_estimators = 50*i)
    ada.fit(X_train, t_train)
    prediction = ada.predict(X_test)
    errorada1.append(findError(prediction, t_test))
print(errorada1)


# In[39]:


#10 adaboost classifier with decision trees of 10 leaves
errorada2 = []
for i in range(1,11):
    ada = AdaBoostClassifier(n_estimators = 50*i, base_estimator = DecisionTreeClassifier(max_leaf_nodes=10))
    ada.fit(X_train, t_train)
    prediction = ada.predict(X_test)
    errorada2.append(findError(prediction, t_test))
print(errorada2)


# In[40]:


#10 adaboost classifiers with decision tree without restriction
errorada3 = []
for i in range(1,11):
    ada = AdaBoostClassifier(n_estimators = 50*i, base_estimator = DecisionTreeClassifier(max_leaf_nodes = None, max_depth = None))
    ada.fit(X_train, t_train)
    prediction = ada.predict(X_test)
    errorada3.append(findError(prediction, t_test))
print(errorada3)


# In[41]:


import matplotlib.pyplot as plt
#DT validation error
tree = findtree(X_train, X_test, t_train, t_test, minerror+2)[0]
prediction = tree.predict(X_test)
error = findError(prediction, t_test)
errortreefin = []
pred = []
for i in range(1,11):
    pred.append(50*i)
    errortreefin.append(error)
print(errortreefin)

#Plots the validation error for each ensemble methods
plt.plot(pred, errorbag, color = 'red', label = 'Bagging')
plt.plot(pred, errorrf, color = 'blue', label = 'Random forest')
plt.plot(pred, errorada1, color = 'green', label = 'Adaboost1: Decision Stump')
plt.plot(pred, errorada2, color = 'purple', label = 'Adaboost2: DT with 10 Leaf Nodes')
plt.plot(pred, errorada3, color = 'black', label = 'Adaboost3: Decision Tree')
plt.plot(pred, errortreefin, color = 'pink', label = 'Decision tree')
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.ylim([0, 0.1])
plt.show()


# In[42]:


#Plot the CV error of the decision tree

cv_leaves = list(range(2,401))
plt.plot(cv_leaves, errortree)
plt.show()


# In[ ]:




