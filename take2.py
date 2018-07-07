import numpy as np
import random


data=[]
#x,y,bias
for _ in range(1,100):
    data.append([[random.uniform(2, 4), random.uniform(2, 4), -1], -1])
    data.append([[random.uniform(5, 7), random.uniform(4, 7), -1], 1])

XY = np.array(data)

def test(w,X,Labels):
    correct=0
    for i, x in enumerate(X):
        if np.sign(np.dot(X[i], w)) == np.sign(Labels[i]):
            correct+=1
    return correct/len(X)

#X is data
#Y is label
def svm(XY):

    w = np.zeros(len(XY[0][0]))
    rate = 0.001
    iterations = 100000
    lamda = 100000

    for _ in range(1, iterations):
        sample = np.array(random.sample(XY, 32))
        X = np.array(sample[:, 0])
        Y = np.array(sample[:, 1])
        gradient = 0
        for i, x in enumerate(X):
            if (Y[i] * np.dot(X[i], w)) < 1:
                gradient += (2.0 * w / lamda) - (np.array(X[i]) * Y[i])
            else:
                gradient += (2.0 * w / lamda)
        w = w - rate * gradient

    return w


w = svm(XY)
print "vector:", w
print "success in test "+str(test(w,np.array(XY[:, 0]), np.array(XY[:, 1]))*100)+"%"
