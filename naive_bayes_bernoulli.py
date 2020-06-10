import numpy as np
import os
import csv
import collections
from math import exp, sqrt, pi, log

class naive_bayes_bernoulli:

    def replaceMatrixZeroes(self, data):
        min_nonzero_prob = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero_prob
        return data

    def fit(self,X, y, laplace=False):
        print("Fitting Naive Bayes model")
        xMax = X.max()
        X = X / xMax
        # input("Enter")
        # print(X)

        # Counting pos and neg amounts
        np.seterr(divide = 'warn')
        posCount = y.sum(axis=0)
        negCount = len(y) - posCount

        # Finding probability of success
        # log( P(y=1) / P(y=0) )
        prior = np.log(posCount / negCount)

        # P(xi=1|y=1)
        p1s = np.dot(y.T, X)
        # input("Enter")
        # print(p1s)
        # P(xi=1|y=0)
        p0s = np.dot(np.ones(len(y)) - y.T, X)
        # input("Enter")
        # print(p0s)

        if (laplace == False):
            p1s = self.replaceMatrixZeroes(p1s)
            p0s = self.replaceMatrixZeroes(p0s)

        # log( P(xi=0|y=1) / P(xi=0|y=0) )
        w0 = np.log((posCount - p1s + laplace) / (posCount + 2 * laplace)) \
            - np.log((negCount - p0s + laplace) / (negCount + 2 * laplace))
        # input("Enter")
        # print(w0)

        # log( P(xi=1|y=1) / P(xi=1|y=0) )
        w1 = np.log((p1s + laplace) / (posCount + 2 * laplace)) \
            - np.log((p0s + laplace) / (negCount + 2 * laplace))
        # input("Enter")
        # print(w1)

        return w0, w1, prior, xMax


    # Function: Implementing Bernoulli Naive Bayes model
    def predict(self, w0, w1, p, xMax, X_test):
        print("Predicting Naive Bayes model")
        X_test = X_test/ xMax


        # log( P(y=1) / P(y=0) ) + Σ ( P(xi=0|y=1) / P(xi=0|y=0) )
        W0 = p + w0.sum(axis=0)
        # input("Enter W0")
        # print(W0)

        # P(xi=1|y=1) / P(xi=1|y=0) - P(xi=0|y=1) / P(xi=0|y=0)
        W = w1 - w0
        # input("Enter W")
        # print(W)
        # print(W0.shape)
        # print(np.dot(X_test, W.T).shape)
        # input("Enter W")

        return np.heaviside(np.dot(X_test, W.T), 0).astype(int) # predict results


    def evaluate_acc(self, y_actual, y_predicted):
        right = 0
        for i in range(len(y_actual)-1):
            #print( y_actual[i], " - ", y_predicted[i])
            if y_actual[i] == y_predicted[i]:
                right += 1
        return right / float(len(y_actual)) * 100.0
