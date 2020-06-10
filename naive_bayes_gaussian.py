import numpy as np
import os
import csv
import collections
from math import exp, sqrt, pi
from Data_preprocess import gaussianTest_preprocess, bernoulliTest_preprocess


class naive_bayes_gaussian:

    def __mean(self, data):
        return np.sum(data)/np.float(len(data))

    def __stdDev(self, data):
        mean = self.__mean(data)
        variance = sum([(x-mean)**2 for x in data])/float(len(data)-1)
        return sqrt(variance)

    def __gaussianPreprocess(self, data):
        meanDevAmtMatrix = []

        for column in zip(*data):
            meanDevAmtMatrix.append([self.__mean(column), self.__stdDev(column), len(column)])

        del(meanDevAmtMatrix[-1]) #Get rid f the Y (the answer stuff)
        return meanDevAmtMatrix

    def __getGaussianProbability(self, x, mean, stdDev):
        if(stdDev == 0):
            return 1
        return exp(-((x-mean)**2/(2*stdDev**2)))/(sqrt(2*pi)*stdDev)


    def fit(self, neg, pos):
        statsDict = dict()

        count = []
        count.append(len(neg))
        count.append(len(pos))

        neg_stats = self.__gaussianPreprocess(neg)
        pos_stats = self.__gaussianPreprocess(pos)

        statsDict = {
        0 : neg_stats,
        1 : pos_stats}
        totalDataAmt = sum(count)

        return statsDict, totalDataAmt

    def predict(self, statsDict, totalDataAmt, testData):
        odds = dict()
        predictions = []
        for row in testData:
            for index, stats_row in statsDict.items():
                odds[index] = statsDict[index][0][-1]/float(totalDataAmt)
                for i in range(len(stats_row)):
                    mean, stdDev, _ = stats_row[i]
                    odds[index] *= self.__getGaussianProbability(row[i], mean, stdDev)
            predictions.append(1 if odds[0] < odds[1] else 0)
            odds = dict()
        return predictions


    def evaluate_acc(self, y_actual, y_predicted):
        right = 0
        for i in range(len(y_actual)):
            if y_actual[i] == y_predicted[i]:
                right += 1
        return right / float(len(y_actual)) * 100.0
