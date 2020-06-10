import pandas as pd
import numpy as np
from lregression import LogisticRegression
from naive_bayes_gaussian import naive_bayes_gaussian
from naive_bayes_bernoulli import naive_bayes_bernoulli
import Data_preprocess as preprocess
import k_fold_crossv as kf


"""
Import Data
"""


runfeature = 10

if (runfeature== 0):
    content = pd.read_csv(**preprocess.weather_data_config).dropna()
    num_feature = len(content.keys()) - 1
    content_x = np.matrix(content.values[:, range(num_feature)])
    content_y = (np.matrix(content.values[:, num_feature])).T

    neg, pos = preprocess.naive_data_analysis(content_x, content_y)
    model = naive_bayes_gaussian()
    statsDict, totalDataAmt = model.fit(neg,pos)

    prediction = model.predict(statsDict,totalDataAmt, np.array(content_x))
    print(model.evaluate_acc(prediction, content_y))

elif (runfeature == 1):
    content = pd.read_csv(**preprocess.adult_data_config).dropna()
    content_One_hot=preprocess.one_hot(content)

    num_feature = len(content_One_hot.keys()) - 1
    content_x = np.matrix(content_One_hot.values[:, range(num_feature)])
    content_y = (np.matrix(content_One_hot.values[:, num_feature])).T

    #neg, pos = preprocess.naive_data_analysis(content_x, content_y, 'weather')
    model = naive_bayes_bernoulli()
    #statsDict, totalDataAmt = model.fit(neg,pos, 1)

    #prediction = model.predict(statsDict,totalDataAmt, np.array(content_x))
    w0, w1, prior, xMax = model.fit(content_x,content_y, True)
    prediction = model.predict(w0, w1, prior, xMax, content_x)
    print(len(prediction))
    print(len(prediction[0]))
    #print(content_y)
    print(model.evaluate_acc(content_y.tolist(), prediction.tolist()))

elif (runfeature == 2): #BERNOULLI DATASET 1 CV
    content = pd.read_csv(**preprocess.adult_data_config).dropna()
    content = preprocess.one_hot(content)
    model = naive_bayes_bernoulli()

    num_feature = len(content.keys()) - 1
    content_x = np.matrix(content.values[:, range(num_feature)])
    content_y = (np.matrix(content.values[:, num_feature])).T
    print(kf.k_fold_validation_Bernoulli(model,content_x, content_y, 5, False))

elif (runfeature == 3): #GAUSSIAN DATASET 1 CV
    content = pd.read_csv(**preprocess.weather_data_config).dropna()
    model = naive_bayes_gaussian()

    num_feature = len(content.keys()) - 1
    content_x = np.matrix(content.values[:, range(num_feature)])
    content_y = (np.matrix(content.values[:, num_feature])).T
    print(kf.k_fold_validation_Gaussian(model,content_x, content_y, 5, False))

elif (runfeature == 4): #GAUSSIAN DATASET 2 CV
    content = pd.read_csv(**preprocess.adult_data_config).dropna()
    content = preprocess.one_hot(content)
    model = naive_bayes_gaussian()

    num_feature = len(content.keys()) - 1
    content_x = np.matrix(content.values[:, range(num_feature)])
    content_y = (np.matrix(content.values[:, num_feature])).T
    print(kf.k_fold_validation_Gaussian(model,content_x, content_y, 5, False))

elif (runfeature == 5): #GAUSSIAN DATASET 4 CV
    content = pd.read_csv(**preprocess.divorce_data_config).dropna().astype(float)
    model = naive_bayes_gaussian()

    num_feature = len(content.keys()) - 1
    content_x = np.matrix(content.values[:, range(num_feature)])
    content_y = (np.matrix(content.values[:, num_feature])).T
    print(kf.k_fold_validation_Gaussian(model,content_x, content_y, 5, False))

elif (runfeature == 6): #GAUSSIAN DATASET 3 CV
    content = pd.read_csv(**preprocess.cryotherapy_data_config).dropna()
    model = naive_bayes_gaussian()

    num_feature = len(content.keys()) - 1
    content_x = np.matrix(content.values[:, range(num_feature)])
    content_y = (np.matrix(content.values[:, num_feature])).T
    print(kf.k_fold_validation_Gaussian(model,content_x, content_y, 5, False))
elif (runfeature == 7):
    content1 = pd.read_csv(**preprocess.weather_data_config).dropna()
    content2 = pd.read_csv(**preprocess.adult_data_config).dropna()
    content2 = preprocess.one_hot(content2)
    content3 = pd.read_csv(**preprocess.cryotherapy_data_config).dropna()
    content4 = pd.read_csv(**preprocess.divorce_data_config).dropna().astype(float)
    print(content1.shape)
    print(content2.shape)
    print(content3.shape)
    print(content4.shape)

elif (runfeature ==10): #### 3.3 DataSet 2
    content2 = pd.read_csv(**preprocess.adult_data_config).dropna()
    content = preprocess.one_hot(content2)

    num_feature = len(content.keys()) - 1
    X = np.matrix(content.values[:, range(num_feature)])
    y = (np.matrix(content.values[:, num_feature])).T
    train_validation_ratio=[0.15,0.30,0.45,0.60,0.75,0.90]
    Train_vali=[]
    for i in train_validation_ratio:
        #reset model everytime after implementing it
        index = len(X)*i
        train_x = X[0:index, :]
        test_y = y[0:index]
        model = naive_bayes_gaussian()
        neg, pos = preprocess.naive_data_analysis(train_x, train_y)
        statsDict, totalDataAmt = model.fit(neg,pos)
        prediction = model.predict(statsDict,totalDataAmt, np.array(X))
        accuracy = model.evaluate_acc(prediction, content_y)
        Train_vali.append(accuracy)
    plt.figure(num=None, figsize=(10,6))
    plt.grid(color='b', linestyle='-', linewidth=0.1)
    plt.title('DATASET1: Validation Accuracy vs Ratio for Validation and Training Dataset')
    plt.xlabel('Ratio for Validation and Training Dataset')
    plt.ylabel('Validation Dataset Accuracy')
    plt.plot([1-i for i in train_validation_ratio], Train_vali)
    plt.savefig("Dataset2__Validation_Accuracy_vs_ratio.png", dpi = 300)
elif (runfeature == 50):
    pos = [[0,1,0,1,1],
           [1,0,1,0,1],
           [0,1,0,1,1],
           [0,1,0,1,1],
           [0,1,0,1,1]]

    neg = [[1,0,1,0,0],
           [1,0,1,0,0],
           [0,1,0,1,0],
           [0,1,1,0,0],
           [1,0,1,0,0]]
    content = np.concatenate((pos, neg), axis = 0)

    print(len(content[0]))
    num_feature = len(content[0]) - 1
    content_x = np.matrix(content[:, range(num_feature)])
    content_y = np.matrix(content[:, num_feature])
    model = naive_bayes_bernoulli()
    #statsDict, totalDataAmt = model.fit(neg,pos, True)
    prediction = model.naiveBayes(content_x,content_y,content, True)
    print(prediction)
    #prediction = model.predict(statsDict,totalDataAmt, content)
    print(model.evaluate_acc(prediction, content_y))
