import numpy as np
import sys
import pandas as pd
import Data_preprocess
from lregression import LogisticRegression
from Data_preprocess import weather_data_config, adult_data_config, one_hot

'''
1. import Data
'''
a = 1

if a == 1:
    weather_data_pd= pd.read_csv(**weather_data_config).dropna()
    weather_data_np = weather_data_pd.to_numpy()

    x_data = weather_data_np[:,0:34]
    y_target = weather_data_np[:,34:35]

elif a == 2:

    adult_data = one_hot(pd.read_csv(**adult_data_config).dropna())
    adult_data_np = adult_data.to_numpy()

    a = adult_data_np.shape[1]
    x_data = adult_data_np[:, range(a)]
    y_target = (adult_data_np[0:,95 ]).T

    #print(adult_data)
    #print(y_target)

'''
2. run model
'''

model = LogisticRegression()
learning_rate = 0.0005
epsilon = 0.1

#weights = model.fit(x_data, y_target,learning_rate, epsilon, 10000000)
#print(weights)

#predict = model.predict(x_data, weights)
#print(predict)

#accuracy = model.evaluate_acc(y_target, predict)
#print('Accuracy is ', accuracy)

[optimised_weights, accuracy] = model.k_fold_validation(x_data, y_target, 5, 0.0015, 0.005, 10000)

print(optimised_weights)
print('Accuracy is', accuracy)

