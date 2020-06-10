import pandas as pd
import numpy as np
from lregression import LogisticRegression
from Data_preprocess import weather_data_config, adult_data_config, one_hot


"""
Import Data
"""
adult_data = one_hot(pd.read_csv(**adult_data_config).dropna())
adult_data_np = adult_data.to_numpy()

a = adult_data_np.shape[1]
x_data = adult_data_np[:, range(a)]
y_target = (adult_data_np[0:,95 ]).T

"""
Learning Rate Sensitivity Analysis
"""

model = LogisticRegression()
learning_rate = 0.0005
epsilon = 0.1


for n

    [weights, iterations] = model.fit(x_data, y_target,learning_rate, epsilon, 10000000)
    predict = model.predict(x_data, weights)
    accuracy = model.evaluate_acc(y_target, predict)

    accuracy[:,n] = accuracy



