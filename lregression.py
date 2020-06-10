import numpy as np
import math

class Report:
    def __init__(self):
        self.time = 0
        self.epochs = []
        self.accs = []
        self.errors = []
        self.val_accs = []
        self.val_errors = []

class LogisticRegression:

    # Logistic regression class
    # Inputs:
    #  X with size N x D
    #  Y with size N x 1
    #  learning rate
    def __init__(self):
        self.MAX_ERROR = 10000000000000
        self.lr = None
        self.eps = None
        self.model_param = None
        self.model_output = None
        self.report = Report()

    def fit(self, x_data, y_target, learning_rate, epsilon, stop, shouldReport = False, x_validation = [], y_validation = []):
        """
        Fit model coefficients.

        Inputs:
        X: 1D or 2D numpy array
        y: 1D numpy array
        learning_rate
        epsilon
        """
        N, D = x_data.shape
        w = np.zeros(D)
        gradient = np.inf
        wT = np.array([w]).T
        np.seterr(all='ignore')
        i = 0
        #norm = 0
        #if use change in error, use following condition
        err_before = 0
        err = 0
        #(abs(err - err_before)
        #np.linalg.norm(gradient) > epsilon
        while  (abs(err - err_before) > epsilon  and i < stop) or i == 0 or err == self.MAX_ERROR:
            i += 1
            a = np.dot(x_data, wT)  # N x 1
            function = self.logistic(a)
            b = y_target - function
            gradient = x_data.T*b
            #norm = np.linalg.norm(gradient)
            err_before = err
            err = self.error(y_target, function)
            if i % 10==0 and shouldReport:
                self.report.epochs.append(i)
                self.report.errors.append(err)
                function[function < 0.5] = 0
                function[function >= 0.5] = 1
                self.report.accs.append(self.evaluate_acc(y_target, function))
                if len(x_validation) > 0 and len(y_validation) > 0:
                    predictions = self.predict(x_validation, wT)
                    val_y_pre = self.logistic(np.dot(x_validation, wT))
                    self.report.val_errors.append(self.error(y_validation, val_y_pre))
                    self.report.val_accs.append(self.evaluate_acc(y_validation, predictions))

            #print(err)
            wT = wT + (learning_rate * gradient)

        print(i)
#             print('err',err)
#             print('err_before',err_before)
#             print('err-errbefore',(err - err_before))

        # set attributes
        self.model_param = wT

        return self.model_param,i

    def predict(self, x_data, weights):
        """
        Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        weights: 1D numpy array
        """
        model_output = self.logistic(np.dot(x_data, weights))
        model_output[model_output < 0.5] = 0
        model_output[model_output >= 0.5] = 1
        self.model_output = model_output
        return self.model_output

    def logistic(self,a):
        return (1 / (1 + np.exp(-a)))
    def evaluate_acc(self,y_target, y_predict):

        tptn = np.equal(y_target,y_predict)
        TPTN = np.sum([tptn])
        FPFN = np.size(y_target) - TPTN
        accuracy = TPTN/(FPFN+TPTN)
        return [accuracy]

    def error(self, y_target, y_predict):
        error=np.sum(np.multiply(y_target, -np.log(y_predict)) + np.multiply((1-y_target),-np.log((1-y_predict))))
        if np.isnan(error):
            error=self.MAX_ERROR
        #print(error)
        return error


    def k_fold_validation(self, x_data, y_data, k, learning_rate, epsilon, stop):

        # 1. Get data set size and calculate fold size
        n_rows = x_data.shape[0] # calculate number fo rows
        n_columns = x_data.shape[1]
        start_0 = 0 # assigns initial start
        end_0 = int(n_rows/k)    # assigns initial end

        # initialize
        k_weights = np.ones((n_columns, k))
        k_accuracies = np.ones((1, k))

        for n in range(0, k-1):

            # get validation indices
            start = start_0 + (n * end_0)
            end = end_0 + (n * end_0)
            validate_indices = np.array(range(start, end))

            # construct training and validation data set
            x_data_training = np.delete(x_data,validate_indices, 0)
            x_data_validate = x_data[start:end]
            y_data_training = np.delete(y_data,validate_indices, 0)
            y_data_validate = y_data[start:end]

            # calculate the weights for training data set
            weights = self.fit(x_data_training, y_data_training, learning_rate, epsilon, stop)
            # calculate accuracy on validation data set
            predict = self.predict(x_data_validate, weights)
            accuracy = self.evaluate_acc(y_data_validate, predict)

            # compile results
            k_weights[:, n] = weights[:, 0]
            k_accuracies[:, n] = accuracy

        # calculate mean weights and accuracy
        mean_weights = np.mean(k_weights, axis = 1)
        mean_accuracy = np.mean(k_accuracies, axis = 1)

        return mean_weights, mean_accuracy
