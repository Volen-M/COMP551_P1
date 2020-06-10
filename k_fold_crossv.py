import lregression
import numpy as np
import Data_preprocess as preprocess


def k_fold_validation( model_template, X,y,fold_size,ls,epislon,stop):
    #divide data
    size = np.shape(X)[0]
    fold_sizes = np.array([int(size / fold_size)] * fold_size)

    for i in range(size % fold_size):
        fold_sizes[i] += 1
    acc = []
    cross_validation = []
    accuracy=[]
    for k in range(0, fold_size):
        #model.reset()
        validation_start_index = sum(fold_sizes[range(0, k)])
        validation_set_X = X[range(validation_start_index, validation_start_index+fold_sizes[k])]
        validation_set_y = y[range(validation_start_index, validation_start_index+fold_sizes[k])]
        final_training_set_X = np.zeros((1, X.shape[1]))
        final_training_set_y = []
        for i in range(0, fold_size):
            if (i == k):
                continue
            else:
                start_index = sum(fold_sizes[range(0, i)])
                training_set_X = X[range(start_index, start_index+fold_sizes[i])]
                training_set_y = y[range(start_index, start_index+fold_sizes[i])]

                final_training_set_X = np.vstack((final_training_set_X, training_set_X))
                final_training_set_y = np.concatenate((final_training_set_y, training_set_y))

        model = model_template()
        #drop the first row
        final_training_set_X = final_training_set_X[range(1, len(final_training_set_X))]
       # print(final_training_set_y.shape)
        final_training_set_y = np.matrix(final_training_set_y).T

        weights,i = model.fit(final_training_set_X, final_training_set_y,ls,epislon,stop)
        predict= model.predict(validation_set_X, weights)
        #print(model.evaluate_acc(np.matrix(validation_set_y).T, predict))
        acc.append(model.evaluate_acc(np.matrix(validation_set_y).T, predict))
        #print(acc)
    return np.mean(acc), np.sum(predict),weights


def k_fold_validation_Bernoulli(model, x_data, y_data, k, laplace):

        # 1. Get data set size and calculate fold size
    n_rows = x_data.shape[0] # calculate number fo rows
    n_columns = x_data.shape[1]
    start_0 = 0 # assigns initial start
    end_0 = int(n_rows/k)    # assigns initial end

        # initialize
    k_accuracies = []

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

        # calculate accuracy on validation data set
        w0, w1, prior, xMax = model.fit(x_data_training,y_data_training, laplace)
        prediction = model.predict(w0, w1, prior, xMax, x_data_validate)
        accuracy = model.evaluate_acc(y_data_validate, prediction)

        # compile results
        k_accuracies.append(accuracy)
        print(accuracy)
        # calculate mean weights and accuracy

    mean_accuracy = np.mean(k_accuracies)

    return mean_accuracy




def k_fold_validation_Gaussian(model, x_data, y_data, k, laplace):

        # 1. Get data set size and calculate fold size
    n_rows = x_data.shape[0] # calculate number fo rows
    n_columns = x_data.shape[1]
    start_0 = 0 # assigns initial start
    end_0 = int(n_rows/k)    # assigns initial end

    k_accuracies = []
    firstPass = True
    finalDict = dict()
    finalAmt = 0
    for n in range(0, k):
        print(n)
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
            # calculate accuracy on validation data set
        neg, pos = preprocess.naive_data_analysis(x_data_training, y_data_training)
        statsDict, totalDataAmt = model.fit(neg,pos)

        prediction = model.predict(statsDict,totalDataAmt, np.array(x_data_validate))
        accuracy = model.evaluate_acc(y_data_validate, prediction)

        # compile results
        k_accuracies.append(accuracy)
        print(accuracy)
        # calculate mean weights and accuracy
        # if (firstPass):
        #     finalAmt =




    mean_accuracy = np.mean(k_accuracies)

    return mean_accuracy

def fitModelWithValidate(model, X, y, validationRatio, ls,epislon,stop, shouldReport = True):
    breakIndex = int(X.shape[0] * validationRatio)
    train_x = X[breakIndex:,0:]
    validate_x=X[0:breakIndex:,0:]
    train_y=y[breakIndex:, :]
    validate_y=y[0:breakIndex,:]
    mean_accuracy, final_w0, final_w1, final_prior, final_xMax = model.k_fold_validationfit_Bernoulli(model, train_x, train_y, k, True)
    prediction = model.predict(final_w0, final_w1, final_prior, final_xMax, validate_x)
    accuracy = model.evaluate_acc(prediction, validate_y)
    return accuracy
