import numpy as np
import pandas as pd
import os
import csv
import collections
import glob
import Data_preprocess
from naive_bayes import naive_bayes


# Function: Main function
def main():
     print("Importing all data")

     #importedData = importDataFile('data/Dataset_1/ionosphere.data')
     #content = pd.read_csv(**adult_data_config).dropna()
     #print(content)
     #print("Matrix is " + str(len(importedData)) + " x " + str(len(importedData[0])))
     #print("Here is the split data")
     dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]

     dataset2 = [[1,1,1],
            [0,0,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [0,0,0],
            [0,0,0],
            [1,1,0],
            [1,0,0],
            [0,0,0]]

     dataset2 = [[0,1,0,1,1],
            [1,0,1,0,1],
            [0,1,0,1,1],
            [0,1,0,1,1],
            [0,1,0,1,1],
            [1,0,1,0,0],
            [1,0,1,0,0],
            [0,1,0,1,0],
            [0,1,1,0,0],
            [1,0,1,0,0]]
     nb = naive_bayes()
     print(nb.getBernoulliOutcomeProbability(dataset2))
     #print(nb.getGaussianOutcomeProbability(dataset))
     return


# Function: Import data set
# def importDataFile(filepath):
#     with open(filepath, 'r', encoding="utf8") as file:
#         matrix = [[num for num in line.split(',')] for line in file]
#     return matrix
# Function: Analysis dataset



# Function: Throw Data set to LogisticRegression
def data_split(file, argument):
    positive=[]
    negative=[]
    for i in range(0,len(file['salary'])):
        if file[argument].iloc[i] ==1:
            positive.append(list(file.iloc[i]))
        elif file[argument].iloc[i]==0:
            negative.append(list(file.iloc[i]))
    return negative,positive

def weather_35_converter(gb):
    if gb =='g':
        gb=1
    elif gb=='b':
        gb=0
    else:
        gb=np.nan
    return gb

def adult_salary_converter(k50):
    if k50==' >50K':
        k50=1
    elif k50==' <=50K':
        k50=0
    else:
        k50=np.nan
    return k50


weather_data_config = {
    'filepath_or_buffer': 'data/Dataset_1/ionosphere.data',
    'na_values': ['?'],
    'sep': ',',
    'names': [1, 2, 3, 4, 5,6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    'header': None,
    # drop the index column
    'usecols': range(0, 35),
    'converters': {35: weather_35_converter},
}
adult_data_config = {
    'filepath_or_buffer': 'data/Dataset_2/adult.data',
    'na_values': [' ?'],
    'sep': ',',
    'names': ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race', 'sex','capital-gain','capital-loss','hours-per-week','native-country','salary'],
    'header': None,
    # drop the index column
    'usecols': range(0, 15),
    'converters': {'salary': adult_salary_converter},
}


# Function: Throw Date set to NaiveBayes



# Call Order to run assignment
main()
