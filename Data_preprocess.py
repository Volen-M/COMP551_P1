import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt


def gaussianTest_preprocess():
    pos = [[7.423436942,4.696522875,1],
   [5.745051997,3.533989803,1],
   [9.172168622,2.511101045,1],
   [7.792783481,3.424088941,1],
   [7.939820817,0.791637231,1]]
    neg = [[3.393533211,2.331273381,0],
   [3.110073483,1.781539638,0],
   [1.343808831,3.368360954,0],
   [3.582294042,4.67917911,0],
   [2.280362439,2.866990263,0]]
    return neg, pos

def bernoulliTest_preprocess():
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
    return neg, pos

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
def diverce_converter(cls):
    if cls=='1':
        cls=1
    elif cls=='0':
        cls=0
    else:
        cls=np.nan
    return cls
def cryotherapy_converter(cls):
    if cls=='1':
        cls=1
    elif cls=='0':
        cls=0
    else:
        cls=np.nan
    return cls


def data_analysis(file, argument):
    #argument = 35 for weather
    #argument = 'salary' for adult
    #arugment = 54 for
    positive=[]
    negative=[]
    for i in range(0,file.shape[0]):
        if file[i,-1] ==1:
            positive.append(list(file[i]))
        elif file[i,-1] ==0:
            negative.append(list(file[i]))
    return negative,positive

def naive_data_analysis(X, y):
    #argument = 35 for weather
    #argument = 'salary' for adult
    file = np.concatenate((X,y), axis = 1).tolist()
    positive=[]
    negative=[]
    for row in file:
        if row[-1] ==1:
            positive.append(row)
        elif row[-1]==0:
            negative.append(row)
    return np.array(negative),np.array(positive)

def draw_hist(file, pos,neg, folder = './'):
    for i in range(0,len(pos[1])):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(20,8)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        fig.suptitle('The histogram of {}'.format(file.keys()[i]))
        ax1.title.set_text('{} positive'.format(file.keys()[i]))
        ax1.hist([pos[j][i] for j in range(len(pos))])
        ax2.title.set_text('{} negative'.format(file.keys()[i]))
        ax2.hist([neg[j][i] for j in range(len(neg))])
        plt.savefig((folder + "/{}.png").format(i), dpi = 300)


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
    'names': ['age','workclass','fnlwgt','education','educationnum','maritalstatus','occupation','relationship','race', 'sex','capitalgain','capitalloss','hoursperweek','nativecountry','salary'],
    'header': None,
    # drop the index column
    'usecols': range(0, 15),
    'converters': {'salary': adult_salary_converter},
}

divorce_data_config = {
    'filepath_or_buffer': 'data/Dataset_4/divorce.csv',
    'na_values': ['?'],
    'sep': ';',
    'names': [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53,54],
    'header': None,
    # drop the index column
    'usecols': range(0, 54),
    'converters': {54: diverce_converter},
}

cryotherapy_data_config = {
    'filepath_or_buffer': 'data/Dataset_3/cryotherapy.csv',
    'na_values': ['?'],
    'sep': ',',
    # drop the index column
    'usecols': range(0, 7),
    'converters': {7: cryotherapy_converter},
}
'''
One Hot Encoding
'''

def one_hot(content2):

    workc=pd.get_dummies(content2.workclass,drop_first=True)
    educ=pd.get_dummies(content2.education,drop_first=True)
    marital_st=pd.get_dummies(content2.maritalstatus,drop_first=True)
    occup=pd.get_dummies(content2.occupation,drop_first=True)
    relationship=pd.get_dummies(content2.relationship,drop_first=True)
    race=pd.get_dummies(content2.race,drop_first=True)
    sex=pd.get_dummies(content2.sex,drop_first=True)
    native_country=pd.get_dummies(content2.nativecountry,drop_first=True)
    content_new = pd.DataFrame()

    for key in workc.keys():
        content_new[key] = workc[key]
    for key1 in educ.keys():
        content_new[key1]= educ[key1]
    for key2 in marital_st.keys():
        content_new[key2]= marital_st[key2]
    for key3 in occup.keys():
        content_new[key3]=occup[key3]
    for key4 in relationship.keys():
        content_new[key4] = relationship[key4]
    for key5 in race.keys():
        content_new[key5]= race[key5]
    for key6 in sex.keys():
        content_new[key6]= sex[key6]
    for key7 in native_country.keys():
        content_new[key7]=native_country[key7]

    content_new['age'] = content2.age
    content_new['fnlwgt'] = content2.fnlwgt
    content_new['educationnum'] = content2.educationnum
    content_new['capitalgain'] = content2.capitalgain
    content_new['capitalloss'] = content2.capitalloss
    content_new['hoursperweek'] = content2.hoursperweek
    content_new['salary']=content2.salary
    return content_new
