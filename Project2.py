import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam, SGD

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

from statistics import mean, stdev
import scipy.stats as st

boo

def data_processing():
    dataframe = pd.read_csv('./train_data.txt', header=None)  # Load dataset
    dataframe = dataframe.drop(dataframe.columns[[0, 27]], axis = 1) #Removed first column containing participant IDs and UPDRS column
    dataframe.columns = map(str, (list(range(0,27))))
    #print(dataframe[dataframe.columns[1:]].corr()['28'])


    features = dataframe.iloc[:, :-1]
    target = dataframe.iloc[:, -1:]

    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size = 0.6, random_state=20, stratify=target)
    
    return x_train, x_test, y_train, y_test

def MinMax_Normalisation(x_train, x_test):

    #All features normalized except column labeled '24' as it is binary
    min_max = MinMaxScaler()
    
    #Training set normalisation:
    x_train_norm = min_max.fit_transform(x_train)
   
    #testing set normalisation:
    x_test_norm = min_max.fit_transform(x_test)
    

    return x_train_norm, x_test_norm
    
def Standardization(x_train, x_test):

    #All features normalized except column labeled '24' as it is binary
    scaler = StandardScaler()

    #Training set standardized:
    x_train_std = scaler.fit_transform(x_train)
    
    #testing set normalisation:
    x_test_std = scaler.fit_transform(x_test)
    
    return x_train_std, x_test_std

#def Plots(dataframe):

    #plt1 = sns.pairplot(pd.concat((dataframe.iloc[:, :6], target), axis=1), hue='28')
    #plt1.savefig('plot1.png')
    
    #plt2 = sns.pairplot(pd.concat((dataframe.iloc[:, 6:12], target), axis=1), hue='28')
    #plt2.savefig('plot2.png')

    #plt3 = sns.pairplot(pd.concat((dataframe.iloc[:, 12:17], target), axis=1), hue='28')
    #plt3.savefig('plot3.png')

    #plt4 = sns.pairplot(pd.concat((dataframe.iloc[:, 17:21], target), axis=1), hue='28')
    #plt4.savefig('plot4.png')
    
    #df = dataframe.iloc[:, 21:]
   
    #plt5 = sns.pairplot((df.drop('25', axis=1)), hue='28')
    #plt5.savefig('plot5.png')

def NueralNetwork_1layer(x_train, x_test, y_train, y_test, n_1, af_1):

    "This function is a neural network with only 1 hidden layer. This function allows testing various optimisers, learning rates, and momentums while keeping all else constant."
    
    print('hidden layer 1: ', n_1, af_1)
    
    total_test_acc = []
    total_test_error = []

    total_train_acc = []
    total_train_error = []
    
    for _ in range(0,10):
        model = Sequential()
        model.add(Dense(n_1, input_dim=26, activation=af_1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0)
        acc_test = model.evaluate(x_test, y_test, verbose=0)
        acc_train = model.evaluate(x_train, y_train, verbose=0)

        total_test_acc.append(acc_test[1])
        total_test_error.append(acc_test[0])

        total_train_acc.append(acc_train[1])
        total_train_error.append(acc_train[0])

    test_acc_CI = st.t.interval(0.95, len(total_test_acc)-1, loc=np.mean(total_test_acc), scale=st.sem(total_test_acc))
    train_acc_CI = st.t.interval(0.95, len(total_train_acc)-1, loc=np.mean(total_train_acc), scale=st.sem(total_train_acc))
    
    #acc_err_df = pd.DataFrame([total_test_acc, total_test_error, total_train_acc, total_train_error])
    #acc_err_df = acc_err_df.T
    #acc_err_df.columns = ('test_acc', 'test_error', 'train_acc', 'train_error')

    return total_test_acc, test_acc_CI, total_train_acc, train_acc_CI


def NueralNetwork_2ayer(x_train, x_test, y_train, y_test, n_1, af_1):

    "This function is a neural network eith only 1 hidden layer to test different numbers of neurons."
    
    print('hidden layer 1: ', n_1, af_1)
    
    total_test_acc = []
    total_test_error = []

    total_train_acc = []
    total_train_error = []
    
    for _ in range(0,10):
        model = Sequential()
        model.add(Dense(n_1, input_dim=26, activation=af_1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.075, momentum=0.001), metrics=['accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0)
        acc_test = model.evaluate(x_test, y_test, verbose=0)
        acc_train = model.evaluate(x_train, y_train, verbose=0)

        total_test_acc.append(acc_test[1])
        total_test_error.append(acc_test[0])

        total_train_acc.append(acc_train[1])
        total_train_error.append(acc_train[0])

    test_acc_CI = st.t.interval(0.95, len(total_test_acc)-1, loc=np.mean(total_test_acc), scale=st.sem(total_test_acc))
    train_acc_CI = st.t.interval(0.95, len(total_train_acc)-1, loc=np.mean(total_train_acc), scale=st.sem(total_train_acc))
    
    #acc_err_df = pd.DataFrame([total_test_acc, total_test_error, total_train_acc, total_train_error])
    #acc_err_df = acc_err_df.T
    #acc_err_df.columns = ('test_acc', 'test_error', 'train_acc', 'train_error')

    return total_test_acc, test_acc_CI, total_train_acc, train_acc_CI

def NueralNetwork_3layers(x_train, x_test, y_train, y_test, n_1, af_1, n_2, af_2, n_3, af_3):
    
    print('hidden layer 1: ', n_1, af_1)
    print('hidden layer 2: ', n_2, af_2)
    print('hidden layer 3: ', n_3, af_3)
   

    total_test_acc = []
    total_test_error = []

    total_train_acc = []
    total_train_error = []
    
    for _ in range(0,10):
        model = Sequential()
        model.add(Dense(n_1, input_dim=26, activation=af_1))
        model.add(Dense(n_2, activation=af_2))
        model.add(Dense(n_3, activation=af_3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam(learning_rate=0.001,beta_1=0.075), metrics=['accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0)
        acc_test = model.evaluate(x_test, y_test, verbose=0)
        acc_train = model.evaluate(x_train, y_train, verbose=0)

        total_test_acc.append(acc_test[1])
        total_test_error.append(acc_test[0])

        total_train_acc.append(acc_train[1])
        total_train_error.append(acc_train[0])

    test_acc_CI = st.t.interval(0.95, len(total_test_acc)-1, loc=np.mean(total_test_acc), scale=st.sem(total_test_acc))
    train_acc_CI = st.t.interval(0.95, len(total_train_acc)-1, loc=np.mean(total_train_acc), scale=st.sem(total_train_acc))
    
    return total_test_acc, test_acc_CI, total_train_acc, train_acc_CI

def NueralNetwork_4layers(x_train, x_test, y_train, y_test, n_1, af_1, n_2, af_2, n_3, af_3, n_4, af_4):
    
    print('hidden layer 1: ', n_1, af_1)
    print('hidden layer 2: ', n_2, af_2)
    print('hidden layer 3: ', n_3, af_3)
    print('hidden layer 4: ', n_4, af_4)

    total_test_acc = []
    total_test_error = []

    total_train_acc = []
    total_train_error = []
    
    for _ in range(0,10):
        model = Sequential()
        model.add(Dense(n_1, input_dim=26, activation=af_1))
        model.add(Dense(n_2, activation=af_2))
        model.add(Dense(n_3, activation=af_3))
        model.add(Dense(n_4, activation=af_4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam(learning_rate=0.001,beta_1=0.075), metrics=['accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0)
        acc_test = model.evaluate(x_test, y_test, verbose=0)
        acc_train = model.evaluate(x_train, y_train, verbose=0)

        total_test_acc.append(acc_test[1])
        total_test_error.append(acc_test[0])

        total_train_acc.append(acc_train[1])
        total_train_error.append(acc_train[0])

    test_acc_CI = st.t.interval(0.95, len(total_test_acc)-1, loc=np.mean(total_test_acc), scale=st.sem(total_test_acc))
    train_acc_CI = st.t.interval(0.95, len(total_train_acc)-1, loc=np.mean(total_train_acc), scale=st.sem(total_train_acc))
    
    return total_test_acc, test_acc_CI, total_train_acc, train_acc_CI

def main():

    x_train, x_test, y_train, y_test = data_processing()
    
   # x_train_norm, x_test_norm = MinMax_Normalisation(x_train, x_test)
    x_train_std, x_test_std = Standardization(x_train, x_test)

    test_acc, test_CI, train_acc, train_CI = NueralNetwork_1layer(x_train_std, x_test_std, y_train, y_test, 10, 'sigmoid')
    
    print('test acc:\n', round(mean(test_acc), 3), test_CI)
    print('train acc:\n', round(mean(train_acc), 3), train_CI)

    #for i in [1, 5, 10, 15, 20, 25, 30]:
      
        #test_acc, test_CI, train_acc, train_CI = NueralNetwork_1layer(x_train_std, x_test_std, y_train, y_test, 10, 'softplus', SGD(lr=0.075, momentum=0.001))
        #test_acc, test_CI, train_acc, train_CI = NueralNetwork_2ayer(x_train_std, x_test_std, y_train, y_test, i, 'tanh')
        #test_acc, test_CI, train_acc, train_CI = NueralNetwork_3layers(x_train_std, x_test_std, y_train, y_test, 30, 'tanh', 30, 'softplus', i, 'softplus')
        #test_acc, test_CI, train_acc, train_CI = NueralNetwork_4layers(x_train_std, x_test_std, y_train, y_test, 30, 'tanh', 30, 'softplus', 5, 'softplus', i, 'softplus')
        

        #print('adam(learning_rate=0.001, beta_1=0.075)')
        #print('test acc:\n', round(mean(test_acc), 3), test_CI)
        #print('train acc:\n', round(mean(train_acc), 3), train_CI)

if __name__=='__main__':
    main() 

