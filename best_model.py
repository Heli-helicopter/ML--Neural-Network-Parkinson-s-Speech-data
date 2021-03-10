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
from sklearn.metrics import roc_curve, roc_auc_score

from statistics import mean, stdev
import scipy.stats as st



def data_processing():
    dataframe = pd.read_csv('./train_data.txt', header=None)  # Load dataset
    #print(dataframe.describe())
    dataframe = dataframe.drop(dataframe.columns[[0, 27]], axis = 1) #Removed first column containing participant IDs, and second last column contianing UPDRS
    dataframe.columns = map(str, (list(range(0,27))))
    
    #print(dataframe[dataframe.columns[1:]].corr()['28'])

     
    #print(dataframe.head())

    features = dataframe.iloc[:, :-1]
    target = dataframe.iloc[:, -1:]

    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size = 0.6, random_state=20, stratify=target)
    
    return x_train, x_test, y_train, y_test

def MinMax_Normalisation(x_train, x_test):

    #All features normalized 
    min_max = MinMaxScaler()
    
    #Training set normalisation:
    train_scaled = x_train.drop('24', axis=1)
    train_scaled = min_max.fit_transform(train_scaled)
    col_24 = x_train[['24']].to_numpy()
    x_train_norm = np.concatenate((train_scaled, col_24), axis=1)

    #testing set normalisation:
    #test_scaled = x_test.drop('24', axis=1)
    x_test_norm = min_max.fit_transform(x_test)
    #col_24 = x_test[['24']].to_numpy()
    #x_test_norm = np.concatenate((test_scaled, col_24), axis=1)

    return x_train_norm, x_test_norm
    
def Standardization(x_train, x_test):

    #All features normalized except column labeled '24' as it is binary
    scaler = StandardScaler()

    #Training set standardized:
    #train_scaled = x_train.drop('24', axis=1)
    x_train_std = scaler.fit_transform(x_train)
    #col_24 = x_train[['25']].to_numpy()
    #x_train_std = np.concatenate((train_scaled, col_24), axis=1)

    #testing set normalisation:
    #test_scaled = x_test.drop('24', axis=1)
    x_test_std = scaler.fit_transform(x_test)
    #col_24 = x_test[['24']].to_numpy()
    #x_test_std = np.concatenate((test_scaled, col_24), axis=1)

    return x_train_std, x_test_std


def NueralNetwork_3layers(x_train, x_test, y_train, y_test, n_1, af_1, n_2, af_2, n_3, af_3):
    
    model = Sequential()
    model.add(Dense(n_1, input_dim=26, activation=af_1))
    model.add(Dense(n_2, activation=af_2))
    model.add(Dense(n_3, activation=af_3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam(learning_rate=0.001,beta_1=0.075), metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0)
    acc_test = model.evaluate(x_test, y_test, verbose=0)
    acc_train = model.evaluate(x_train, y_train, verbose=0)

    y_pred = model.predict(x_test)
    y_pred_round = tf.round(y_pred)

    matrix = confusion_matrix(y_test, y_pred_round)
    class_names = ['No', 'Yes']
    dataframe_Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
    
    print('Classification Report:\n', classification_report(y_test, y_pred_round))
    print('Test accuracy: ', round(acc_test[1], 2))
    print('Train accuracy: ', round(acc_train[1], 2))

    return y_pred, dataframe_Confusion

def plot_confusion_matrix(matrix):

    sns.heatmap(matrix, annot=True,  cmap="Blues", fmt=".0f")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig('./confusion_matrix.png')
    plt.close()

def ROC_AUC_plot(y_test, y_pred):
    
    auc = roc_auc_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
    print('Model ROC AUC: ', round(auc, 2))

    plt.plot(fpr, tpr, marker='.', label='Neural Network')

    plt.title('Receiver Operating Characteristics for Neural Network')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('ROC.png')
    plt.close()

def main():

    x_train, x_test, y_train, y_test = data_processing()
    
    x_train_std, x_test_std = Standardization(x_train, x_test)

    y_pred, confu_matrix = NueralNetwork_3layers(x_train_std, x_test_std, y_train, y_test, 30, 'tanh', 30, 'softplus', 5, 'softplus')
    plot_confusion_matrix(confu_matrix)
    ROC_AUC_plot(y_test, y_pred)

    

if __name__=='__main__':
    main() 
