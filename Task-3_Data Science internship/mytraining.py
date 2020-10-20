#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


if __name__ == "__main__":
    # read the data
    df=pd.read_csv('Iris.csv')
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    X = df.drop(columns=['Species','Id'])
    Y = df['Species']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    # close the file
    file.close()
    

