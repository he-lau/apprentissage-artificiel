import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA





def print_accuracy_tree_classifier(x_train, x_test, y_train, y_test):

    model = DecisionTreeClassifier()

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    print("Tree Classifier accuracy :",accuracy_score(y_test,y_pred))


def print_accuracy_knn(x_train, x_test, y_train, y_test, n):
        

    neigh = KNeighborsClassifier(n_neighbors=n)

    neigh.fit(x_train, y_train)

    y_pred = neigh.predict(x_test)

    print("KNN-"+str(n)+ " accuracy :", accuracy_score(y_test,y_pred)) 
    

def normalisation(x):
    
    # avec normalisation 

    #from sklearn.preprocessing import StandardScaler

    #norm = StandardScaler()

    #x = norm.fit_transform(x)


    norm = MinMaxScaler()

    x = norm.fit_transform(x)


    principal = PCA(n_components=10)

    principal.fit(x)

    x = principal.transform(x)
    
    return x


def visualisation_hist(data, title):
    
    plt.hist(data)
    plt.title(title)
    plt.show()
    
    
    
    