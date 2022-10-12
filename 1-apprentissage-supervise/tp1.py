import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('credit_scoring.csv',sep=";")

labels = "Seniority;Home;Time;Age;Marital;Records;Job;Expenses;Income;Assets;Debt;Amount;Price"
labels = labels.split(";")

# taking all the rows and all the columns except the last column
x = dataset.iloc[:,:-1].values

# all the rows and only the last column
y = dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)    

# Visualisation

# valeurs x_age de test

plt.hist(x_test[:,3])
plt.title("x_age de test")
plt.show()

# avec normalisation 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)

# from sklearn.preprocessing import MinMaxScaler

# sc = MinMaxScaler()

# x_train = sc.fit_transform(x_train)

# x_test = sc.fit_transform(x_test)


# https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-tutor3-python-scikit.pdf

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


model = DecisionTreeClassifier()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Tree Classifier accuracy :",accuracy_score(y_test,y_pred))



from sklearn.neighbors import KNeighborsClassifier

voisin = 5

neigh = KNeighborsClassifier(n_neighbors=voisin)

neigh.fit(x_train, y_train)

y_pred = neigh.predict(x_test)

print("KNN-"+str(voisin)+ " accuracy :", accuracy_score(y_test,y_pred)) 





from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(x)








