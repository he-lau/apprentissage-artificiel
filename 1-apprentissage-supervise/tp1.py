import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('credit_scoring.csv',sep=";")

labels = "Seniority;Home;Time;Age;Marital;Records;Job;Expenses;Income;Assets;Debt;Amount;Price"
labels = labels.split(";")

# iloc[:,:-1] ":" : tout le champ
x = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values

# Visualisation

# valeurs x_age de test

plt.hist(x[:,3])
plt.title("x_age")
plt.show()


# avec normalisation 

from sklearn.preprocessing import StandardScaler

norm = StandardScaler()

x = norm.fit_transform(x)

# from sklearn.preprocessing import MinMaxScaler

# norm = MinMaxScaler()

# x = norm.fit_transform(x)

from sklearn.decomposition import PCA


principal = PCA(n_components=10)

principal.fit(x)

x = principal.transform(x)

# avec le PCA les resultats sont casiment identique



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)    




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









