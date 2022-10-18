# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1 - Réduction de dimensions et Visualisation des données

data = pd.read_csv('./villes.csv', sep=';')

X = data.iloc[:, 1:13].values
labels = data.iloc[:, 0].values

from sklearn.preprocessing import StandardScaler

norm = StandardScaler()

X_norm = norm.fit_transform(X)

from sklearn.decomposition import PCA

# 32*12*0.7 = 268.8, 32*10 = 320, 32*9 = 288, 

# principal = PCA(.70)

principal = PCA(n_components=9)

principal.fit(X_norm)

X_pca = principal.transform(X_norm)


plt.hist(X_pca[:,0])
plt.hist(X_pca[:,1])
plt.title("X_pca_jan_fev")
plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1])

for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')

plt.show()


data_crime = pd.read_csv('./crimes.csv', sep=';')

X_crime = data.iloc[:, 1:8].values
labels_crime = data_crime.iloc[:, 0].values

from sklearn.preprocessing import StandardScaler

X_norm_crime = norm.fit_transform(X_crime)



from sklearn.decomposition import PCA

# 50*8*0.7 = 280, 280/50 = 5.6

# principal = PCA(.70)

principal = PCA(n_components=6)

principal.fit(X_norm_crime)

X_pca_crime = principal.transform(X_norm_crime)


plt.scatter(X_pca_crime[:, 0], X_pca_crime[:, 1])

for label, x, y in zip(labels_crime, X_pca_crime[:, 0], X_pca_crime[:, 1]):
    
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')

plt.show()




from sklearn.cluster import KMeans
import matplotlib.colors as mcolors


kmeans = KMeans(n_clusters=3, random_state=0).fit(X_pca)

# kmeans.labels_ : ndarray of shape (n_samples,) Labels of each point

colors = ['red','yellow','blue','pink']

plt.scatter(X_pca[:, 0], X_pca[:, 1], c= kmeans.labels_, cmap=mcolors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
    
plt.title("X_pca_kmeans-3")    
plt.show()



from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(linkage="ward").fit(X_pca).labels_


plt.scatter(X_pca[:, 0], X_pca[:, 1], c= clustering, cmap=mcolors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
    
plt.title("X_pca_agglomerative_ward")    
plt.show()

clustering = AgglomerativeClustering(linkage="average").fit(X_pca).labels_


plt.scatter(X_pca[:, 0], X_pca[:, 1], c= clustering, cmap=mcolors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
    
plt.title("X_pca_agglomerative_average")    
plt.show()

from sklearn.metrics import silhouette_score

score = silhouette_score(X_pca, kmeans.labels_)

print('Silhouetter Score test : %.3f' % score)


from sklearn import metrics

for i in np.arange(2, 6):
    clustering = KMeans(n_clusters=i).fit_predict(X)    
    print("Kmean-"+str(i)+" : "+ str(metrics.silhouette_score(X, clustering,metric='euclidean')))


# La meilleure partition est 2



