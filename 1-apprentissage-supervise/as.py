import numpy as np

# python 2
# np.set_printoptions(threshold=np.nan)

# python 3
# https://stackoverflow.com/questions/55258882/threshold-must-be-numeric-and-non-nan-when-printing-numpy-array-why-numpy-nan
np.set_printoptions(threshold=np.inf, linewidth=np.nan)
np.set_printoptions(threshold=np.inf)

import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# 1 - Chargement des données et préparation

# Read CSV file into DataFrame df
df = pd.read_csv('credit_scoring.csv')

# Show dataframe
#print(df.values)


print("-------------- source --------------")

source = np.array(df)


print(source[0])
print(source[0][0])
print(source.size)
print(source.shape)

print("Length df : " + str(len(df.values)))
print("Value at 0 df : " + str(df.values[0][0]))
print("Value at 0 df : " + str(len(df.values[0][0])))


status = []
caracteristiques = []
labels = "Seniority;Home;Time;Age;Marital;Records;Job;Expenses;Income;Assets;Debt;Amount;Price"
labels = labels.split(";")


for i in range (0,len(df.values)):
    
    split = df.values[i][0].split(";")

    status.append(split[13])

    split.pop(13)

    caracteristiques.append(split)



print("-------------- status --------------")

status = np.array(status)

print("size : " + str(status.size))
print("shape : " + str(status.shape))

status.reshape(-1,1)


print("-------------- variables caractéristiques --------------")

caracteristiques = np.array(caracteristiques)

print("size : " + str(caracteristiques.size))
print("shape : " + str(caracteristiques.shape))
print("shape : " + str(caracteristiques.shape))


#print(caracteristiques)


age = []

for i in caracteristiques :
    age.append(i[3]) 

# age.sort()

plt.style.use('ggplot')

plt.hist(status) 

# plt.show()


from sklearn.model_selection import train_test_split

np.array(caracteristiques, dtype=float) #  convert using numpy
np.array(status, dtype=float) #  convert using numpy

# split the data into train-test subsets *
x_train, x_test, y_train, y_test = train_test_split(caracteristiques, status, test_size=0.3)


x_train = np.array(x_train, dtype=float) #  convert using numpy

y_train = np.array(y_train, dtype=float) #  convert using numpy

x_test = np.array(x_train, dtype=float) #  convert using numpy

y_test = np.array(y_train, dtype=float) #  convert using numpy

print("-------------- train --------------")

train_df = pd.DataFrame(x_train, columns = labels)
train_df.to_csv('train.csv',index=False)

print(x_train.shape)

print("-------------- test --------------")

test_df = pd.DataFrame(x_test, columns = labels)
test_df.to_csv('test.csv',index=False)
print(x_test.shape)

# 2 - Apprentissage  et  évaluation  de  modèles

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(x_train,y_train)

score = model.score(x_train,y_train)

print("R-squared:", score) 