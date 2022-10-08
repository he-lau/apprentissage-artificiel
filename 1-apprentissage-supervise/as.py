import numpy as np

# python 2
# np.set_printoptions(threshold=np.nan)

# python 3
# https://stackoverflow.com/questions/55258882/threshold-must-be-numeric-and-non-nan-when-printing-numpy-array-why-numpy-nan
np.set_printoptions(threshold=np.inf, linewidth=np.nan)

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


for i in range (0,len(df.values)):
    
    split = df.values[i][0].split(";")

    status.append(split[13])

    split.pop(13)

    caracteristiques.append(split)



print("-------------- status --------------")

status = np.array(status)

print("size : " + str(status.size))
print("shape : " + str(status.shape))


print("-------------- variables caractéristiques --------------")

caracteristiques = np.array(caracteristiques)

print("size : " + str(caracteristiques.size))
print("shape : " + str(caracteristiques.shape))


print(caracteristiques)


age = []

for i in caracteristiques :
    age.append(i[3]) 

# age.sort()
print(age)

plt.style.use('ggplot')

plt.hist(status) 

plt.show()