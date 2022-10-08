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

status0 = []
status1 = []

print("-------------- source --------------")

test = np.array(df)


print(test[0])
print(test[0][0])
print(test.size)
print(test.shape)

print("Length df : " + str(len(df.values)))
print("Value at 0 df : " + str(df.values[0][0]))
print("Value at 0 df : " + str(len(df.values[0][0])))





for i in range (0,len(df.values)):
    # print(df.values[i][0][len(df.values[i][0])-1])   
    split = df.values[i][0].split(";")

    if df.values[i][0][len(df.values[i][0])-1] == '0' :
        status0.append(split)
    else :
        status1.append(split)

print("-------------- status0 --------------")

status0 = np.array(status0)

print("size : " + str(status0.size))
print("shape : " + str(status0.shape))
print("Premier elem status0 : " + str(status0[0]))

#for i in status0 :
    #print(i)
    #print("\n")

print("-------------- status1 --------------")

status1 = np.array(status1)

print("size : " + str(status1.size))
print("shape : " + str(status1.shape))





status0_age = []
status1_age = []

for i in status0 :
    status0_age.append(i[3]) 

for i in status1 :
    status1_age.append(i[3])     


status0_age.sort()

for i in status0_age :
    print(i)

plt.style.use('ggplot')
plt.hist(status0_age, bins=10) 
plt.title("status0 - Age") 
plt.xlabel("Age")
plt.ylabel("Age Disribution")

plt.show()