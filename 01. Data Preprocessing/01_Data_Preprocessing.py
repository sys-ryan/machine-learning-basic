# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values    # [rows, columns]
y = dataset.iloc[:, -1].values

# Taking care of missing data - data preprocessing tools.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x[:, 1:])           # connect to the object
x[:, 1:] = imputer.transform(x[:, 1:])     # replace missing values according to the strategy
print(x)


# Encoding categorical data - one hot incoding (to avoid numerical order)

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

x = np.array(ct.fit_transform(x))
print(x)

# Encoding Dependant Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


#Splitting the dataset into Training set and Test set
    # split the dataset and then apply feature scaling
    # test set is supposed to be a brand new set
    # test set is not something you're suppose to work with during the learning process
    # * To prevent information leakage
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        #random_state : seed for random variable
print('x_train: ')
print(x_train)
print('x_test: ')
print(x_test)
print('y_train: ')
print(y_train)
print('y_test: ')
print(y_test)


#Feature Scaling
    # 이렇듯 단위가 서로 상이하면 머신러닝의 결과가 잘못될 수 있습니다. 정확한 분석을 위해서는 Feature를 서로 정규화시켜주는 작업이 필요합니다. 이를 Feature Scaling이라고 합니다.
    # Standardisation, Normalization 이 있음.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:]) #apply the same transform used for x_train

'''
'fit' just calculates the mean and standard deviation parameters required for scaling and saves these nos. in the object 'sc' internally

'transform' applies the standardization formula using previously calculated mean and standard deviation on all values

'fit_transform' does it together
we do not use fit_transform on test data because we want to calculate values for test set using std dev and mean of train set as we are predicting dependent values for test set using train set values.
'''

print(x_train)
print(x_test)
