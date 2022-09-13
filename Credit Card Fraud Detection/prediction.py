# Data set Link
# https://www.kaggle.com/datasets/ealaxi/paysim1

import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions.csv')

# Summary statistics on amount column
print(transactions.info())

# Create isPayment field
transactions['isPayment'] = transactions['type'].apply(lambda x: 1 if x == 'PAYMENT' or x == 'DEBIT' else 0)
# Create isMovement field
transactions['isMovement'] = transactions['type'].apply(lambda x: 1 if x == 'CASH_OUT' or x == 'TRANSFER' else 0)

# Create accountDiff field
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])
 
# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
labels = transactions[['isFraud']]
print(features.head())
print(labels.head())

# Split dataset
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3)
# print(feature_train.shape)
# print(label_train.shape)


# Normalize the features variables
se = StandardScaler()
# reshaped = np.array(feature_train).reshape(-1,1)
feature_train_scaled = se.fit_transform(feature_train)
feature_test_scaled = se.transform(feature_test)

# Fit the model to the training data
lre = LogisticRegression()
lre.fit(feature_train_scaled, label_train)
# Score the model on the training data
print(lre.score(feature_train_scaled, label_train))

print(lre.score(feature_test_scaled, label_test))

print(lre.coef_)

# transaction data given 
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# custom transaction 

transaction4 = np.array([20000.54, 0.0, 1.0, 23000.98])

# Score the model on the test data
sample_transactions = np.array([transaction1, transaction2, transaction3, transaction4])

sample_transactions_scaled = se.fit_transform(sample_transactions)

print(lre.predict(sample_transactions_scaled))

print(lre.predict_proba(sample_transactions_scaled))

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
