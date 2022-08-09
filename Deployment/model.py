import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("D:\stroke deploy\healthcare-dataset-stroke-data.csv")

# Preprocessing
df = df[df['gender'] != 'Other']
df.drop(['id', 'bmi'], axis=1, inplace=True)
df.drop(df.index[[162,245]], inplace=True)

def smoke(text):
    if text == 'never smoked' or text == 'Unknown':
        return 'never smoked'
    else:
        return 'smoke'

df['smoking_status'] = df['smoking_status'].apply(smoke)

le = LabelEncoder()

#get all object features
obj_feat = df.dtypes[df.dtypes == 'O'].index.values
for i in obj_feat:
    df[i] = le.fit_transform(df[i])

X = df.drop('stroke', axis=1)
y = df['stroke']

scaler = StandardScaler()

scaler.fit(X)
X_scaled = scaler.transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# Modeling
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

pickle.dump(dtc, open('model.pkl', 'wb'))