# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:05:02 2022

@author: zeroease
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# Create Functions
# get hist graph from df
def histGraph(df,col,title,ptltitle,y1,x1,name):
    df[col].value_counts().plot(kind='bar',title=title,figsize=(16,9))
    plt.title(ptltitle + name + ')')
    plt.ylabel(y1)
    plt.xlabel(x1)
    plt.show()
# Get correlcation from df
def corr(df):
    plt.figure(figsize=(16,6))
    sns.heatmap(df.corr(),vmin=1,vmax=1,annot=True,cmap='BrBG')
    plt.show()

# Import File
df = pd.read_csv('top50.csv', encoding = 'unicode_escape')

# Column Names
print(list(df.columns))

# Rename columns and Set Unnamed column to index
df = df.rename(columns={'Unnamed: 0':'index','Track.Name':'Track_Name','Artist.Name':'Artist_Name', \
 'Beats.Per.Minute':'Beats_Per_Minute','Loudness..dB..':'Loudness_dB','Valence.':'Valence','Length.':'Length', \
 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'})
df = df.set_index('index')

# Column Names
print(list(df.columns))

# Sample Data
df.head()

# Number of Records (columns, rows)
df.shape

# Data Types
df.dtypes
df.info()

# Numeric Description
df.describe(include=[np.number])

# Categorical Description
df.describe(include=['O'])

# Set Aritist Name and Genre to category
df['Genre'] = df['Genre'].astype('category')
df['Artist_Name'] = df['Artist_Name'].astype('category')
# Set int to float
df = df.astype({"Beats_Per_Minute":'float', "Loudness_dB":'float', "Valence":'float', "Length":'float', \
               "Acousticness":'float',"Speechiness":'float'})

# Data Types
df.dtypes

# Null Values
df.isnull().sum()

# Duplicates
df[df.duplicated()].count()

# Unique Count
df.nunique()

# Histograms
histGraph(df,'Beats_Per_Minute','Beats_Per_Minute','df(','frequency','Beats_Per_Minute','Beats_Per_Minute')
histGraph(df,'Energy','Energy','df(','frequency','Energy','Energy')
histGraph(df,'Danceability','Danceability','df(','frequency','Danceability','Danceability')
histGraph(df,'Loudness_dB','Loudness_dB','df(','frequency','Loudness_dB','Loudness_dB')
histGraph(df,'Liveness','Liveness','df(','frequency','Liveness','Liveness')
histGraph(df,'Valence','Valence','df(','frequency','Valence','Valence')
histGraph(df,'Length','Length','df(','Length','Length','Length')
histGraph(df,'Acousticness','Acousticness','df(','frequency','Acousticness','Acousticness')
histGraph(df,'Speechiness','Speechiness','df(','frequency','Speechiness','Speechiness')
histGraph(df,'Popularity','Popularity','df(','frequency','Popularity','Popularity')

# Outliers
# Calculate Outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
IQR

sns.boxplot(x=df['Beats_Per_Minute'])
plt.show()
sns.boxplot(x=df['Energy'])
plt.show()
sns.boxplot(x=df['Danceability'])
plt.show()
sns.boxplot(x=df['Loudness_dB'])
plt.show()
sns.boxplot(x=df['Liveness'])
plt.show()
sns.boxplot(x=df['Valence'])
plt.show()
sns.boxplot(x=df['Length'])
plt.show()
sns.boxplot(x=df['Acousticness'])
plt.show()
sns.boxplot(x=df['Speechiness'])
plt.show()
sns.boxplot(x=df['Popularity'])
plt.show()

# Correlation
corr(df)
df.corr()

# Distubution
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64','int64'])
df_num.head()
df_num.hist(figsize=(16,20),bins=50,xlabelsize=8,ylabelsize=8);
plt.show()

# Plotting Beats_Per_Minute
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Beats_Per_Minute', fontsize=18)
sns.distplot(df['Beats_Per_Minute'])
plt.show()
# Plotting Energy
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Energy', fontsize=18)
sns.distplot(df['Energy'])
plt.show()
# Plotting Danceability
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Danceability', fontsize=18)
sns.distplot(df['Danceability'])
plt.show()
# Plotting Loudness_dB
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Loudness_dB', fontsize=18)
sns.distplot(df['Loudness_dB'])
plt.show()
# Plotting Liveness
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Liveness', fontsize=18)
sns.distplot(df['Liveness'])
plt.show()
# Plotting Valence
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Valence', fontsize=18)
sns.distplot(df['Valence'])
plt.show()
# Plotting Length
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Length', fontsize=18)
sns.distplot(df['Length'])
plt.show()
# Plotting Acousticness
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Acousticness', fontsize=18)
sns.distplot(df['Acousticness'])
plt.show()
# Plotting Speechiness
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Speechiness', fontsize=18)
sns.distplot(df['Speechiness'])
plt.show()
# Plotting Popularity
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Popularity', fontsize=18)
sns.distplot(df['Popularity'])
plt.show()

## Normalize Data - Popularity will be considered our Target Variable
sc = StandardScaler()
num = ['Beats_Per_Minute','Energy','Danceability','Loudness_dB','Liveness','Valence','Length','Acousticness','Speechiness']
df[num] = sc.fit_transform(df[num])
df.head()

# Linear Regression
#X = df[num].values
#y = df['Popularity'].values
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values
print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction of test set
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
# Predicted values
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("Prediction for test set: {}".format(y_pred))

# Intercept and Coefficient
print("Intercept: ", regressor.intercept_)
print("Coefficients:")
list(zip(X, regressor.coef_))

# Actual value and the predicted Value
r_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
r_diff.head()


# Evaluating the Model
# model Evaluation
from sklearn import metrics
meanErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('R squared: {:.2f}'.format(regressor.score(X,y)*100))
print('Mean Absolute Error:', meanErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)










