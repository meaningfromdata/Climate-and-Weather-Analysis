# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:27:01 2018

@author: David

EDA of Temperature Anomaly and Greenhouse Gas (CO2, SO2, Methane) Data

Datasets sourced from: https://www.co2.earth/global-warming-update (compiles NOAA datasets)

"""

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# SEE THIS EXAMPLE WITH sklearn for regression:
# https://chrisalbon.com/machine_learning/linear_regression/linear_regression_scikitlearn/

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
import random


# import datasets

# read CSV file containing historical temperature anomaly data into pandas dataframe
tempAnom_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\Climate Data\\Global_Temperature_Data_File_meanCol1_lowessCol2_addedHeaders.csv')


# read CSV file containing historical CO2 data into pandas dataframe
co2_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\Climate Data\\mole_fraction_of_carbon_dioxide_in_air_0000-2014_simplifiedCols.csv')

# rename 'data_mean_global' column to co2 (more intuitive and shorter) 
co2_df.rename(columns={'data_mean_global':'co2'}, inplace=True)


# read CSV file containing historical SO2 data into pandas dataframe
so2_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\Climate Data\\mole_fraction_of_so2f2_in_air_input_0000-2014_simplifiedCols.csv')

# rename 'data_mean_global' column to so2 (more intuitive and shorter) 
so2_df.rename(columns={'data_mean_global':'so2'}, inplace=True)


# read CSV file containing historical methane data into pandas dataframe
methane_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\Climate Data\\mole_fraction_of_methane_in_air_input_0000-2014_simplifiedCols.csv')

# rename 'data_mean_global' column to methane (more intuitive and shorter) 
methane_df.rename(columns={'data_mean_global':'methane'}, inplace=True)







# plot temperature anomaly timeseries data (raw and smoothed data were both provided by source)
tempAnom_df.plot(x='year', y=['anom_raw', 'anom_lowess'], title='Temperature Anomaly vs Time')

# plot co2 timeseries data 
co2_df.plot(x='year', y='co2', title='CO2 Concentration vs Time')

# plot so2 timeseries data 
so2_df.plot(x='year', y='so2', title='SO2 Concentration vs Time')

# plot methane timeseries data 
methane_df.plot(x='year', y='methane', title='Methane Concentration vs Time')



# inner merge of temperature anomaly and co2 dataframes on year to generate new "climate" df
climate_df = pd.merge(left=tempAnom_df, right=co2_df, how='inner', on='year')

# inner merge of climate and so2 dataframes on year
climate_df = pd.merge(left=climate_df, right=so2_df, how='inner', on='year')

# inner merge of climate and methane dataframes on year
climate_df = pd.merge(left=climate_df, right=methane_df, how='inner', on='year')


# create double y-axis plot to show temperature anomaly and co2 on same plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
climate_df.plot(ax=ax1, x='year', y=['anom_raw','anom_lowess'], legend=False)
climate_df.plot(ax=ax2, x='year', y='co2', legend=False, color='g')
ax1.set_ylabel('temp anom')
ax2.set_ylabel('CO2 Conc')
plt.show()


# create double y-axis plot to show temperature anomaly and so2 on same plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
climate_df.plot(ax=ax1, x='year', y=['anom_raw','anom_lowess'], legend=False)
climate_df.plot(ax=ax2, x='year', y='so2', legend=False, color='g')
ax1.set_ylabel('temp anom')
ax2.set_ylabel('SO2 Conc')
plt.show()


# create double y-axis plot to show temperature anomaly and so2 on same plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
climate_df.plot(ax=ax1, x='year', y=['anom_raw','anom_lowess'], legend=False)
climate_df.plot(ax=ax2, x='year', y='methane', legend=False, color='g')
ax1.set_ylabel('temp anom')
ax2.set_ylabel('Methane Conc')
plt.show()


# compute correlation matrix to get overview of how variables are related
climate_corrMat = climate_df.corr()

# close open plots before generating any more figures
plt.close('all')

# generate heatmap of correlations between variables
sns.heatmap(climate_corrMat, annot=True, fmt=".2f")

# generate scatterplots with regression line of temperature anomaly against greenhouse gas concentration 
sns.regplot(x=climate_df['co2'], y=climate_df['anom_raw'], data=climate_df)
sns.regplot(x=climate_df['so2'], y=climate_df['anom_raw'], data=climate_df)
sns.regplot(x=climate_df['methane'], y=climate_df['anom_raw'], data=climate_df)


# Use CO2 as the predictor/independent variable
# Use the unsmoothed (raw) temperature anomaly as our response/dependent variable
X = climate_df['co2']
y = climate_df['anom_raw']

# use train/test split with different random_state values, producing different random splits
X_train, X_test, y_train, y_test=train_test_split(X, y)

# fit OLS model regressing the temperature anomaly onto the CO2 concentration
ols = linear_model.LinearRegression()
model = ols.fit(X_train.values.reshape(-1,1), y_train)
print(model.coef_)
print(model.intercept_)
print(model.score(X_test.values.reshape(-1,1), y_test))

# ten-fold cross-validation
# data needs to be shuffled before cross-validation because data pairs are ordered (because they are from timeseries)
kf = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(model, X.values.reshape(-1,1), y, cv=kf)
print(scores.mean())


