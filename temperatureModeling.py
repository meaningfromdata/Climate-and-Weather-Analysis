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
# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn import metrics
# import random


import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std





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



# fit OLS model in sklearn regressing the temperature anomaly onto the CO2 concentration
ols = linear_model.LinearRegression(fit_intercept=True)
model_sk = ols.fit(X.values.reshape(-1,1), y)
print(model_sk.coef_)
print(model_sk.intercept_)





# fit OLS model in statmodels regressing the temperature anomaly onto the CO2 concentration
# be sure to use add_constant (not adding constant can make big difference to results)
X_sm = sm.add_constant(X)  # add intercept to model (adds column of 1s to feature matrix)
model_sm = sm.OLS(y, X_sm)
results = model_sm.fit()
print(results.summary())
print('Parameters: ', results.params)
print('R2: ', results.rsquared)
print('Standard errors: ', results.bse)
# print('Predicted values: ', results.predict())


# show fit with confidence interval
# adapted from example at http://www.statsmodels.org/dev/examples/notebooks/generated/ols.html


# generating interpolation feature matrix to pass to results.predict for plotting

X_min_floor = np.floor(X.values.min())
X_max_ceiling =  np.ceil(X.values.max())
X_dataRange_Series = pd.Series(np.linspace(X_min_floor, X_max_ceiling, num=(X_max_ceiling-X_min_floor+1))  
const_Series = pd.Series(np.ones(len(X_dataRange)))
X_dataRange_df = pd.DataFrame(dict(const = const_Series, dataRange = X_dataRange))


# get confidence intervals for 
prstd, iv_l, iv_u = wls_prediction_std(results)


# plot data, interpolated linear fit and confidence intervals

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(X.values, y.values, 'o', label="data")  # show the data points
ax.plot(X_dataRange_df['dataRange'], results.predict(X_dataRange_df), 'r-', label="OLS")  
ax.plot(X.values, iv_u, 'r--', label="CI")  # upper confidence interval
ax.plot(X.values, iv_l, 'r--')  # lower confidence interval
ax.legend(loc='best');




# THERE IS DEBATE BUT OVERALL SEEMS CONSENSUS IS THAT THERE IS NO NEED TO DO TEST/TRAIN SPLIT
# OR CROSS-VALIDATION WITH SIMPLE LINEAR REGRESSION 
# BELOW: code to do train/test split and cross-validation of linear regression model in sklearn



# use train/test split with different random_state values, producing different random splits
# X_train, X_test, y_train, y_test=train_test_split(X, y)

# print(model.score(X_test.values.reshape(-1,1), y_test))

# ten-fold cross-validation
# data needs to be shuffled before cross-validation because data pairs are ordered (because they are from timeseries)
# kf = KFold(n_splits=10, shuffle=True)
# scores = cross_val_score(model, X.values.reshape(-1,1), y, cv=kf)
# print(scores.mean())



