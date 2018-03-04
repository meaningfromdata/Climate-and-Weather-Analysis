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

import statsmodels.formula.api as sm


# import datasets

# read CSV file containing historical temperature anomaly data into pandas dataframe
tempAnom_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\Climate Data\\Global_Temperature_Data_File_meanCol1_lowessCol2_addedHeaders.csv')

# read CSV file containing historical CO2 data into pandas dataframe
co2_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\Climate Data\\mole_fraction_of_carbon_dioxide_in_air_0000-2014_simplifiedCols.csv')

# rename 'data_mean_global' column to co2 (more intuitive and shorter) 
co2_df.rename(columns={'data_mean_global':'co2'}, inplace=True)


# plot temperature anomaly timeseries data (raw and smoothed data were both provided by source)
tempAnom_df.plot(x='year', y=['anom_raw', 'anom_lowess'], title='Temperature Anomaly vs Time')

# plot co2 timeseries data 
co2_df.plot(x='year', y='co2', title='CO2 Concentration vs Time')


# inner merge of temperature anomaly and co2 dataframes on year
climate_df = pd.merge(left=tempAnom_df, right=co2_df, how='inner', on='year')


# create double y-axis plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
climate_df.plot(ax=ax1, x='year', y=['anom_raw','anom_lowess'], legend=False)
climate_df.plot(ax=ax2, x='year', y='co2', legend=False, color='g')
ax1.set_ylabel('temp anom')
ax2.set_ylabel('CO2 Conc')
plt.show()



# NEXT STEP:  CREATE GITHUB REPOSITORY; THEN LOOK AT CORRELATION (scatterplot); consider timeseries modeling for forecasting;
# Cross-correlation could be interesting because there are likely lags (delays) in impact of CO2
