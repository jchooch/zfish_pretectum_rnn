# testbed.py

import pandas as pd
import datetime
from datetime import date
import time
import numpy as np

### TIME TEST ###

dtData = 0.5
dtModel = 0.25

znn_acts = pd.read_csv('znn_acts.csv', header=None) # create pandas dataframe of recorded znn calcium activities
print('Number of ZNN neurons: ', znn_acts.shape[0])
'''
data_start_time, real_start_time, model_start_time = np.zeros(3)   
data_end_time = znn_acts.shape[1] # number of timepoints in data (number of activity measurements for each neuron)
real_end_time = data_end_time * dtData  # number of seconds in data (how long were the data collected in real time in seconds)
model_end_time = real_end_time / dtModel # number of model timepoints (model-time is higher resolution! => more integration steps than real or data timepoints)

print('Number of timepoints in data: ', data_end_time)
print('Number of timepoints in model: ', model_end_time)
print('Number of seconds in real world: ', real_end_time)

real_times = np.arange(real_start_time, real_end_time, 1)
data_times = np.arange(real_start_time, real_end_time, dtData)
model_times = np.arange(real_start_time, real_end_time, dtModel)

print('Real timeseries: ', real_times[0:20])
print('Length of real timeseries: ', len(real_times))
print('Data timeseries: ', data_times[0:20])
print('Length of data timeseries: ', len(data_times))
print('Model timeseries: ', model_times[0:20])
print('Length of model timeseries: ', len(model_times))

print(real_times[-1], data_times[-1], model_times[-1])

lastchi2s = np.zeros(len(real_times))
print(lastchi2s.shape)
'''

#stdData = std(reshape(znn_acts, N*data_end_time, 1))

J0 = 1.25 * np.random.randn(350, 350) / np.sqrt(350)
print(J0)

''' # removed this from main code. can just add nFreePre as an argument to fit(), default value 0, I think.
try: 
    nFreePre    # checks if nFreePre exists. If not
except NameError:
    nFreePre = 0    # I still need to add a way for nFreePre to possibly be an input (and figure out what it does!)
'''
