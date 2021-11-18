# Pretectum RNN Master
"""
Terminology: 
    Biological data/vars labelled with "znn". Biological NEURONS, with activities.
    In silico model data/vars labelled with "rnn". Artificial UNITS, with activities.
    Stimuli: we presented moving gratings to the zebrafish. 
        Gratings were presented to each eye independently, in one of four orientations: right, up, left, or down.
        The timecourse of stimuli presented to the fish is one-hot encoded in a matrix loaded from 'stimcourse.csv'.
"""

## Setup Libraries/Packages
from datetime import date
import pandas as pd
import numpy as np
np.random.seed(2021)

## Variable Model Settings
test_mode = False
visualize = True
with_stimuli = True
freeze_weights = False # wtf is the freezing about? figure out
nFree = 5 # epochs to freeze learning before final epoch. what does this mean? figure out
nFreePre = 0  # how many epochs from first one that learning is frozen
num_of_inputs = 8

## Load Data
znn_acts = pd.read_csv('znn_acts.csv', header=None) # create pandas dataframe of recorded znn calcium activities
if with_stimuli == True:
    stim_course = pd.read_csv('stimcourse.csv', header=None) # create pandas dataframe of stimuli metadata (which stimuli when). Row order: LR,LU,LL,LD,RR,RU,RL,RD 

## Model Hyperparameters (exogenous to data)
learn = True # what does this do?
nRunTot = 1500 # number of steps (what is a step? an epoch?) [maybe this should be called number of runs]
amp_rgc = 0.025 # scale/amplitude of RGC inputs; Andalman et al. recommended 10x larger than noise amplitude
# something about a control parameter here...
g = 1.25   # Peak synaptic conductance constant; Andalman used 1.2-1.3. Elsewhere 1.0-1.5 recommended. (Where?)
dtData = 0.5 # Frame rate for experimental biological data in seconds
dtModel = 0.25 # Integration step rate in seconds. Andalman et al. used 0.25
tau = 2.5 # Neuron membrane time constants; Andalman et al. used 2.5
P0 = 0.1 # What is this? Whit says "free parameter to learning rule for J". Sussillo & Abbott advise 0.01-1

## Model Parameters (endogenous to data)
N = znn_acts.shape[0] # Number of znn neurons (number of rows in znn_acts); also, set N to be a double or floating point or whatever (why? I don't know)
data_start_time, real_start_time, model_start_time = np.zeros(3)   
data_end_time = znn_acts.shape[1] # number of timepoints in data (number of activity measurements for each neuron)
real_end_time = data_end_time * dtData  # number of seconds in data (how long were the data collected in real time in seconds)
model_end_time = real_end_time / dtModel # number of model timepoints (model-time is higher resolution! => more integration steps than real or data timepoints)

# create vectors of timeseries for real world, data, and model
real_times = np.arange(real_start_time, real_end_time, 1)
data_times = np.arange(real_start_time, real_end_time, dtData)
model_times = np.arange(real_start_time, real_end_time, dtModel)

## Train Model (see <visualize> variable in Settings ^^^)
'''
Things to keep track of: loss/error/chi2, updates, pVar, ...?
'''
'''
def fit(nFreePre=0):
    outputname = 'rnn_output_' + str(date.today())
    pass
'''

lastchi2s = np.zeros(len(real_times))
iModelSample = np.zeros(len(real_times))

for i in range(len(real_times)):
    tModelSample = np.min(np.abs(model_times - real_times[i]))     # WHAT IS THIS DOING?
    iModelSample[i] = np.min(np.abs(model_times - real_times[i]))  # WHAT IS THIS DOING?
'''WTF is this...
for i=1:length(real_times)
    [tModelSample, iModelSample(i)] = min(abs(real_times(i)-t));
end
'''

stdevData = np.std(znn_acts.to_numpy().flatten()) # why is this computed all the way up here?!

# White Noise
WN_amp = 0.0025 # where does this value come from? ("amplitude no more than 1% of baseline according to Andalman")
WN_input = WN_amp * np.random.randn(N, len(model_times)) # used to be inputN

# Initialize 
#J0 = g * np.random.randn(N, N) / np.sqrt(N)
#J = J0
J = g * np.random.randn(N, N) / np.sqrt(N) # initialize weight matrix with Gaussian samples scaled by peak conductance constant

#################################### UP TO HERE ####################################

R = np.empty((N, len(model_times))) # RNN unit activities *** AM I RIGHT IN USING MODEL TIMES HERE? ***
AR = np.empty((N + num_of_inputs, len(model_times))) # augmented activity for training input wts *** WHAT IS THIS AND WHY +8? ***
JR = np.empty((N, 1)) # product of weights and unit activity

# Initialize external input weights
W_input = np.random.randn(N, num_of_inputs) / np.sqrt(N)

if learn == True:
    PJ = P0 * np.identity(N + num_of_inputs) # I don't think this should be called "PJ" unless it's related to J, but I'm not sure if it is

print('Data steps: ', len(data_times))
print('Data dt: ', dtData)
print('Model dt: ', dtModel)
print('Number of real seconds: ', len(real_times))
print('Number of data timesteps: ', len(data_times))
print('Number of model timesteps: ', len(model_times))

chi2 = np.zeros(nRunTot) # used to be (1, nRunTot), but I think this is simpler
pVars = np.zeros(nRunTot) # used to be (1, nRunTot), but I think this is simpler

for nRun in range(nRunTot): # Epoch number, out of number of epochs [THIS SHOULD MAYBE BE CALLED A RUN?]
    H = znn_acts[0] # Initialize RNN activities with ZNN activities at first time point
    R[:,0] = np.tanh(H) # Nonlinearly transformed activities
    tLearn = 0 # param for when to update J matrix
    iLearn = 1 # Used to index znn_acts to subtract from model predicted rates in err.
    # [epoch_LR,epoch_LU,epoch_LL,epoch_LD,epoch_RR,epoch_RU,epoch_RL,epoch_RD] = deal(0); # set epochs of external inputs to 0
    input_epochs = np.zeros(num_of_inputs) # this replaced the line above

    for tt in range(1, len(model_times)): # time steps for each epoch; used to be for tt = 2:len(t)-2
        tLearn = tLearn + dtModel # update for each time step THIS IS JUST USED TO COUNT WHETHER THE TIMESTEP IS ONE WHERE AN ERROR CAN BE COMPUTED OR NOT
        R[:, tt] = np.tanh(H) # nonlinear transformation of activities

        for i in range(num_of_inputs):
            if stim_course.iloc[i,tt] == 1: # conditionals for training external input weights when stim is on
                input_epochs[i] == 1
        
        # generate external inputs at each time step - input on if time vectors are 1 at time tt
        inputs = np.empty((num_of_inputs, N, len(model_times)))
        for i in range(inputs.shape[0]):
            for j in range(N):
                print('*** TESTING ***')
                print('i ', i)
                print('j ', j)
                print('tt ', tt)
                print('amp_rgc ', amp_rgc)
                print('stim_course.iloc[i,tt] ', stim_course.iloc[i,tt])
                print('W_input[i] ', W_input[i])
                print(inputs.shape)
                print(W_input.shape)
                inputs[i,j,tt] = amp_rgc * stim_course.iloc[i,tt] * W_input[i]

        # Update RNN unit activities: multiply activities by weights, add noise, add inputs *** I DON'T THINK THESE SHOULD BE CALLED "JR"... THEY SHOULD JUST BE CALLED "R" ***
        JR = J @ R[:,tt]
        JR = JR + WN_input[:,tt]
        JR = JR + inputs[:,:,tt] # Is this adding these in the right way?

        H = H + (dtModel * (-H + JR)) / tau; # model prediction of calcium activities at each time step
        
        if tLearn >= dtData: # model updates weights if tLearn exceeds dtData. Since dtData = 2*dt, this happens every other time step.
            tLearn = 0
            err = JR - znn_acts[:, iLearn+1] # As in Andalman. znn_acts has entries every 0.5 s and JR at every 0.25 s.
            meanerr2 = np.mean(np.power(err, 2)) # what is displayed as chi2 when training. Use it to assess model convergence.
            chi2[nRun] = chi2[nRun] + meanerr2 # nRun is epoch number so accumulates meanerr2 every update to weight matrix. Want this to decreaase over training
            lastchi2s[iLearn] = meanerr2 # The last meanerr2 for the last weight update during an epoch.
            if (learn == True) and (nRun <= nRunTot - nFree) and (nRun > nFreePre): # nFree and nFreePre are weight freezing parameters. learn is set to 1 at start of code.
                # augmented Dyn variable. Each trainable external input is added here.
                AR = R[:,tt]            
                for i in range(num_of_inputs):
                    AR = np.stack(AR, input_epochs[i])

                # compute estimate of inverse cross correlation matrix of network activities, to scale weight update. See Sussillo & Abbott (2009)
                k = PJ @ AR  
                rPr = np.transpose(AR) @ k
                c = 1.0 / (1.0 + rPr)
                PJ = PJ - c @ (k @ np.transpose(k))

                # Updating external input weights if they are on
                for i in range(num_of_inputs):
                    if input_epochs[i] == 1:
                        W_input[:,i] = W_input[:,i] - c @ err @ k[i-num_of_inputs] # why is k indexed in this way?
                
                J = J - c @ err @ np.transpose(k[:N]) # update J by err and proportional to inverse cross correlation network rates
                
            iLearn = iLearn + 1 # Set index of Adata to time of next frame
            input_epochs = np.zeros(len(input_epochs)) # Set epochs of external inputs to 0

    # Summary of model fit - pVar means percentage of variance explained
    rModelSample = R[:, iModelSample]    
    pVar = 1 - np.power((np.linalg.norm(znn_acts - rModelSample) / (np.sqrt(N * len(data_times)) * stdevData)), 2)
    pVars[nRun] = pVar
    print('Run: {} \n pVar: {} \n chi2: {}'.format(nRun, pVar, chi2[nRun]))

varData = np.var(np.reshape(znn_acts, N * len(data_times)), 1)   # transliterated this directly from the old matlab code. not sure if the dims are all right
chi2 = chi2 / (np.sqrt(N * len(data_times)) * varData)
lastchi2s = lastchi2s / (np.sqrt(N * len(data_times)) * varData)
    
'''
Save the following data:
    R, N, t, chi2, znn_acts(?), tData, tData_ind, nRunTot, nFree, nFreePre, data_start_time, data_end_time, inputN, inputs, J, pVars, lastchi2s, W_input
'''

## Compute Statistics

## Visualize Outputs and Statistics