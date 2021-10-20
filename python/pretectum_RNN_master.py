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

## Load Data
znn_acts = pd.read_csv('znn_acts.csv', header=None) # create pandas dataframe of recorded znn calcium activities
if with_stimuli == True:
    stimcourse = pd.read_csv('stimcourse.csv', header=None) # create pandas dataframe of stimuli metadata (which stimuli when). Row order: LR,LU,LL,LD,RR,RU,RL,RD 

## Model Hyperparameters (exogenous to data)
learn = 1 # what does this do?
nRunTot = 1500 # number of steps (what is a step? an epoch?)
amp_rgc = 0.025 # scale/amplitude of RGC inputs; Andalman et al. recommended 10x larger than noise amplitude
# something about a control parameter here...
g = 1.25;   # Peak synaptic conductance constant; Andalman used 1.2-1.3. Elsewhere 1.0-1.5 recommended. (Where?)
dtData = 0.5; # Frame rate for experimental biological data in seconds
dtModel = 0.25; # Integration step rate in seconds. Andalman et al. used 0.25
tau = 2.5; # Neuron membrane time constants; Andalman et al. used 2.5
P0 = 0.1; # What is this? Whit says "free parameter to learning rule for J". Sussillo & Abbott advise 0.01-1

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

for i in np.range(len(real_times)):
    tModelSample, iModelSample[i] = np.min(np.abs(real_times[i] - model_times))
'''WTF is this...
for i=1:length(real_times)
    [tModelSample, iModelSample(i)] = min(abs(real_times(i)-t));
end
'''

stdevData = np.std(znn_acts.to_numpy().flatten()) # why is this computed all the way up here?!

# White Noise
WN_amp = 0.0025 # where does this value come from? ("amplitude no more than 1% of baseline according to Andalman")
WN_input = WN_amp * np.random.randn(N, len(model_times))

# Initialize 
J0 = g * np.random.randn(N, N) / np.sqrt(N) # initialize weight matrix with Gaussian samples scaled by peak conductance constant
R = np.nan(N, len(t)); # rnn (unit) activities
AR = np.nan(N+8, length(t)); #augmented activity for training input wts
JR = np.zeros(N, 1); #product of weights and nonlinear activity
J = J0

%inputs - e.g. LU means Left eye Up stim
[inputLR,inputLU,inputLL,inputLD,inputRR,inputRU,inputRL,inputRD] = deal(zeros(N,length(t)));

%external input weights
[wLR0,wLU0,wLL0,wLD0,wRR0,wRU0,wRL0,wRD0] = deal(randn(N,1)/sqrt(N));
wLR = wLR0;wLU = wLU0;wLL = wLL0;wLD = wLD0;wRR = wRR0;wRU = wRU0;wRL = wRL0;wRD = wRD0;

if (learn)
        PJ = P0 * eye(N+8, N+8);
    end
    
    [data_units, data_steps] = size(Adata);
    fprintf('data_steps: %d\n', data_steps);
    fprintf('dt: %f\n', dtModel);
    fprintf('length(t): %d\n', length(t));
    assert(data_steps >= length(t) * dtModel);

chi2 = zeros(1, nRunTot);
pVars = zeros(1, nRunTot);
for nRun = 1:nRunTot %Number of epochs
    H = Adata(:, 1); %Initialize activities of all neurons with real values at first time point
    R(:, 1) = tanh(H); %nonlinearly transformed activities
    tLearn = 0; %param for when to update J matrix
    iLearn = 1; %Used to index Adata to subtract from model predicted rates in err.
    [epoch_LR,epoch_LU,epoch_LL,epoch_LD,epoch_RR,epoch_RU,epoch_RL,epoch_RD] = deal(0); %set epochs of external inputs to 0
    
    for tt = 2:length(t)-2 %time steps for each epoch. 
        tLearn = tLearn + dt; %update for each time step
        R(:, tt) = tanh(H); %nonlinear transformation of activities
        if tLR(tt) == 1 %conditionals for training external input weights when stim is on
            epoch_LR = 1;
        end
        if tLU(tt) == 1
            epoch_LU = 1;
        end
        if tLL(tt) == 1
            epoch_LL = 1;
        end
        if tLD(tt) == 1
            epoch_LD = 1;
        end
        if tRR(tt) == 1
            epoch_RR = 1;
        end
        if tRU(tt) == 1
            epoch_RU = 1;
        end
        if tRL(tt) == 1
            epoch_RL = 1;
        end
        if tRD(tt) == 1
            epoch_RD = 1;
        end

        JR = J*R(:, tt) + inputN(:, tt); %product of synaptic strength weight matrix and nonlinear activities + gaussian noise
            
        %external inputs at each time step - input on if time vectors
        %are 1 at time tt.
        inputLR(:, tt) = amp_rgc*tLR(tt).*wLR;
        inputLU(:, tt) = amp_rgc*tLU(tt).*wLU;
        inputLL(:, tt) = amp_rgc*tLL(tt).*wLL;
        inputLD(:, tt) = amp_rgc*tLD(tt).*wLD;
        inputRR(:, tt) = amp_rgc*tRR(tt).*wRR;
        inputRU(:, tt) = amp_rgc*tRU(tt).*wRU;
        inputRL(:, tt) = amp_rgc*tRL(tt).*wRL;
        inputRD(:, tt) = amp_rgc*tRD(tt).*wRD;

        %add external inputs onto JR at each time step tt
            
        JR = JR + inputLR(:, tt) + inputLU(:, tt) + inputLL(:, tt) + inputLD(:, tt) + inputRR(:, tt) + inputRU(:, tt) + inputRL(:, tt) + inputRD(:, tt);
        
        H = H + dtModel*(-H + JR )/tau; %model prediction of calcium activities at each time step for every dt.
        
        if (tLearn >= dtData) %model updates weights if tLearn exceeds dtData. Since dtData = 2*dt, this happens every other time step.
            tLearn = 0; 
            err = JR - Adata(:, iLearn+1); %As in Andalman. Adata has entries every 0.5 s and JR at every 0.25 s.
            meanerr2 = mean(err.^2); %what is displayed as chi2 when training. Use it to assess model convergence.
            chi2(nRun) = chi2(nRun) + meanerr2; %nRun is epoch number so accumulates meanerr2 every update to weight matrix. Want this to decreaase over training
            lastchi2s(iLearn) = meanerr2; %The last meanerr2 for the last weight update during an epoch.
            if ((learn) && (nRun <= nRunTot - nFree) && (nRun > nFreePre)) %nFree and nFreePre are weight freezing parameters. learn is set to 1 at start of code.
                AR = [R(:, tt); epoch_LR; epoch_LU; epoch_LL; epoch_LD; epoch_RR; epoch_RU; epoch_RL; epoch_RD]; %augmented Dyn variable. Each trainable external input is added here.
                %For the next 4 lines you are computing an estimate of the 
                %inverse cross correlation matrix of network rates that
                %are used to scale the extent of the weight update.
                %Further details in Sussillo & Abbott (2009). 
                k = PJ*AR; 
                rPr = AR'*k;
                c = 1.0/(1.0 + rPr);
                PJ = PJ - c*(k*k');
                %Updating external input weights if they are on
                if epoch_LR
                    wLR = wLR - c*err*k(end-7);
                end
                if epoch_LU
                    wLU = wLU - c*err*k(end-6);
                end
                if epoch_LL
                    wLL = wLL - c*err*k(end-5);
                end
                if epoch_LD
                    wLD = wLD - c*err*k(end-4);
                end
                if epoch_RR
                    wRR = wRR - c*err*k(end-3);
                end
                if epoch_RU
                    wRU = wRU - c*err*k(end-2);
                end
                if epoch_RL
                    wRL = wRL - c*err*k(end-1);
                end
                if epoch_RD
                    wRD = wRD - c*err*k(end);
                end
                
                J = J - c*err*k(1:N)'; %update J by err and proportional to inverse cross correlation network rates
                
            end
            iLearn = iLearn + 1; %Set index of Adata to time of next frame
            [epoch_LR,epoch_LU,epoch_LL,epoch_LD,epoch_RR,epoch_RU,epoch_RL,epoch_RD]=deal(0); %Set epochs of external inputs to 0
        end
    end

    %Summary of model fit - pVar means percentage of variance explained
    rModelSample = R(:, iModelSample);
    pVar = 1 - (norm(Adata - rModelSample, 'fro')/(sqrt(N*length(tData))*stdData)).^2;
    pVars(nRun) = pVar;
    fprintf('trial=%d pVar=%f chi2=%f\n', nRun, pVar, chi2(nRun));
end

varData = var(reshape(Adata, N*length(tData_ind), 1));
    chi2 = chi2/(sqrt(N*length(tData_ind))*varData);
    lastchi2s = lastchi2s/(sqrt(N*length(tData_ind))*varData);
    
    %Variables you are saving from the model. You can load these variables from the
    %output in matlab and save them as a different file type.
    save(outputname, 'R', 'N', 't', 'chi2', 'Adata', 'tData', 'tData_ind', 'nRunTot', 'nFree', 'nFreePre', 'data_start_time', 'data_end_time', 'inputN', 'inputLR', 'inputLU', 'inputLL', 'inputLD', 'inputRR', 'inputRU', 'inputRL', 'inputRD', 'J', 'pVars', 'lastchi2s', 'simfilename', 'wLR', 'wLU', 'wLL', 'wLD', 'wRR', 'wRU', 'wRL', 'wRD', '-v7.3');
   
end

## Compute Statistics

## Visualize Outputs and Statistics