function  Andalman_RNNTrain   % holdover name from original file - I've modified it several times hence the test9
    simfilename = 'C:/Users/joechoo-choy/Documents/Naumann_Lab/Pretectum_RNNs/fr.mat' %replace with input path
    %outputname ='C:/Users/joechoo-choy/Documents/Naumann_Lab/Pretectum_RNNs/output' %replace with output path
    outputname = 'output'
    learn = 1
    nRunTot = 1500 %number of steps
    nFree = 5
    nFreePre = 0
    %firing_rates = load(simfilename); %df/f works too and is what I used.
    % firing_rates = firing_rates.fr;
    firing_rates = load('fr.mat');
    firing_rates = firing_rates.fr;
    
    if (exist('nFreePre') == 0)
      nFreePre = 0;
    end
    
    %variable values
    
    % control parameter - if positive, ntwk produces spontaneous activity
    % with non-trivial dynamics. if negative, it does not (CS)
    g = 1.25;   % Andalman used between 1.2 and 1.3 for g
    dtData = 0.5; % ? frame rate given in Supplemental Info (CS)
    dt = 0.25; % integration step in sec - according to Andalman
    tau = 2.5; % 2.5s time constant of each unit in the network - according to Andalman
    P0 = 1; % ?free parameter to the learning rule for J (CS)
    tauWN = 2.5; % correlation time of external inputs to network in sec (CS)
    ampIn = 0.0002; % scale of external inputs to network (h_0) (CS)
    N = 321; % #neurons
    nLearn = 50;
    
    %Frames and Time length
    
    data_start_0ndx = 0;
    data_end_0ndx = 1999;
    data_start_time = 0;
    data_end_time = 999;
   
    
    N = double(N);

    tData_ind = (data_start_0ndx+1):(data_end_0ndx+1);  
   
    tData = [0:dtData:1999*dtData];
    
    length(tData)
    
    t = data_start_time:dt:data_end_time;  
    lastchi2s = zeros(length(tData), 1);

    iModelSample = zeros(length(tData), 1);
    for i=1:length(tData)
        [tModelSample, iModelSample(i)] = min(abs(tData(i)-t));
    end
    
    
    
    Adata = firing_rates;
    
    %size = N*length(tData_ind);
    
    stdData = std(reshape(Adata, N*length(tData_ind), 1));

    ampWN = sqrt(tauWN/dt);
    iWN = ampWN*randn(N, length(t));
    inputN = ones(N, length(t));
    
    %I have commented out the inputs used in the previous code as they were
    %specific to Andalman. However, I am planning to modify these inputs to
    %represent retinal ganglion cells in the future. For now, the only
    %input is white noise.
    
    
    %inputC = zeros(N, length(t));
    %inputB = zeros(N, length(t));

    %weights

    %wlHb0 = normrnd(0,1,[1,N]);
    %wrHb0 = normrnd(0,1,[1,N]);
    %wBlue0 = normrnd(0,1, [1,N]);
    %J0 = normrnd(0,1, [N,N]);
    
    
    %weights_chr_l_LHb = wlHb0;
    %weights_chr_r_LHb = wrHb0;
    %weights_blue = wBlue0;
    
    %stim_ndx refers to when stimulation applied?
    %What is stim_mk_vec?
    %stim_ndx_lHb = [0:dt:1999];
    %stim_ndx_rHb = [0:dt:1999];
    
    %stim_lHb_when = stim_mk_vec(stim_ndx_lHb, dtData, dt, data_start_time, data_end_time);
    %stim_rHb_when = stim_mk_vec(stim_ndx_rHb, dtData, dt, data_start_time, data_end_time);
    %stim_blue_when = or(stim_lHb_when, stim_rHb_when);
    %stim_lHb_when = [stim_ndx_lHb, dtData, dt, data_start_time, data_end_time];
    %stim_rHb_when = [stim_ndx_rHb, dtData, dt, data_start_time, data_end_time];
    %stim_blue_when = or(stim_lHb_when, stim_rHb_when);
    
    for tt = 2: length(t)
        inputN(:, tt) = iWN(:, tt) + (inputN(:, tt - 1) - iWN(:, tt))*exp(-(dt/tauWN)); %white noise
    end
    inputN = ampIn*inputN;
    
    %Ttrial = size(Adata, 2);
    J0 = g*randn(N,N)/sqrt(N); %synaptic strengrh matrix
    J = J0;
    R = NaN(N, length(t));
    AR = NaN(N, length(t)); %augmented activity for training input wts
    JR = zeros(N, 1);

    if (learn)
        PJ = P0 * eye(N, N);
    end
    
    [data_units, data_steps] = size(Adata);
    fprintf('data_steps: %d\n', data_steps);
    fprintf('dt: %f\n', dt);
    fprintf('length(t): %d\n', length(t));
    assert(data_steps >= length(t) * dt);

    chi2 = zeros(1, nRunTot);
    pVars = zeros(1, nRunTot);
    for nRun = 1:nRunTot
        H = Adata(:, 1);
        R(:, 1) = tanh(H); %nonlinearly transformed activities
        tLearn = 0;
        iLearn = 1;
        %epoch_blue = 0;
        %epoch_left = 0;
        %epoch_right = 0;
        for tt = 2:length(t)
            tLearn = tLearn + dt;
            R(:, tt) = tanh(H);
            %if stim_blue_when(tt) == 1
            %  epoch_blue = 1;
            %end
            %if stim_lHb_when(tt) == 1
            %  epoch_left = 1;
            %end
            %if stim_rHb_when(tt) == 1
            %  epoch_right = 1;
            %end
            
             
            ampBlue = 0;
            
            %inputB(:, tt) = ampBlue*stim_blue_when(tt).*weights_blue;
            %inputB(:, tt) = ampBlue*stim_blue_when(tt).*weights_blue;
            JR = J*R(:, tt) + inputN(:, tt); 
            
            
            ampChr2 = 0;
            
            %What are units_l_LHb and units_r_LHb?
            %units_l_LHb = ones(1,N);
            %units_r_LHb = ones(1,N);
            
            %if ampChr2 ~= 0
                %inputC(:, tt) = ampChr2 * stim_lHb_when(tt) * units_l_LHb' .* weights_chr_l_LHb;
                %inputC(:, tt) = inputC(:, tt) + ampChr2 * stim_rHb_when(tt) * units_r_LHb' .* weights_chr_r_LHb;
                %inputC(:, tt) = ampChr2 * stim_lHb_when(tt) * weights_chr_l_LHb;
                %inputC(:, tt) = inputC(:, tt) + ampChr2 * stim_rHb_when(tt) * weights_chr_r_LHb;
                %inputD(:, tt) = ampChr2 * stim_rHb_when(tt) * weights_chr_r_LHb;
                %inputC(:, tt) = inputC(:,tt) + inputD(:, tt);
                %JR = JR + inputC(:, tt);
            %end
            H = H + dt*(-H + JR )/tau;
            if (tLearn >= dtData)
                tLearn = 0;
                err = JR - Adata(:, iLearn);
                meanerr2 = mean(err.^2);
                chi2(nRun) = chi2(nRun) + meanerr2;
                lastchi2s(iLearn) = meanerr2;
                if ((learn) && (nRun <= nRunTot - nFree) && (nRun > nFreePre))
                    %AR = [R(:, tt); epoch_left; epoch_right; epoch_blue]; %augmented Dyn variable
                    AR = [R(:, tt)];
                    k = PJ*AR;
                    rPr = AR'*k;
                    c = 1.0/(1.0 + rPr);
                    PJ = PJ - c*(k*k');
                    %if epoch_blue
                    %    weights_blue = weights_blue - c*err*k(end);
                    %end
                    %if ampChr2 ~= 0
                    %    if epoch_left
                    %        weights_chr_l_LHb = weights_chr_l_LHb - c*err*k(end-2);
                    %    end
                    %    if epoch_right
                    %        weights_chr_r_LHb = weights_chr_r_LHb - c*err*k(end-1);
                    %    end
                    %end
                    J = J - c*err*k(1:N)';
                end
                iLearn = iLearn + 1;
                %epoch_blue = 0;
                %epoch_left = 0;
                %epoch_right = 0;
            end
        end
        rModelSample = R(:, iModelSample);
        pVar = 1 - (norm(Adata - rModelSample, 'fro')/(sqrt(N*length(tData))*stdData)).^2;
        pVars(nRun) = pVar;
        fprintf('trial=%d pVar=%f chi2=%f\n', nRun, pVar, chi2(nRun));
    end

    varData = var(reshape(Adata, N*length(tData_ind), 1));
    chi2 = chi2/(sqrt(N*length(tData_ind))*varData);
    lastchi2s = lastchi2s/(sqrt(N*length(tData_ind))*varData);
    %display(stdData)
    %display(J);
    %display(Adata);
    %display(R);
    
    %chi2 is error, J has synaptic strength weights
    
    save(outputname, 'R', 'N', 't', 'chi2', 'Adata', 'tData', 'tData_ind', 'nRunTot', 'nFree', 'nFreePre', 'data_start_time', 'data_end_time', 'inputN', 'J', 'pVars', 'lastchi2s', 'simfilename', '-v7.3');
    %save(outputname, 'R', 'N', 't', 'chi2', 'Adata', 'tData', 'tData_ind', 'nRunTot', 'nFree', 'nFreePre', 'ampChr2', 'ampBlue', 'ampIn', 'data_start_time', 'data_end_time', 'inputN', 'inputC', 'inputB', 'J', 'pVars', 'lastchi2s', 'simfilename', 'weights_blue', 'weights_chr_l_LHb', 'weights_chr_r_LHb', '-v7.3');
    %save(outputname, 'R', 'N', 't', 'chi2', 'Adata', 'tData', 'tData_ind', 'nRunTot', 'nFree', 'nFreePre', 'ampChr2', 'ampBlue', 'ampShock', 'ampIn', 'data_start_time', 'data_end_time', 'inputN', 'inputC', 'inputB', 'J', 'pVars', 'lastchi2s', 'datafile', 'simfilename', 'weights_blue', 'weights_chr_l_LHb', 'weights_chr_r_LHb', '-v7.3');
end
