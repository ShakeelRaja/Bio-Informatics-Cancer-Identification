function [net, results]= MultiLayerPerceptron(x,t,numNeurons,learnRate,wtDecay,momentum)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This functions takes in features, labels and configuration parameters as
% input, performs cross validation and outputs trained model and performance
% measures. 
%
% Note: The the inspiration for cross validation has been taken from Gregg
% Heath's solution from mathworks groups. the original idea with code can
% be viewed at:
% http://uk.mathworks.com/matlabcentral/newsreader/view_thread/340857
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results = []
trainFcn = 'traingdx';      % Scaled conjugate gradient backpropagation.
k = 10;
% Read the size of input data for creating folds.
[I N] = size(x);
[O N] = size(t);

% Biased Reference MSE00a is the MSE "a"djusted for the loss in estimation
% degrees of freedom caused by the bias of evaluating the MSE with the same 
% data that was used to build the model. 
MSE00 = mean(var(t',1)); 
MSE00a = mean(var(t'));         % Unbiased Reference

%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('default')                  % for reproducability
ind0 = randperm(N);

M = floor(N/k);    
Ntrn = N-2*M;                   % Length(trnind)
Ntrneq = Ntrn*O;                % No. of training equations
H = 10;
Nw = (I+1)*H+(H+1)*O;           % No. of unknown weights
Ndof = Ntrneq-Nw;               % No. of estimation degrees of freedom
MSEgoal = 0.01*MSE00;	 %  
MinGrad = MSEgoal/100; % 

%Initialize network architecture
net = patternnet(numNeurons, trainFcn);     % Initialise pattern net
net.trainParam.goal = MSEgoal;              % Identify training goal 
net.trainParam.min_grad = MinGrad;          % Minimum performance gradient
net.trainParam.showCommandLine = 0;         % Mute comand line output
net.trainParam.showWindow=0;                % Mute graphical output 
net.trainParam.lr = learnRate ;             % Learning Rate
net.trainParam.lr_dec = wtDecay;            % Weight decay
net.trainParam.mc = momentum;               % Momentum constant
net.trainParam.max_fail = 10;               % Maxium validation failures
net.performFcn = 'mse';                     % MSE - performance function 
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotconfusion', 'plotroc'};

% Setup folds for cross-validation        
cvFolds = crossvalind('Kfold', size(t,2), k);

% Train the Network for all folds.
 for i = 1:k              
        rngstate(i) = rng;
        testIdx = (cvFolds == i);           % Get indices for test examples
        trainIdx = ~testIdx  ;              % Get indices training examples
        trInd=find(trainIdx);               % Find non-zero elements ind and vals 
        tstInd=find(testIdx);               % Find non-zero elements ind and vals 
        net.trainParam.epochs = 100;        % Maximum number of epochs 
        net.divideFcn = 'divideind';        % Divide data for training and validation
        net.divideParam.trainInd=trInd;     % Get training data items
        net.divideParam.testInd=tstInd;     % Get test dat aitems    
        
        valind = 1 + M*(i-1) : M*i;
        
     % If if i is equal to k ( number of folds)   
     if i==k
          tstind = 1:M;
          trnind = [ M+1:M*(k-1) , M*k+1:N ];
     else
          tstind = valind + M;
          trnind = [ 1:valind(1)-1 , tstind(end)+1:N ];
     end
     
        % Get new training and test data and divide for training and
        % validation
        trnInd = ind0(trnind); 
        valInd = ind0(valind); 
        tstInd = ind0(tstind);
        net.divideParam.trainInd = trnInd;
        net.divideParam.valInd = valInd;
        net.divideParam.testInd = tstInd;
        
        
        [net,tr] = train(net,x,t);

        % Test the Network with training data and check performance
        y = net(x);
        e = gsubtract(t,y);
        performance = perform(net,t,y);
        trainTargets = t .* tr.trainMask{1};
        testTargets = t .* tr.testMask{1};
        trainPerformance = perform(net,trainTargets,y)
        testPerformance = perform(net,testTargets,y)
        test(k)=testPerformance;
        save net;                       % Save the trained network
        
        stopcrit{i,1} = tr.stop;
        bestepoch(i,1) = tr.best_epoch;
        R2trn(i,1) = 1 - tr.best_perf/MSE00;
        R2trna(i,1) = 1 - (Ntrneq/Ndof)* tr.best_perf/MSE00a;
        R2val(i,1) = 1 - tr.best_vperf/MSE00;
        R2tst(i,1) = 1 - tr.best_tperf/MSE00;
        
    end

        accuracy=mean(test);
        stopcrit = stopcrit;
        results = [ mean(R2trn) mean(R2trna) mean(R2val) mean(R2tst)];

end