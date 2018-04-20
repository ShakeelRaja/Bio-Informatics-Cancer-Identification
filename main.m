%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        A Comparison of Multilayer Perceptron and Support Vector         %
%                     Machine for Oncological Data                        %
%                                                                         %
%                  Saman Sadeghi Afgeh and Shakeel Raja                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This main.m file contains the training, optimization, evaluation and    %
% visualisations for MLP and SVM. The code program has been written in    %
% a number 0f sections as below                                           %
%                                                                         %
% SECTION 1 - load data and preprocess                                    %
% SECTION 2 - MLP                                                         %
% SECTION 3 - MLP - MODEL optimization (a,b)                              %
% SECTION 4 - SVM                                                         %
% SECTION 5 - SVM - MODEL optimization                                    %
%                                                                         %
% SECTION 2 and 3 use MultiLayerPerceptron.m for training the MLP.        %
% SECTION 4 and 5 use SupportVectorMachine.m for training the SVM.        %
%                                                                         %                  
% note: kindly read readme.txt for further details on used files and      %
% functions. The optimisation sections 3 and 4 may take a few minutes     %
% for a complete run.                                                     %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%                SECTION 1 - load data and preprocess  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all;close all; 
addpath(genpath('func\')); % Add path for external functions
cData = csvread('BreastCancerData_Clean.csv', 1, 0);
rng('default') % for reproducability

% Create Features and Labels from dataset
Features = cData(:,[2:end]);
Labels = cData(:,1);

% Apply Scaling to continuous data using StatisticalNormaliz.m from Neural
% Computing tutorial session 4.This function normalizes each column of an 
% array to values between 0 - 1
sFeatures = StatisticalNormaliz(Features,'scaling');
X = sFeatures; Y= Labels;

% Split Train , Test data (80/20) with random permutation for hold out
% validation
num_points = size(X,1);
split_point = round(num_points*0.8);
seq = randperm(num_points);
X_train = X(seq(1:split_point),:);
Y_train = Y(seq(1:split_point),:);
X_test = X(seq(split_point+1:end),:);
Y_test = Y(seq(split_point+1:end),: );

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%                SECTION 2 - MULTI-LAYER PERCEPTRON 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Load and transpose training examples and labels 
x = X_train'; t = Y_train';

% Following values for hyper-parameters have been dervied from running grid
% based optimization presented in SECTION 3
numNeurons =10;                  % Number of hidden layer neurons
learnRate = .1                   % Learning rate parameter
wtDecay = 0.3                    % Weight decay
momentum = 0.4                   % Momentum

% Gradient descent with momentum and adaptive learning rate backpropagation
trainFcn = 'traingdx'; 
net = patternnet(numNeurons, trainFcn);     % set network type
 
net.trainParam.showCommandLine = false; 
net.trainParam.showWindow= false ;       
net.trainParam.lr = learnRate ;             % Learning Rate
net.trainParam.lr_dec = wtDecay;            % Weight decay
net.trainParam.mc = momentum;               % Momentum constant
net.trainParam.max_fail = 10;                % Choose a Performance Function
net.performFcn = 'mse';                     % mean squared eror
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                'plotconfusion', 'plotroc'};
% Send data with hyper-paramter values to training and validation function
% MultiLayerPerceptron.m  which returns training results with a trained
% model 'net'. 
acc = []
for i=1:1000
tic
[net,tr] = train(net,x,t);
eTime = toc;
% Test the returned model with held out data and calculate performance and
% error
y = net(X_test');  
e = gsubtract(Y_test', y);
error = mean(e);
performance = perform(net,Y_test',y);
%sprintf('Test Performance: %f \performance', performance);
%sprintf('Test Error: %f \error', error);

% Check for %age accuracy for later comparisons. 
% Draw confusion matrix and ROC curve for examination.
label = (round(y))';
target = Y_test;

%figure, plotconfusion(label',target')
%figure, plotroc(label',target')

% Percentage correct
percentage_correct = 100*(1-(sum(label~=target)/length(label)));
acc = [acc; performance percentage_correct eTime];

end
Test_Performace = mean(acc(:,1))
Average_Accuracy = mean(acc(:,2))
Average_Time = mean(acc(:,3))
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%                SECTION 3(a) - MLP - MODEL optimization 
% Optimizing number of neurons and learning rate against validation 
% performance and time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = X_train'; t = Y_train';

% Define arrays with numNeurons and learnRate values to run the grid 
numNeurons =[ 1, 5, 10, 15, 20, 25 , 30];    % Number of hidden layer neurons
learnRate = linspace (0.01, 0.1, 10)         % Learning rate parameter
wtDecay = 0.5                                % Weight decay - constant 
momentum = 0.5                               % Momentum - constant
resultMLP = []

%Run nested For loops iterating through numNeurons and learnRate
for j=1:size(numNeurons, 2)
    for k=1:size(learnRate, 2)
        tic
        [~, result] = MultiLayerPerceptron(x, t, numNeurons(j),...
                      learnRate(k),wtDecay, momentum)
        eTime = toc  % time taken for training
        %Save results for analysis
        resultMLP = [resultMLP; numNeurons(j) learnRate(k) result eTime] ;
    end
end
% DRAW THE GRAPHS
% Use this to draw the grid against time elapsed: 
% fig_table = resultMLP(:,[1 2 7])
% Drawing Grid for performance optimization 
fig_table = resultMLP(:,[1 2 5] )
figure
N=50;
x = 2*pi*rand(N,1);
y = 2*pi*rand(N,1);
z = sin(x).*sin(y);
matrix = [x y z];
tri = delaunay(fig_table(:,1),fig_table(:,2));
colormap jet(100) 
trisurf(tri,fig_table(:,1),fig_table(:,2), fig_table(:,3))
shading interp
hold on  
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%                SECTION 3(b) - MLP - MODEL optimization 
%   Optimizing learning rate decay and momentum for performance and time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = X_train'; t = Y_train';

% Use optimal values for numNeurons and learnRate from 3(a) and set array
% with wtDecay and momentum values to run the grid.

numNeurons =10;                      % Number of hidden layer neurons
learnRate = 0.1;                     % Learning rate paramet
wtDecay = linspace(0.1,1,10);        % Weight decay
momentum = linspace(0,1,10);         % Momentum
resultMLP = [];

%Run the for loops iterating through wtDecay and momentum 
for j=1:size(wtDecay, 2)    
    for k=1:size(momentum, 2)
        tic
        [~, result] = MultiLayerPerceptron(x, t, numNeurons,...
                      learnRate, wtDecay(j), momentum(k));
        eTime = toc
        %Save results for analysis
        resultMLP = [resultMLP; wtDecay(j), momentum(k) result eTime] ;
    end
end
% DRAW THE GRAPHS
% Use this to draw the grid against time elapsed: 
% fig_table = resultMLP(:,[1 2 7])
% Drawing Grid for performance optimization 
fig_table = resultMLP(:,[1 2 5])
figure
N=50;
x = 2*pi*rand(N,1);
y = 2*pi*rand(N,1);
z = sin(x).*sin(y);
matrix = [x y z];
tri = delaunay(fig_table(:,1),fig_table(:,2));
colormap autumn(100) 
colormap(flipud(colormap)) 
trisurf(tri,fig_table(:,1),fig_table(:,2), fig_table(:,3))
shading interp

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%                SECTION 4 - SUPPORT VECTOR MACHINES

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
x = X_train; t = Y_train;
Y_test_neg = 2*Y_test-1;
  
kernel='gaussian' ;         % Kernel parameter
kernelSigma = 1.7556;       % Kernel Sigma
boxConstraint = 3.4;        % Box constraint
tic   
%train SVM model with above parameters
        [SVMModel, validationAccuracySVM] = SupportVectorMachine(x, t,...
                                       kernel, kernelSigma, boxConstraint);
        eTime = toc
        %Predict with test data
        [label,score] = predict(SVMModel,X_test);
        %Calculate confusion matrix
        [C, order] = confusionmat(label,Y_test_neg); 
        C
        %percentage error
        percentage_correct = 100*(1-(sum(label~=Y_test_neg)/length(label)))
        eTime
        label(label==-1) = 0;
        plotconfusion(label',Y_test')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %                SECTION 5 - SVM - MODEL optimization 
% Optimizing kernelSigma and boxConstraint against validation performance 
% and time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = X_train; t = Y_train;
  
rng('default')
 
% Define intervals for parameters to perform grid search on
kernel={'linear', 'gaussian'};                % Set Kernel functions
kernelSigma = linspace(0.1,15,10);            % Set Kernel Sigma interval
boxConstraint = linspace(0.1,10,10);          % Set Box Constraint interval
resultsSVM=[];
ks=1;
  
%Run grid search - if kernel is Gaussian, search over Box Constraint and Kernel Sigma,
%otherwise only over Box Constraint (saving all values, including computing
%time) to resultsSVM for later access
for i=1:size(kernel,2)
    if i==2;       
            for ks = 1:size(kernelSigma,2)               
                for bc = 1:size(boxConstraint,2)
                    tic
                    [~, validationAccuracySVM] = SupportVectorMachine...
                    (x, t, kernel{i}, kernelSigma(ks), boxConstraint(bc));
                    time=toc
                    resultsSVM = [resultsSVM; i kernelSigma(ks) boxConstraint(bc) validationAccuracySVM time];
                end
            end           
    else 
        kernelSigma(ks)=1;    
        for bc = 1:size(boxConstraint,2)
                tic
                [~, validationAccuracySVM] = SupportVectorMachine...
                    (x, t, kernel{i}, kernelSigma(ks), boxConstraint(bc));
                time=toc
                    resultsSVM = [resultsSVM; i kernelSigma(ks) boxConstraint(bc) validationAccuracySVM time];
        end 
    end
end
%%
%Plot accuracy (1-classification error) for different kernels as we change C
%Split x and y for each kernel function
xLin=resultsSVM(1:10,3); , yLin=resultsSVM(1:10,4);
xGausNoKS=resultsSVM(11:20,3); , yGausNoKS=resultsSVM(11:20,4);
xGausBestKS=resultsSVM(21:30,3); , yGausBestKS=resultsSVM(21:30,4);
%Plot line graph
figure
set(gca,'fontsize',11)
plotSVM=plot(xLin,yLin,  xGausNoKS,yGausNoKS, xGausBestKS, yGausBestKS,'LineWidth', 1.5);     
title('SVM Performance grid - Adjusting Box Constraint', 'FontSize', 11);
xlabel('Box constraint', 'FontSize', 11);
ylabel('Accuracy (1-Classification Error)', 'FontSize', 11);
 
%%
% Plot computing time for SVM 
xTimeLin=resultsSVM(1:10,3); , yTimeLin=resultsSVM(1:10,5);
xTimeGausNoKS=resultsSVM(11:20,3); , yTimeGausNoKS=resultsSVM(11:20,5);
figure
set(gca,'fontsize',11)
plotTimeSVM=plot(xTimeLin,yTimeLin,  xTimeGausNoKS,yTimeGausNoKS, 'LineWidth', 1.5);     
title('SVM Performance grid - Computation time', 'FontSize', 11);
xlabel('Box constraint', 'FontSize', 11);
ylabel('Computation time', 'FontSize', 11);
 
%%
% Drawing the accuracy (1- Classification Error) grid for gaussian SVM as
% we change C and Kernel Sigma
fig_table = resultsSVM(11:110,2:4);
figure
N=50;
x = 2*pi*rand(N,1);
y = 2*pi*rand(N,1);
z = sin(x).*sin(y);
matrix = [x y z];
tri = delaunay(fig_table(:,1),fig_table(:,2));
trisurf(tri,fig_table(:,1),fig_table(:,2), fig_table(:,3))
shading interp
title('SVM Performance grid - Gaussian Kernel');
xlabel('Kernel Sigma');
ylabel('Box Constraint');
zlabel('Validation Accuracy (1-Classification Error)');
  
%%
%Identify index number of model with maximum accuracy (1-Classification
%Error). This is the best model, that we test on holdout test set
[M,I] = max(resultsSVM(:,4))
  
