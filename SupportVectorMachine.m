function [SVMModel, validationAccuracySVM] = SupportVectorMachine(x, t, kernel, kernelSigma, boxConstraint, polOrder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This functions takes in features, labels and configuration parameters as
% input, performs cross validation with SVM and outputs trained model and 
% performance measures. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

resultsSVM = [];            % Initialise empty vector to store results of grid search
predTrainSVM = x;           % Set predictor vector
respTrainSVM = 2*t - 1;     % Change response vector from 0/1 to -1/1
 
%train SVM: if kernel is polynomial, use function with extra argument
%PolynomialOrder, otherwise run function without it
switch kernel   
    case 'linear'
                SVMModel=fitcsvm(...
                predTrainSVM, ...
                respTrainSVM, ...
                'KernelFunction', kernel, ...               
                'BoxConstraint', boxConstraint, ...              
                'Standardize', false, ...
                'ClassNames', [-1; 1]);         
    case 'gaussian'
                SVMModel = fitcsvm(...
                predTrainSVM, ...
                respTrainSVM, ...
                'KernelFunction', kernel, ...
                'KernelScale', kernelSigma, ...               
                'BoxConstraint', boxConstraint, ...              
                'Standardize', false, ...
                'ClassNames', [-1; 1]); 
end
 
%labelSVM=predict(SVMModel, predTestSVM);
partitionedModel = crossval(SVMModel, 'KFold', 10);
% Compute validation accuracy
validationAccuracySVM = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

%Print the out put in console window 
switch kernel
    case 'polynomial'
    fprintf('Kernel function: %s \nKernel scale: %f \nBoxConstraint: %f \nValidation Accuracy: %f\n\n-----------\n\n', kernel, kernelSigma, boxConstraint, validationAccuracySVM);
    case'linear'
    fprintf('Kernel function: %s \nBoxConstraint: %f \nValidation Accuracy: %f\n\n-----------\n\n', kernel, boxConstraint, validationAccuracySVM);
    case 'gaussian'
    fprintf('Kernel function: %s \nKernel scale: %f \nBoxConstraint: %f \nValidation Accuracy: %f\n\n-----------\n\n', kernel, kernelSigma, boxConstraint, validationAccuracySVM);
end    
end