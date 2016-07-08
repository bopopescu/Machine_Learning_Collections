% runExample.m ------------------------------------------------------------
%
% This script shows one possible approach to identifying solar panels in
% satellite imagery data using machine learning. This demonstrates the
% basic process of machine learning, but is not an optimized algorithm
% (that's where you come it!). You're encouraged to innovate on each step
% in this process from preprocessing, feature extraction, classification,
% and validation.
%
% Author: Kyle Bradbury
% Organization: Duke University Energy Initiative
% Date: 19 January 2016

% -------------------------------------------------------------------------
% Initialize Parameters
% -------------------------------------------------------------------------

% Initialize parameters for training
imageDirectoryTraining  = './train/' ;   % Adjust this directory to point to your image files for training
trainingLabelsDirectory = './solution/' ;% Adjust this directory to point to your training label/solution files
kFolds                  = 10 ;               % Number of folds for cross validation

% Initialize parameters for testing
runTestData             = 0 ; % Switch for determining whether or not the test data is classified
submissionSampleFile    = './sampleSubmissions/sampleSubmission.csv' ; % Adjust this to point to your sample submission file
                                                                       % Since the sample submission file has a complete list of image ids, we can
                                                                       % use that to be our index
imageDirectoryTesting   = './test/' ;    % Adjust this directory to point to your image files for testing
submissionFile          = 'submission.csv' ; % Output csv submission filename

%--------------------------------------------------------------------------
% Evaluate cross-validated performance on training data
%--------------------------------------------------------------------------
% -- Load training data ----------------------------------------------
trainingLabelsRaw = csvread([trainingLabelsDirectory 'train_solution.csv'],1,0) ; % Extract the class labels
trainingLabels = sortrows(trainingLabelsRaw,1) ; % In case their not in ascending order, sort them
trainingLabels = trainingLabels(:,2) ; % Remove the id since they are now in order and therefore can be indexed by their ids
nObservations  = length(trainingLabels) ; % Total number of observations

% Initialize the training data storage vectors
trainingFeatures = nan(nObservations,2) ; % You'll want to adjust this as you adjust the number of features
trainingDecision = nan(nObservations,1) ;
trainingScores   = nan(nObservations,1) ;

% -- Extract feature vectors -----------------------------------------
for iObservation = 1:nObservations
    % Load each image
    cImage = imread(sprintf('%simage%g.tif',imageDirectoryTraining,iObservation),'tif') ;
    
    % Extract features from the image
    cVectorImage = double(cImage(:)) ;
    trainingFeatures(iObservation,:) = [mean(cVectorImage) std(cVectorImage)] ;
end

% -- Conduct cross-validated training of the data --------------------
partition = cvpartition(nObservations,'KFold',kFolds) ; % This creates a cvpartition class object for the cross validation

for iFold = 1:kFolds % For each fold of the partition
    % Get the indices of the current partition
    iTrain = partition.training(iFold) ;
    iTest  = partition.test(iFold) ;
    
    % Extract the current partition
    cTrainingFeatures = trainingFeatures(iTrain,:) ;
    cTrainingLabels   = trainingLabels(iTrain) ;
    cTestFeature      = trainingFeatures(iTest,:) ;
    
    % Train the classifier on the training data
    trainedModel = fitcknn(cTrainingFeatures,cTrainingLabels,...
                            'NumNeighbors',5) ;
    
    % Test the trained classifier on the test data
    [cTestLabels,cTestScores] = predict(trainedModel,cTestFeature) ;
    
    % Store the decisions and scores
    trainingDecision(iTest) = cTestLabels ;
    trainingScores(iTest)   = cTestScores(:,2) ;
end

% -- Compute and display ROC curve -----------------------------------
positiveClass = 1 ; % This is the "positive class" or the "target" class
[xRoc,yRoc,thresholdRoc,aucRoc] = perfcurve(trainingLabels,trainingScores,1) ;
h(1) = plot([0 1],[0 1],'k--') ; hold on ; % Plot the chance diagonal
h(2) = plot(xRoc,yRoc,'r-') ; % Plot the ROC
xlabel('Probability of False Alarm')
ylabel('Probability of Correct Detection')
legend(h,'Chance Diagonal','Performance',...
        'Location','southeast')
title(sprintf('ROC Curve (AUC = %g)',aucRoc)) ;


%--------------------------------------------------------------------------
% Train classifier on training data and run on test data to produce results
% for submission to Kaggle
%--------------------------------------------------------------------------
if runTestData % This switch is adjusted in the initial parameters

% -- Load test data and initialize parameters ------------------------
testIds = csvread(submissionSampleFile,1,0) ; % Extract the class labels
testIds = sort(testIds(:,1)) ; % In case their not in ascending order, sort them
nTestObservations = length(testIds) ;
testingFeatures   = nan(nTestObservations,2) ;

% -- Extract feature vectors -----------------------------------------
for iObservation = 1:nTestObservations
    % Load each image
    cImage = imread(sprintf('%simage%g.tif',imageDirectoryTesting,testIds(iObservation)),'tif') ;
    
    % Extract features from the image
    cVectorImage = double(cImage(:)) ;
    testingFeatures(iObservation,:) = [mean(cVectorImage) std(cVectorImage)] ;
end

% Train the classifier on all of the training data
trainedModel = fitcknn(trainingFeatures,trainingLabels,'NumNeighbors',5) ;

% Run the classifier on the test data
[testLabels,testScores] = predict(trainedModel,testingFeatures) ;
testScores = testScores(:,2) ; % The second column generally has the test scores for the target class ;

% Output the results to csv file
fid = fopen(submissionFile, 'w') ;
fprintf(fid,'%s,%s\n','id','class') ;
fclose(fid) ;

output = [testIds testScores] ;
dlmwrite(submissionFile, output, '-append') ;

end