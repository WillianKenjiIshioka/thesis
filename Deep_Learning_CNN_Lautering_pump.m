
load dataset_CNN.mat

%     for i=1:2880
%         input{i}(2,:)=[];
%      end   
%      
%          for i=1:2880
%        input{i}(1,:)=[];
%         end   

numObservations = numel(input);
%% resuzir espectro
      %for i=1:numObservations
      % input{i}(:,251:500)=[];
      %end

    %Ajusta categorias normais e faltosa
 for i=1:numObservations
           %if output(i)==1
           %   output(i)=0;
           %end
           if output(i)==2
              output(i)=0;
          end
   end
    
%% cria amostras randomicas de teste e treinamento
    test_idx = randperm(numel(input), round(numel(input)*.2)); % 20% for test
    train_idx = setdiff(1:numel(input), test_idx); % remaining for training

    XTrain = input(train_idx)';
    XValidation = input(test_idx)';

    TTrain=categorical(output(train_idx))';
    TValidation=categorical(output(test_idx))';



%%Hyper parametros do CNN
filterSize = 5;
numFilters = 32;

numFeatures = size(XTrain{1},1);
numClasses = numel(categories(TTrain));

layers = [ ...
    sequenceInputLayer(numFeatures)
    convolution1dLayer(filterSize,numFilters,Padding="causal")
    reluLayer
    layerNormalizationLayer
    convolution1dLayer(filterSize,2*numFilters,Padding="causal")
    reluLayer
    layerNormalizationLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
     classificationLayer];

miniBatchSize = 20;  %deixar 20

options = trainingOptions("adam", ...
    MaxEpochs=500, ...
    InitialLearnRate=0.01, ...
    SequencePaddingDirection="left", ...
    ValidationData={XValidation,TValidation}, ...
    Plots="training-progress", ...
    Verbose=0);


% treinar CNN como deep learning
net = trainNetwork(XTrain,TTrain, layers,options);


YTest = classify(net,XValidation, ...
    SequencePaddingDirection=="left");

acc = mean(YTest == TValidation)

figure
confusionchart(TValidation,YTest)


%Precision = TP/(TP + FP)
%Recall =  TP /(TP + FN)
