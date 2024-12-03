load 'dataset_CNN_wavelet_scattering';


    %
    inputsize = size(XTrain(:,:,1,1));
    
    %Layers definition

    layers = [
    imageInputLayer(inputsize)
    
    convolution2dLayer(3,8,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer];

    options = trainingOptions("sgdm", ...
    InitialLearnRate=0.01, ...
    MaxEpochs=500, ...
    Shuffle="every-epoch", ...
    ValidationData={XValidation,TValidation'}, ...
    ValidationFrequency=30, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);


    net = trainnet(XTrain,TTrain',layers,"crossentropy",options);


    Ytest=predict(net,XValidation);
