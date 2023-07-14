import numpy as np
import time 
import pandas as pd 
import random 
import torch 
import torch.nn as nn
import torch.utils.data as data_utils
import timesynth as ts
import quantus

"""
The metrics and code included in here is adapted code from: https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark
Belonging to the paper : 

"""

def getRowColMaskIndex(mask,rows,columns):
    InColumn=np.zeros((mask.shape[0],columns),dtype=object)
    InRow=np.zeros((mask.shape[0],rows),dtype=object)
    InColumn[:,:]=False
    InRow[:,:]=False
    for i in range(mask.shape[0]):
        cleanIndex = mask[i,:]
        cleanIndex=cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        for index in range(cleanIndex.shape[0]):
            InColumn[i,cleanIndex[index]%columns]=True
            InRow[i,int(cleanIndex[index]/columns)]=True
    return InRow,

#def getAverageOfMaxX(array,X):
#    index = getIndexOfXhighestFeatures(array,X)
#    avg=np.mean(array[index])
#    return avg

def getIndexOfAllhighestSalientValues(array,percentageArray):
    '''
    Function returns Indicies of the fraction highest salient feature specified in the percentage array.
    Attributes:
        array np.array : array of saliency as in (Time, Feat), One Item
        percentageArray : percentage of features to select
    Returns: 
        indexes: 
    '''
  
    X=array.shape[0]
    index=np.argsort(array)
    totalSaliency=np.sum(array)
    indexes=[]
    #X=1 TODO Used to be one 
    for percentage in percentageArray:
        actualPercentage=percentage/100
        #print(f'actualPercentage {actualPercentage}')        
        index_X=index[int(-1*X):]
        #print(index_X)

        # Clalcuates Saliency Drop 
        percentageDroped=np.sum(array[index_X])/totalSaliency
        #print(f'percentageDroped {percentageDroped}')
        if(percentageDroped<actualPercentage):
            X_=X+1
            index_X_=index[int(-1*X_):]
            percentageDroped_=np.sum(array[index_X])/totalSaliency
            if(not (percentageDroped_>actualPercentage)):
                while(percentageDroped<actualPercentage and X<array.shape[0]-1):
                    X=X+1
                    index_X=index[int(-1*X):]
                    percentageDroped=np.sum(array[index_X])/totalSaliency
        elif(percentageDroped>actualPercentage):
            X_=X-1
            index_X_=index[int(-1*X_):]
            percentageDroped_=np.sum(array[index_X_])/totalSaliency
            if(not (percentageDroped_<actualPercentage)):

                while(percentageDroped>actualPercentage and X>1):
                    X=X-1
                    index_X=index[int(-1*X):]
                    percentageDroped=np.sum(array[index_X])/totalSaliency
        #print(f'index_X to append {index_X_}')
        indexes.append(index_X)
    #import sys 
    #sys.exit(1)
    return indexes



def checkAccuracy(test_loader , model ,shapes):
    '''
    Calculates Accuracy 
    Attributes: 
        test_loader: 
        model: PyTorch model 
        shapes Tuple: Input Shapes
    Returns 
        float: Accuracy 
    '''
    device='cpu'
    
    model.eval()

    correct = 0
    total = 0
    for  (samples, labels)  in test_loader:
        if 'CNN' in str(type(model)):
            images = samples.reshape(-1, shapes[1],shapes[0]).float().to(device)
        else:
            images = samples.reshape(-1, shapes[0],shapes[1]).float().to(device)

        outputs = model(images)        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()

    return  (100 * float(correct) / total)

def checkCFAccracy():
    pass


def getIndexOfXhighestFeatures(array,X):
    '''
    Get X  Highes Features. 
    Attributes: 
        array np.array: 
        X int: number of features to select
    Returns: 

    '''
    #print(array.shape)
    #print(X)
    #print(np.argpartition(array, int(-1*X))[int(-1*X):])
    #import sys 
    #sys.exit(1)
    return np.argpartition(array, int(-1*X))[int(-1*X):]

def generateNewSample(dataGenerationProcess, sampler="irregular", NumTimeSteps=50, NumFeatures=50):
    '''
    Generates a new sample.
    Attributes: 
        dataGenerationProcess str: Type of Generation Process used 
        samples str: Type of sampler irregular or regular 
        NumTimeSteps int: Number Time Steps 
        Num Features int: Number of Features 
    Returns: 
        np.array: new sample (feature, time)
    '''
    dhasNoise=True
    #print('DAtaGenerationProcess',dataGenerationProcess)
    #if(dataGenerationProcess==None):
    #    sample=np.random.normal(0,1,[NumTimeSteps,NumFeatures])

    #else:
    time_sampler = ts.TimeSampler(stop_time=20)
    sample=np.zeros([NumTimeSteps,NumFeatures])


    if(sampler=="regular"):
        # keep_percentage=50 exists only for irregular NumTimeSteps*2
        time = time_sampler.sample_regular_time(num_points=NumTimeSteps)
    else:
        time = time_sampler.sample_irregular_time(num_points=NumTimeSteps*2, keep_percentage=50)

    signal= None
    for  i in range(NumFeatures):
        if(dataGenerationProcess== "Harmonic"):
                signal = ts.signals.Sinusoidal(frequency=2.0)
                
        elif(dataGenerationProcess=="GaussianProcess"):
            signal = ts.signals.GaussianProcess(kernel="Matern", nu=3./2)

        elif(dataGenerationProcess=="PseudoPeriodic"):
            signal = ts.signals.PseudoPeriodic(frequency=2.0, freqSD=0.01, ampSD=0.5)

        elif(dataGenerationProcess=="AutoRegressive"):
            signal = ts.signals.AutoRegressive(ar_param=[0.9])

        elif(dataGenerationProcess=="CAR"):
            signal = ts.signals.CAR(ar_param=0.9, sigma=0.01)

        elif(dataGenerationProcess=="NARMA"):
            signal = ts.signals.NARMA(order=10)
        else: 
            sample=np.random.normal(0,1,[NumTimeSteps,NumFeatures])

        if signal is not None:
            if(dhasNoise):
                noise= ts.noise.GaussianNoise(std=0.3)
                timeseries = ts.TimeSeries(signal, noise_generator=noise)
            else:
                timeseries = ts.TimeSeries(signal)

            feature, signals, errors = timeseries.sample(time)
            sample[:,i]= feature
    return sample

def maskData(data,mask,dataGenerationProcess, sampler,shapes,noise=False):
    '''
    Masks the timesteps / features denoted in the mask 
    Attributes: 
        data : Raw data of shape (num samples, time * feat)
        mask np.array: single mask 
        dataGenerationProcess str: Name of data Generation Process
        sampler:
        shape: Shape of data 
        noise: Noise Sample to be used # TODO --> Currently DAta masked by setting to 0 

    Returns: 
        np.array: masked data of shape (num samples, time * feat)
    '''
    newData= np.zeros((data.shape))
    if(noise):
        noiseSample= generateNewSample(dataGenerationProcess, sampler,NumFeatures=shapes[0], NumTimeSteps=shapes[1])
        noiseSample=noiseSample.reshape(data.shape[1])
    for i in range(mask.shape[0]):
        # Go through all masks 
        newData[i,:]=data[i,:]
        cleanIndex = mask[i,:]
        cleanIndex=cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        if(noise):
            newData[i,cleanIndex]=noiseSample[cleanIndex]
        else:
            newData[i,cleanIndex]=0
    return newData


def find_masks(saliency):
    '''
    Takes the Feature importance maps and calculates the mask dict
    Attributes: 
        saliency np.array : Feature Importance maps in form (len, time, feat)
    Returns: 
        dict: Indicies to mask Percentage wise the whole test dataset (percentage [# testdata, time * feat])  
    '''
    mask_dic={}
    percentages=[ i for i in range(10,91,10)]
    #print(saliency.shape)
    saliency_= saliency
    saliency_=saliency_.reshape(saliency_.shape[0],-1)
    indexGrid=np.zeros((saliency_.shape[0],saliency_.shape[1],len(percentages)),dtype='object') 
    indexGrid[:,:,:]=np.nan
    #Results in Grid ('Num, Time*Features, Percentages)

    # Goes through saliency of each item
    for i in range(saliency_.shape[0]):
        #Gets per intem indexes for percenatages
        indexes = getIndexOfAllhighestSalientValues(saliency_[i,:],percentages)
        #print(f'indexes {indexes}')
        for l in range(len(indexes)):
            # Goes through percentage wise indexes and annds to index Frid
            indexGrid[i,:len(indexes[l]),l]=indexes[l]
        #print(indexGrid)

        for p,percentage in enumerate(percentages):
            #Fills Index Grid in Mask DIC 
            mask_dic[percentage]=indexGrid[:,:,p]
            #print(f'Mask DIC {percentage} {mask_dic[percentage][0]}')

        #import sys
        #sys.exit(1)
    #print(f'Mask DIC {mask_dic}')
    return mask_dic

def get_masked_accuracy(m,test_loaderRNN,raw_testing,y,masks,scaler,dataGenerationProcess,sampler, shapes):
    '''
    Attributes:
        m : Pytorch classification model 
        test_loaderRNN: Dataloader with batch size 1 
        raw_testing: Plain Data in shape (num samples, time * feat)
        y:
        masks dict: dict of masks percentage: [num samples, time*feat]
        scaler: Sklearn Min-Max Scalter 
        dataGenerationProcess str: Generation Process of Synthetic Data 
        shapes Tupe: shape of input 
    Returns
        dict: (Masking Percentage, Accuracy)
    '''
    maskedPercentages=[ i for i in range(0,101,10)]
    X_DimOfGrid=len(maskedPercentages)
    Grid = np.zeros((X_DimOfGrid),dtype='object')#,Y_DimOfGrid),dtype='object')
    pretrained_model = m 
    Test_Unmasked_Acc  = checkAccuracy(test_loaderRNN , pretrained_model,shapes)
    Test_Masked_Acc=Test_Unmasked_Acc

    for i , maskedPercentage in enumerate(maskedPercentages):
        # Goes Through masked Percentages
        if(maskedPercentage==0):
            Grid[i]=Test_Unmasked_Acc
        elif(Test_Masked_Acc==0):
            Grid[i]=Test_Masked_Acc
        else:
            if(maskedPercentage !=100):
                mask = masks[maskedPercentage]
                toMask=np.copy(raw_testing.reshape(-1,shapes[0]*shapes[1]))
                #print(raw_testing.shape)
                #TODO Before shape , sampler
                MaskedTesting=maskData(toMask,mask,dataGenerationProcess,sampler,shapes,True)
                #print('Masked Testing',MaskedTesting.shape)
                MaskedTesting=scaler.transform(MaskedTesting.reshape(-1,shapes[0]*shapes[1]))
                MaskedTesting=MaskedTesting.reshape(-1,shapes[0],shapes[1])

            else:
                #print(shapes)
                MaskedTesting=np.zeros((raw_testing.shape[0] , shapes[0]*shapes[1]))
                #TODO is this the right shape
                sample = generateNewSample(dataGenerationProcess,NumTimeSteps=shapes[1], NumFeatures=shapes[0]).reshape(shapes[0]*shapes[1])
                MaskedTesting[:,:]= sample

                MaskedTesting=scaler.transform(MaskedTesting)
                MaskedTesting=MaskedTesting.reshape(-1,shapes[0]*shapes[1])
                #if(plot):

                #    randomIndex = 10
                #    plotExampleBox(MaskedTesting[randomIndex],args.Graph_dir+args.DataName+"_"+models[m]+"_"+saliency+"_percentMasked"+str(maskedPercentage),flip=True)

            Maskedtest_dataRNN = data_utils.TensorDataset(torch.from_numpy(MaskedTesting),torch.from_numpy(y))
            Maskedtest_loaderRNN = data_utils.DataLoader(Maskedtest_dataRNN, batch_size=1, shuffle=False)

            Test_Masked_Acc  =   checkAccuracy(Maskedtest_loaderRNN , pretrained_model,shapes)
            Grid[i]=Test_Masked_Acc
    return Grid


def get_ground_truth_mask(raw_testing,metadata, shapes):
    TestingLabel=metadata[:,0]
    TargetTS_Starts=metadata[:,1]
    TargetTS_Ends=metadata[:,2]
    TargetFeat_Starts= metadata[:,3]
    TargetFeat_Ends= metadata[:,4]
    referencesSamples=np.zeros((raw_testing.shape[0],shapes[0],shapes[1]))
    referenceIndxAll=np.zeros((raw_testing.shape[0],shapes[0]*shapes[1]))
    referenceIndxAll[:,:]=np.nan

    for i in range(TestingLabel.shape[0]):
        # This Section Calculates the Grount Truth Masks
        # Iterate through all testing data 
        referencesSamples[i,int(TargetTS_Starts[i]):int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]):int(TargetFeat_Ends[i])]=1
        numberOfImpFeatures=int(np.sum(referencesSamples[i,:,:]))
        ind = getIndexOfXhighestFeatures(referencesSamples[i,:,:].flatten() , numberOfImpFeatures)
        #print(f'ind {ind}')
        referenceIndxAll[i,:ind.shape[0]]=ind
    return referenceIndxAll

#def get_precision_recall_CF(sal,raw_testing,metadata,masks,shapes):
    '''
    Find Spots Wehre CF Was Changed
    #TODO
    '''

    #referenceIndxAll=get_ground_truth_mask(raw_testing,metadata,shapes)
    #sal= sal.reshape(-1, shapes[0], shapes[1])
    #raw_testing=raw_testing.reshape(-1, shapes[0], shapes[1])
    #index=np.where(sal!=raw_testing,axis=1)
    
    #precision=[]
    #recall=[]
    #for i in sal.shape[0]:
    #    TP=0
    #    FP=0
    #    FN=0
            
    #    for j in range(postiveWithTrue.shape[0]):
    #        if(postiveWithTrue[j]):
    #            #In postive and true so true postive
                    
    #            TP+=saliencyValues[i,postiveIndex[j]]
    #            countTP+=1
    #        else:
    #            #In postive but not true so false postive
    #            FP+=saliencyValues[i,postiveIndex[j]]
    #            countFP+=1
    #    for j in range(TrueWithpostive.shape[0]):
    #        if(not TrueWithpostive[j]):
    #            #In true but not in postive False negtive
    #            FN+=saliencyValues[i,trueIndex[j]]
    #            countFN+=1#


    #    if((TP+FP)>0):
    #        examplePrecision=TP/(TP+FP)
    #        Pcount+=1
    #    else:
    #        examplePrecision=0
    #    if((TP+FN)>0):
     #       exampleRecall=TP/(TP+FN)
    #        Rcout+=1
    #    else:
    #        exampleRecall=0
    #   overallPrecision+=examplePrecision
    #    overallRecall+=exampleRecall
    #        #if Pcount != 0 or Rcout!=0:
    #            #TODO When does this Occur ? 
    #overallPrecision=overallPrecision/Pcount
    #overallRecall=overallRecall/Rcout
    #precision.append(overallPrecision)
    #recall.append(overallRecall)
    #return precision, recall

def get_reference_samples(metadata,raw_testing,shapes):
    TestingLabel=metadata[:,0]
    TargetTS_Starts=metadata[:,1]
    TargetTS_Ends=metadata[:,2]
    TargetFeat_Starts= metadata[:,3]
    TargetFeat_Ends= metadata[:,4]
    referencesSamples=np.zeros((raw_testing.shape[0],shapes[0],shapes[1]))
    for i in range(TestingLabel.shape[0]):

        referencesSamples[i,int(TargetTS_Starts[i]):int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]):int(TargetFeat_Ends[i])]=1
    

    return referencesSamples

def get_quantus_metrics(meta, sal,data,label,mod,*shape):

    masks=get_reference_samples(meta,data,shape)
                
    #sal=np.array(sal).reshape(data.shape[0],*shape)
    #data=np.array(data).reshape(data.shape[0],*shape)
    masks.reshape(sal.shape)
    pointing=quantus.PointingGame()(model=mod,x_batch=data, y_batch=label,  a_batch=sal, s_batch=masks, device='CPU')
    rank=quantus.RelevanceRankAccuracy()(model=mod,x_batch=data, y_batch=label,  a_batch=sal, s_batch=masks, device='CPU')
    Auc=quantus.AUC()(model=mod,x_batch=data, y_batch=label,  a_batch=sal, s_batch=masks, device='CPU')
    return pd.DataFrame([pointing,rank, Auc],columns=["Poinint", "Relevance Rank","AuC"])
    

def get_precision_recall_plain(sal,raw_testing,metadata,shapes, tolerance=0):
    '''
    Plain Calulation of Recall and Precision. If tolerance = 0, Precision and Recall are Identical. 
    Without Mask Check if Top XXXX
    '''
    sal=np.array(sal)
   
    #referenceIndxAll=get_ground_truth_mask(raw_testing,metadata,shapes)
    TestingLabel=metadata[:,0]
    TargetTS_Starts=metadata[:,1]
    TargetTS_Ends=metadata[:,2]
    TargetFeat_Starts= metadata[:,3]
    TargetFeat_Ends= metadata[:,4]
    referencesSamples=np.zeros((raw_testing.shape[0],shapes[0],shapes[1]))
    referenceIndxAll={}
    

    for i in range(TestingLabel.shape[0]):

        referencesSamples[i,int(TargetTS_Starts[i]):int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]):int(TargetFeat_Ends[i])]=1

        referenceIndxAll[i]=np.where(referencesSamples[i]==1)[0]
    #print(referenceIndxAll[0])             
    saliencyValues= np.array(sal)
    saliencyValues=saliencyValues.reshape(-1,shapes[0]*shapes[1])
   
    Rcout=0
    Pcount=0
    overallPrecision=0
    overallAcc=0
    overallRecall=0
    recall=[]
    precision=[]
    acc=[]
    # TODO User Reference Samples
    for i in range(sal.shape[0]):
        #Get Index of X Highest Feature of saliency Values
        postiveIndex = getIndexOfXhighestFeatures(saliencyValues[i],int(len(referenceIndxAll[i])*(1+tolerance)))
        #print(saliencyValues)
        #print(postiveIndex)
        #print(f'postiveIndex {postiveIndex}')

        positiveSamples=np.zeros_like(saliencyValues[i])
        np.put(positiveSamples,postiveIndex,np.ones(len(postiveIndex)))#[postiveIndex]==1
        positiveSamples=(positiveSamples==1)
        #print(postiveIndex)
        #print('Positive Samples',positiveSamples)
        #print('xx' ,len(np.where(positiveSamples==1)[0]))
        sample=referencesSamples[i].reshape(-1)
        trueSamples= (sample==1)
       # print('True Samples',trueSamples)


        # Get Indicies of  GT

        #reference_sample
        trueIndex=referenceIndxAll[i][:]
        trueIndex=trueIndex[np.logical_not(pd.isna(trueIndex))]
        trueIndex = trueIndex.astype(np.int64)
        
        #print(f'{trueIndex}')

        # get FT False iNdex: 
        falseIndex = range(0, len(saliencyValues[0]))
        falseIndex=np.isin(falseIndex,trueIndex, invert=True)
        #print(f'fales Index {falseIndex}')

        negativeIndex = range(0, len(saliencyValues[0]))
        negativeIndex=np.isin(negativeIndex,postiveIndex, invert=True)
        #print(f'negative Index {negativeIndex}')

        TP=0
        FP=0
        FN=0
        TN=0
        #TP_false=0
        #FP_false=0
        #FN_false=0
            
        #for j in range(postiveWithTrue.shape[0]):
        #    # for j in high saliency values in GT
        #    if(postiveWithTrue[j]):
        #        #In postive and true so true postive
        #        TP+=1#saliencyValues[i,postiveIndex[j]]
        #        #print(f'TP  {TP}')
        #        countTP+=1
        #    else:
        #        #In postive but not true so false postiv
        #        FP+=1#saliencyValues[i,postiveIndex[j]]
        #        #print(f'FP  {FP}')
        #        countFP+=1
        #for j in range(TrueWithpostive.shape[0]):
        #    if(not TrueWithpostive[j]):
        #        #In true but not in postive False negtive
        #            FN+=1#saliencyValues[i,trueIndex[j]]
        #            #print(f'FN  {FN}')
        #            countFN+=1
        # TN Not needed for calc ?
        for j in range(positiveSamples.shape[0]):
            #print(j)
            #print('Sample',positiveSamples[j])
            #print('GT',trueSamples[j])
            if(positiveSamples[j] and trueSamples[j]):
                #print(TP)
                TP+=1
            elif(positiveSamples[j] and not trueSamples[j]): 
                #print('FP')
                FP +=1
            elif(not positiveSamples[j] and trueSamples[j]): 
                #print('FN')
                FN +=1
            else: 
                TN+=1
        '''
        THiS iS FALSE Case
        for j in range(negativeIndex.shape[0]):
                if(negativeIndex[j] and falseIndex[j]):
                    #In postive and true so true postive
                    TP_false+=1#saliencyValues[i,postiveIndex[j]]

                else:
                    #In postive but not true so false postiv
                    FP_false+=1#saliencyValues[i,postiveIndex[j]]

        for j in range(negativeIndex.shape[0]):
            #if j not in referenceIndxAll[i][:]:
                if(not negativeIndex[j]):
                    #In true but not in postive False negtive
                    FN_false+=1#saliencyValues[i,trueIndex[j]]
        print(f'TP {TP_false}')
        print(f'FP {FP_false}')
        print(f'FN {FN_false}')
        '''
        if((TP+FP)>0):
            examplePrecision=TP/(TP+FP)
            Pcount+=1
        else:
            examplePrecision=0
        if((TP+FN)>0):
            exampleRecall=TP/(TP+FN)
            Rcout+=1
        else:
            exampleRecall=0

        overallPrecision+=examplePrecision
        overallRecall+=exampleRecall
        overallAcc += (TP+TN)/len(saliencyValues[0])
        recall.append(exampleRecall)
        precision.append(examplePrecision)
        acc.append((TP+TN)/len(saliencyValues[0]))
        #print(f'Precision {examplePrecision}, Recall {exampleRecall}, ACC {(TP+TN)/len(referenceIndxAll[i])}')

    if Pcount != 0:
        overallPrecision=overallPrecision/Pcount
    else: 
        overallPrecision=0
    if Rcout !=0:
        overallRecall=overallRecall/Rcout
    else: 
        overallRecall=0
    overallAcc=overallAcc/len(saliencyValues)
    df= pd.DataFrame([])
    df['Precision']=precision
    df['Recall']=recall
    df['Accuracy']=acc
    return overallPrecision, overallRecall,overallAcc,df

def get_precision_recall(sal,raw_testing,metadata,masks,shapes):
    '''
    Calculates Precision / Recall for different Tolerance Specification. 
    Attributes: 
        sal np.array: saliency method shaped as (Num Items, time, fetura) 
        raw_testing np.array: Data for testing shapes as (Num Items, Num Features * Num Times)
        metadata np.array: meta data / Ground truth 
        masks np.array: masks shaped as (Num Items, Num Features * Num Times)
        shapes Tuple: shape 
    '''

    maskedPercentages=[ i for i in range(0,101,10)]
    referenceIndxAll=get_ground_truth_mask(raw_testing,metadata,shapes)
                    
    precision=[]
    recall=[]
    saliencyValues= np.array(sal)
    saliencyValues=saliencyValues.reshape(-1,shapes[0]*shapes[1])
    # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for maskNumber in range(0,100,10):
        # Go through masks 
        overallRecall=0
        overallPrecision=0

        if(maskNumber !=100 and maskNumber !=0 ):
            mask = masks[maskNumber]
            #print(mask)     
            Rcout=0
            Pcount=0

            for i in range(mask.shape[0]):
                postiveIndex = mask[i,:]
                postiveIndex=postiveIndex[np.logical_not(pd.isna(postiveIndex))]
                postiveIndex = postiveIndex.astype(np.int64)

                trueIndex=referenceIndxAll[i,:]
                trueIndex=trueIndex[np.logical_not(pd.isna(trueIndex))]
                trueIndex = trueIndex.astype(np.int64)

                postiveWithTrue = np.isin(postiveIndex, trueIndex)
                TrueWithpostive = np.isin(trueIndex, postiveIndex)

                countTP=0
                countFP=0
                countFN=0

                TP=0
                FP=0
                FN=0
            
                for j in range(postiveWithTrue.shape[0]):
                    if(postiveWithTrue[j]):
                        #In postive and true so true postive
                        TP+=saliencyValues[i,postiveIndex[j]]
                        countTP+=1
                    else:
                        #In postive but not true so false postive
                        FP+=saliencyValues[i,postiveIndex[j]]
                        countFP+=1
                for j in range(TrueWithpostive.shape[0]):
                    if(not TrueWithpostive[j]):
                        #In true but not in postive False negtive
                        FN+=saliencyValues[i,trueIndex[j]]
                        countFN+=1


                if((TP+FP)>0):
                    examplePrecision=TP/(TP+FP)
                    Pcount+=1
                else:
                    examplePrecision=0
                if((TP+FN)>0):
                    exampleRecall=TP/(TP+FN)
                    Rcout+=1
                else:
                    exampleRecall=0

                overallPrecision+=examplePrecision
                overallRecall+=exampleRecall
            #if Pcount != 0 or Rcout!=0:
            #TODO When does this Occur ? 
            if Pcount!= 0:
                overallPrecision=overallPrecision/Pcount
            else:
                precision.append(np.nan)
            if Rcout != 0:
                overallRecall=overallRecall/Rcout
            else: 
                recall.append(np.nan)
            precision.append(overallPrecision)
            recall.append(overallRecall)
        else:
            precision.append(np.nan)
            recall.append(np.nan)
    return precision, recall

def get_accuracy_metrics(precision, recall, accuracy): 
    precision_row=[]
    recall_row=[]
    a=[]
    b=0.1
    for j in range(0, len(precision)):
        if(not np.isnan(precision[j])):
            precision_row.append(precision[j])
            a.append(b)
            b+=0.1

    AUP= np.trapz( precision_row,x=a)
    a=[]
    b=0.1
    for j in range(0,len(recall)):
        if(not np.isnan(recall[j])):
            recall_row.append(recall[j])
            a.append(b)
            b+=0.1                 
    
    AUR= np.trapz(recall_row,x=a)
    index_ = np.argsort(recall_row)
    precision_row = np.array(precision_row)
    recall_row = np.array(recall_row)
    AUPR=np.trapz(precision_row[index_],x=recall_row[index_])
    accuracy_row=[]
    a=[]
    b=0.1
    for j in range(0,len(accuracy)):
        if(not np.isnan(accuracy[j])):
            accuracy_row.append(accuracy[j])
            a.append(b)
            b+=0.1
    AUC= np.trapz(accuracy_row,x=a)

    return AUP, AUR, AUPR, AUC


#def get_feature_time_spec(sal,metadata,raw_data,masks,shapes):
#    maskedPercentages=[ i for i in range(0,101,10)]


#    TestingLabel=metadata[:,0]
#    TargetTS_Starts=metadata[:,1]
#    TargetTS_Ends=metadata[:,2]
#    TargetFeat_Starts= metadata[:,3]
#    TargetFeat_Ends= metadata[:,4]

#    referencesSamples=np.zeros((raw_data.shape[0],shapes[0],shapes[1]))
#    referenceIndxAll=np.zeros((raw_data.shape[0],shapes[0]*shapes[1]))
#    referenceIndxAll[:,:]=np.nan


#    for i in range(TestingLabel.shape[0]):

#        referencesSamples[i,int(TargetTS_Starts[i]):int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]):int(TargetFeat_Ends[i])]=1
#        numberOfImpFeatures=int(np.sum(referencesSamples[i,:,:]))
#        ind = getIndexOfXhighestFeatures(referencesSamples[i,:,:].flatten() , numberOfImpFeatures)
#        referenceIndxAll[i,:ind.shape[0]]=ind
        
#    referenceIndxAll_Time,referenceIndxAll_Features=getRowColMaskIndex(referenceIndxAll)

#    precision_time=np.zeros((len(maskedPercentages)),dtype=object)
#    recall_time=np.copy(precision_time)
#    precision_features=np.copy(precision_time)
#    recall_features=np.copy(precision_time)



 #   timePrecision=[]
 #   timeRecall=[]

  #  featurePrecision=[]
  #  featureRecall=[]
  #  saliencyValues= sal#


#    saliencyValues_time_AverageOfMaxX=np.zeros((saliencyValues.shape[0],shapes[1]))
#    saliencyValues_feature_AverageOfMaxX=np.zeros((saliencyValues.shape[0],shapes[0]))

#    for d in range(saliencyValues.shape[0]):
#        for r in range (shapes[1]):
#            saliencyValues_time_AverageOfMaxX[d,r]=getAverageOfMaxX(saliencyValues[d,r,:],int(TargetTS_Ends[d]-TargetTS_Starts[d]))

                                
#        for c in range(shapes[0]):
#            saliencyValues_feature_AverageOfMaxX[d,c]=getAverageOfMaxX(saliencyValues[d,:,c],int(TargetFeat_Ends[d]-TargetFeat_Starts[d]))

#    for maskNumber in range(0,100,10):
#        timeOverallRecall=0
#        timeOverallPrecision=0

#        featureOverallRecall=0
#        featureOverallPrecision=0

#        if(maskNumber !=100 and maskNumber !=0 ):
#            mask = masks[maskNumber]

#            maskTime,maskFeatures=getRowColMaskIndex(mask,shapes[0],shapes[1])

#            TimeRcout=0
#            TimePcount=0
#            FeatureRcout=0
#            FeaturePcount=0
#            for i in range(mask.shape[0]):
#                TP=0
#                FP=0
#                FN=0
#                                    
#                for t in range(shapes[1]):
#                    if(referenceIndxAll_Time[i,t] and maskTime[i,t]):
#                        TP+=saliencyValues_time_AverageOfMaxX[i,t]
#                    elif((not referenceIndxAll_Time[i,t]) and maskTime[i,t]):
#                        FP+=saliencyValues_time_AverageOfMaxX[i,t]
#                    elif(referenceIndxAll_Time[i,t] and  (not maskTime[i,t])):
#                        FN+=saliencyValues_time_AverageOfMaxX[i,t]



#                if((TP+FP)>0):
#                    timeExamplePrecision=TP/(TP+FP)
#                    TimePcount+=1
#                else:
#                    timeExamplePrecision=0
#                if((TP+FN)>0):
#                    timeExampleRecall=TP/(TP+FN)
#                    TimeRcout+=1
#                else:
#                    timeExampleRecall=0#
#
#                timeOverallPrecision+=timeExamplePrecision
#                timeOverallRecall+=timeExampleRecall
#
#                TP=0
#                FP=0
#                FN=0
#                #For Feature
#                for f in range(shapes[0]):
#                    if(referenceIndxAll_Features[i,f] and maskFeatures[i,f]):
#                        TP+=saliencyValues_feature_AverageOfMaxX[i,f]
#                    elif((not referenceIndxAll_Features[i,f]) and maskFeatures[i,f]):
#                        FP+=saliencyValues_feature_AverageOfMaxX[i,f]
#                    elif(referenceIndxAll_Features[i,f] and  (not maskFeatures[i,f])):
#                        FN+=saliencyValues_feature_AverageOfMaxX[i,f]
#
#                if((TP+FP)>0):
#                    featureExamplePrecision=TP/(TP+FP)
#                    FeaturePcount+=1
#                else:
#                    featureExamplePrecision=0
#                if((TP+FN)>0):
#                    featureExampleRecall=TP/(TP+FN)
#                    FeatureRcout+=1
#                else:
#                    featureExampleRecall=0#

#                featureOverallPrecision+=featureExamplePrecision
#                featureOverallRecall+=featureExampleRecall




#            timeOverallPrecision=timeOverallPrecision/TimePcount
#            timeOverallRecall=timeOverallRecall/TimeRcout
#            timePrecision.append(timeOverallPrecision)
#            timeRecall.append(timeOverallRecall)
#            featureOverallPrecision=featureOverallPrecision/FeaturePcount
#            featureOverallRecall=featureOverallRecall/FeatureRcout
#            featurePrecision.append(featureOverallPrecision)
#            featureRecall.append(featureOverallRecall)
#        else: 
#            featurePrecision.append(np.nan)
#            featureRecall.append(np.nan)
#            timePrecision.append(np.nan)
#            timeRecall.append(np.nan)
#    precision_time[1:]=timePrecision
#    recall_time[1:]=timeRecall

#    precision_features[1:]=featurePrecision
#    recall_features[1:]=featureRecall

#    return precision_time, recall_time, precision_features,recall_features

#def get_accuracy_metric_feature_time(time_precision, time_recall,feature_precision, feature_recall ):
#    a=[]
#    b=0.1
#    time_precision_row=[]
#    time_recall_row=[]
#    feature_precision_row=[]
#    feature_recall_row=[]
#    for j in range(0,len(time_precision)):
#        if(not np.isnan(time_precision[j])):
#                            
#            time_precision_row.append(time_precision[j])
#            time_recall_row.append(time_recall[j])

                
#            feature_precision_row.append(feature_precision[j])
#            feature_recall_row.append(feature_recall[j])


                               
#            a.append(b)
#            b+=0.1

#    time_AUP= np.trapz(time_precision_row,x=a)
#    time_AUR= np.trapz(time_recall_row,x=a)
#    index_ = np.argsort(time_recall_row)
#    time_precision_row = np.array(time_precision_row)
#    time_recall_row = np.array(time_recall_row)
#    time_AUPR=np.trapz(time_precision_row[index_],x=time_recall_row[index_])#

#    feature_AUP= np.trapz(feature_precision_row,x=a)
#    feature_AUR= np.trapz(feature_recall_row,x=a)
#    index_ = np.argsort(feature_recall_row)
#    feature_precision_row = np.array(feature_precision_row)
#    feature_recall_row = np.array(feature_recall_row)
#    feature_AUPR=np.trapz(feature_precision_row[index_],x=feature_recall_row[index_])

 #   return time_AUP,time_AUR,time_AUPR, feature_AUP,feature_AUR,feature_AUPR
